"""NVIDIA DistanceMetrics.evaluate: last-group GT, dummy-logprob ADE, 3s gate.

Weight-free. Complementary to open #24/#25, which locked the ADE / minADE /
grouped-corner primitives — not this wrapper's last-traj-group selection,
zero-logprob sample pick, 3.0s horizon gate, or prefix rewrite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _stub_nv_logging() -> None:
    if "alpamayo_r1.common.logging" in sys.modules:
        return
    logging_mod = ModuleType("alpamayo_r1.common.logging")

    class RankedLogger:
        def __init__(self, *args, **kwargs):
            pass

        def setLevel(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

    logging_mod.RankedLogger = RankedLogger
    sys.modules["alpamayo_r1.common.logging"] = logging_mod


def _load_distance_metrics():
    _ensure_src_on_path()
    _stub_nv_logging()
    path = _SRC / "alpamayo_r1" / "metrics" / "metric_api.py"
    spec = importlib.util.spec_from_file_location("nv_metric_api", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA DistanceMetrics unavailable: {exc}")
    return module


def _eye_rot(torch, *lead):
    return torch.eye(3).expand(*lead, 3, 3).clone()


def test_evaluate_returns_empty_without_predictions(torch):
    mod = _load_distance_metrics()
    metric = mod.DistanceMetrics(prefix="eval/", time_step=0.1)
    out = metric.evaluate(None, {}, {})
    assert out == {}


def test_evaluate_uses_last_traj_group_and_zero_logprob_sample0(torch):
    mod = _load_distance_metrics()
    metric = mod.DistanceMetrics(prefix="", time_step=0.1)
    T = 8
    gt_last = torch.stack(
        [torch.linspace(0.0, 1.0, T), torch.zeros(T), torch.zeros(T)], dim=-1
    ).unsqueeze(0)
    gt_first = gt_last + 10.0
    # [B=1, n_traj=2, T, 3] — a silent first-group pick would report ADE ≈ 10.
    data = {
        "ego_future_xyz": torch.stack([gt_first, gt_last], dim=1),
        "ego_future_rot": _eye_rot(torch, 1, 2, T),
    }
    # K=2: sample-0 matches last-group GT; sample-1 is offset.
    pred0 = gt_last[:, None, None, :, :]
    pred1 = pred0 + 5.0
    pred_xyz = torch.cat([pred0, pred1], dim=2)
    output = {
        "pred_xyz": pred_xyz,
        "pred_rot": _eye_rot(torch, 1, 1, 2, T),
    }
    got = metric.evaluate(None, data, output)
    assert "min_ade" in got
    assert float(got["min_ade"].item()) == pytest.approx(0.0, abs=1e-5)
    assert float(got["ade"].item()) == pytest.approx(0.0, abs=1e-5)
    assert "sample_ade" in output
    assert output["sample_ade"].shape[-1] == 2


def test_evaluate_3s_horizon_and_prefix(torch):
    mod = _load_distance_metrics()
    short = mod.DistanceMetrics(prefix="eval/", time_step=0.1)
    T_short = 20
    xyz = torch.zeros(1, 1, 1, T_short, 3)
    data = {
        "ego_future_xyz": torch.zeros(1, 1, T_short, 3),
        "ego_future_rot": _eye_rot(torch, 1, 1, T_short),
    }
    output = {"eval/pred_xyz": xyz, "eval/pred_rot": _eye_rot(torch, 1, 1, 1, T_short)}
    got_short = short.evaluate(None, data, output)
    assert "eval/ade" in got_short
    assert "eval/min_ade" in got_short
    assert "eval/ade/by_t=3.0" not in got_short
    assert "eval/sample_ade" in output

    long = mod.DistanceMetrics(prefix="", time_step=0.1)
    T_long = 30
    data_long = {
        "ego_future_xyz": torch.zeros(1, 1, T_long, 3),
        "ego_future_rot": _eye_rot(torch, 1, 1, T_long),
    }
    output_long = {
        "pred_xyz": torch.zeros(1, 1, 1, T_long, 3),
        "pred_rot": _eye_rot(torch, 1, 1, 1, T_long),
    }
    got_long = long.evaluate(None, data_long, output_long)
    assert "ade/by_t=3.0" in got_long
    assert float(got_long["ade/by_t=3.0"].item()) == pytest.approx(0.0, abs=1e-6)
