"""NVIDIA official ADE / minADE / box-corner contracts.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #17/#18/#22
minADE layout tests on the MLX eval helper — this is the CUDA metric path.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def _load_metrics():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.geometry.coordinates import xyzrot_to_corners
        from alpamayo_r1.metrics.distance_metrics import compute_ade, compute_minade
        from alpamayo_r1.metrics.metric_utils import apply_prefix, summarize_metric
    except Exception as exc:
        pytest.skip(f"NVIDIA metrics unavailable: {exc}")
    return xyzrot_to_corners, compute_ade, compute_minade, apply_prefix, summarize_metric


def test_compute_ade_is_xy_only_and_truncates_horizon(torch):
    _, compute_ade, _, _, _ = _load_metrics()
    # Pred matches GT in XY; Z is a large mismatch that must not score.
    gt = torch.zeros(1, 4, 3)
    gt[0, :, 0] = torch.arange(4, dtype=torch.float32)
    pred = torch.zeros(1, 1, 2, 4, 3)
    pred[0, 0, 0] = gt[0]
    pred[0, 0, 0, :, 2] = 99.0
    pred[0, 0, 1] = gt[0]
    pred[0, 0, 1, :, 0] += 2.0
    ade = compute_ade(pred, gt, only_xy=True)
    assert ade.shape == (1, 1, 2)
    assert abs(float(ade[0, 0, 0].item()) - 0.0) < 1e-6
    assert abs(float(ade[0, 0, 1].item()) - 2.0) < 1e-6

    xyz = compute_ade(pred, gt, only_xy=False)
    assert float(xyz[0, 0, 0].item()) > 10.0

    short = compute_ade(pred, gt, timestep_horizon=2, only_xy=True)
    assert abs(float(short[0, 0, 1].item()) - 2.0) < 1e-6
    try:
        compute_ade(pred, gt, timestep_horizon=5)
    except ValueError as exc:
        assert "timestep_horizon" in str(exc)
    else:
        raise AssertionError("expected ValueError when horizon exceeds T")


def test_compute_minade_picks_best_sample_and_skips_long_horizons(torch):
    _, _, compute_minade, _, _ = _load_metrics()
    gt = torch.zeros(1, 4, 3)
    gt[0, :, 1] = torch.arange(4, dtype=torch.float32)
    pred = torch.zeros(1, 2, 2, 4, 3)
    pred[0, 0, 0] = gt[0]
    pred[0, 0, 1, :, 1] = torch.arange(4, dtype=torch.float32) + 4.0
    pred[0, 1, 0, :, 1] = torch.arange(4, dtype=torch.float32) + 1.0
    pred[0, 1, 1] = gt[0]
    out = compute_minade(
        pred,
        gt,
        disable_summary=True,
        timestep_horizons=[2, 4, 50],
        time_step=0.1,
    )
    assert set(out) >= {"min_ade", "min_ade/by_t=0.2", "min_ade/by_t=0.4"}
    assert "min_ade/by_t=5.0" not in out
    # Best-of-K per group is the matching sample; summarize then averages groups.
    assert torch.allclose(out["min_ade"], torch.zeros(1), atol=1e-6)
    assert "min_ade_std" not in out


def test_summarize_metric_adds_std_only_for_multi_group(torch):
    _, _, _, apply_prefix, summarize_metric = _load_metrics()
    one = summarize_metric({"min_ade": torch.tensor([[1.0]], dtype=torch.float32)})
    assert set(one) == {"min_ade"}
    assert float(one["min_ade"].item()) == 1.0

    two = summarize_metric({"min_ade": torch.tensor([[1.0, 3.0]], dtype=torch.float32)})
    assert set(two) == {"min_ade", "min_ade_std"}
    assert abs(float(two["min_ade"].item()) - 2.0) < 1e-6
    assert abs(float(two["min_ade_std"].item()) - 1.0) < 1e-6

    try:
        summarize_metric({"min_ade": torch.tensor([1.0, 2.0])})
    except ValueError as exc:
        assert "[B,N]" in str(exc)
    else:
        raise AssertionError("expected ValueError for a rank-1 metric")

    prefixed = apply_prefix("eval/", {"ade": torch.tensor([0.5])})
    assert set(prefixed) == {"eval/ade"}


def test_xyzrot_to_corners_identity_then_translate(torch):
    xyzrot_to_corners, _, _, _, _ = _load_metrics()
    dims = torch.tensor([2.0, 4.0, 2.0])
    rot = torch.eye(3)
    xyz = torch.tensor([10.0, 0.0, 0.0])
    corners = xyzrot_to_corners(xyz, rot, dims)
    assert corners.shape == (8, 3)
    # Length along X is 2 → ±1 around the center x=10.
    xs = sorted({round(float(v), 5) for v in corners[:, 0].tolist()})
    assert xs == [9.0, 11.0]
    assert torch.allclose(corners.mean(dim=0), xyz, atol=1e-6)
