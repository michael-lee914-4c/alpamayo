"""NVIDIA comfort finite-diff holds the last Δ (and yaw wrap) to T.

Weight-free. Complementary to open #22/#25 (yaw wrap, exclusive bounds,
violent-traj fail) — this locks the hold-last pad that keeps dynamics
length == T, and the PLANNING_FREQ=10 scale used by every comfort term.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FINETUNE = _REPO / "finetune"


def _load_comfort():
    path = _FINETUNE / "rl" / "rewards" / "comfort_reward.py"
    spec = importlib.util.spec_from_file_location("nv_comfort_hold_last", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA comfort_reward unavailable: {exc}")
    return module


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_diff_holds_last_delta_and_scales_by_10hz(torch):
    comfort = _load_comfort()
    # Positions 0,1,3 → raw Δ = [1, 2]; hold-last → [1, 2, 2]; ×10 Hz.
    x = torch.tensor([[[[[0.0, 1.0, 3.0]]]]], dtype=torch.float32)
    got = comfort._diff(x)
    assert got.shape == x.shape
    assert torch.allclose(got, torch.tensor([[[[[10.0, 20.0, 20.0]]]]]))


def test_diff_yaw_holds_last_and_unwraps(torch):
    comfort = _load_comfort()
    # π-0.1 then −π+0.1 is a +0.2 step, not a −2π drop. Hold-last copies it.
    yaw = torch.tensor([[[[[torch.pi - 0.1, -torch.pi + 0.1]]]]], dtype=torch.float32)
    got = comfort._diff_yaw(yaw)
    assert got.shape == yaw.shape
    assert float(got[..., 0].item()) == pytest.approx(2.0, abs=1e-5)
    assert float(got[..., 1].item()) == pytest.approx(2.0, abs=1e-5)


def test_gather_dynamics_keeps_time_axis(torch):
    comfort = _load_comfort()
    T = 5
    xyz = torch.zeros(1, 1, 1, T, 3)
    xyz[0, 0, 0, :, 0] = torch.linspace(0.0, 0.4, T)
    rot = torch.eye(3).expand(1, 1, 1, T, 3, 3).clone()
    dyn = comfort.gather_dynamics(xyz, rot)
    for key in (
        "ego_yaw_rate",
        "ego_dv",
        "ego_jerk",
        "ego_yaw_accel",
        "ego_dv_lon",
        "ego_jerk_lon",
        "ego_dv_lat",
    ):
        assert dyn[key].shape[-1] == T
    # Constant +X speed 1.0 m/s at 10 Hz (0.1 m / 0.1 s). Accel/jerk hold 0.
    assert torch.allclose(dyn["ego_dv_lon"], torch.zeros(1, 1, 1, T), atol=1e-5)
    assert torch.allclose(dyn["ego_dv_lat"], torch.zeros(1, 1, 1, T), atol=1e-5)
