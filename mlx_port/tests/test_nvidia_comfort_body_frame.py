"""Comfort lon/lat are body-frame (heading), not world XY.

Weight-free. Complementary to open #27 hold-last (identity heading, +X
coast → both accels 0) — this locks the projection that turns a +Y
world accel into lat at heading 0 and lon at heading π/2.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FINETUNE = _REPO / "finetune"


def _load_comfort():
    path = _FINETUNE / "rl" / "rewards" / "comfort_reward.py"
    spec = importlib.util.spec_from_file_location("nv_comfort_body_frame", path)
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


def _yaw_R(torch, yaw: float):
    c, s = float(torch.cos(torch.tensor(yaw))), float(torch.sin(torch.tensor(yaw)))
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )


def _accel_along_y(torch, T: int = 5):
    # y = 0.05 t^2  → world +Y accel ≈ 1 m/s² at 10 Hz after the hold-last scale.
    t = torch.arange(T, dtype=torch.float32)
    xyz = torch.zeros(1, 1, 1, T, 3)
    xyz[0, 0, 0, :, 1] = 0.05 * t * t
    return xyz


def test_plus_y_accel_is_lat_at_heading_zero(torch):
    comfort = _load_comfort()
    xyz = _accel_along_y(torch)
    rot = _yaw_R(torch, 0.0).expand(1, 1, 1, xyz.shape[-2], 3, 3).clone()
    dyn = comfort.gather_dynamics(xyz, rot)
    # World +Y with heading 0 → lateral, not longitudinal.
    assert dyn["ego_dv_lat"].abs().mean() > dyn["ego_dv_lon"].abs().mean()
    assert dyn["ego_dv_lat"].abs().max() > 0.5
    assert dyn["ego_dv_lon"].abs().max() < 0.2


def test_plus_y_accel_is_lon_at_heading_pi_over_2(torch):
    comfort = _load_comfort()
    xyz = _accel_along_y(torch)
    rot = _yaw_R(torch, float(torch.pi / 2)).expand(1, 1, 1, xyz.shape[-2], 3, 3).clone()
    dyn = comfort.gather_dynamics(xyz, rot)
    assert dyn["ego_dv_lon"].abs().mean() > dyn["ego_dv_lat"].abs().mean()
    assert dyn["ego_dv_lon"].abs().max() > 0.5
    assert dyn["ego_dv_lat"].abs().max() < 0.2
