"""Comfort yaw is first-column SO(3), not the delta-tokenizer first row.

Weight-free. Complementary to open #26 (delta ``atan2(R[0,1], R[0,0])``
= −yaw) and #28 (body-frame lon/lat). Comfort uses
``atan2(R[...,1,0], R[...,0,0])``. A first-row swap flips yaw sign and
then every lon/lat comfort gate.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_FINETUNE = _REPO / "finetune"
_COMFORT = _FINETUNE / "rl" / "rewards" / "comfort_reward.py"
_DELTA = _SRC / "alpamayo_r1" / "models" / "delta_tokenizer.py"


def _ensure_paths() -> None:
    for path in (str(_REPO), str(_SRC), str(_FINETUNE)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_gather_dynamics():
    _ensure_paths()
    pytest.importorskip("torch")
    try:
        from rl.rewards.comfort_reward import gather_dynamics

        return gather_dynamics
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_comfort_heading", _COMFORT)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_COMFORT}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            pytest.skip(f"NVIDIA comfort_reward unavailable: {exc}")
        return module.gather_dynamics


def test_source_comfort_uses_first_column_delta_uses_first_row():
    comfort = _COMFORT.read_text()
    delta = _DELTA.read_text()
    assert "torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])" in comfort
    assert "torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])" in delta


def test_plus_90_yaw_is_positive_not_negative():
    torch = pytest.importorskip("torch")
    gather_dynamics = _load_gather_dynamics()
    # Yaw +90°: +X body → +Y world. First column is (0, 1, 0); first row is (0, -1, 0).
    yaw = math.pi / 2
    rot = torch.tensor(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    pred_rot = rot.view(1, 1, 1, 1, 3, 3).expand(1, 1, 1, 4, 3, 3).clone()
    pred_xyz = torch.zeros(1, 1, 1, 4, 3)
    # Stationary pose: heading must still be +π/2 from the first column.
    heading = torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])
    row_heading = torch.atan2(pred_rot[..., 0, 1], pred_rot[..., 0, 0])
    torch.testing.assert_close(heading, torch.full_like(heading, math.pi / 2))
    torch.testing.assert_close(row_heading, torch.full_like(row_heading, -math.pi / 2))
    assert not torch.allclose(heading, row_heading)

    dyn = gather_dynamics(pred_xyz, pred_rot)
    torch.testing.assert_close(dyn["ego_yaw_rate"], torch.zeros_like(dyn["ego_yaw_rate"]))
    # Constant +X world speed at +90° heading is pure lateral, so lon accel stays 0.
    moving = torch.zeros(1, 1, 1, 4, 3)
    moving[..., 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dyn = gather_dynamics(moving, pred_rot)
    torch.testing.assert_close(dyn["ego_dv_lon"], torch.zeros_like(dyn["ego_dv_lon"]))
    # First-row heading (−90°) would call the same +X step longitudinal.
    first_row_lon_vel = 10.0 * torch.cos(row_heading)
    assert torch.all(first_row_lon_vel < 0)
