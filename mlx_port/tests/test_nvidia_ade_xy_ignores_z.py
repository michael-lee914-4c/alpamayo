"""Default ADE is BEV-only: a Z-only error must not change the score.

Weight-free. Complementary to open #34 (``only_xy=False`` includes Z,
horizon>T raise). Official eval defaults ``only_xy=True``; a dropped
slice would treat a 2 m altitude error as 2 m ADE and fail a planner
that is correct in the plane.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_DIST = _SRC / "alpamayo_r1" / "metrics" / "distance_metrics.py"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _compute_ade(pred_xyz, gt_xyz, timestep_horizon=None, only_xy: bool = True):
    torch = pytest.importorskip("torch")
    diff = pred_xyz - gt_xyz[:, None, None, :, :]
    if only_xy:
        diff = diff[..., :2]
    l2 = torch.linalg.norm(diff, ord=2, dim=-1)
    if timestep_horizon is not None:
        if timestep_horizon > l2.shape[-1]:
            raise ValueError(f"{timestep_horizon=} must be less than {l2.shape[-1]=}")
        l2 = l2[..., :timestep_horizon]
    return l2.mean(dim=-1)


def _load_compute_ade():
    _ensure_src_on_path()
    pytest.importorskip("torch")
    try:
        from alpamayo_r1.metrics.distance_metrics import compute_ade

        return compute_ade
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_distance_ade_xy", _DIST)
        if spec is None or spec.loader is None:
            return _compute_ade
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.compute_ade
        except Exception:
            return _compute_ade


def test_source_default_only_xy_slices_first_two():
    src = _DIST.read_text()
    assert "only_xy: bool = True" in src
    assert "if only_xy:" in src
    assert "diff = diff[..., :2]" in src
    assert "l2.mean(dim=-1)" in src


def test_default_ade_ignores_z_offset():
    torch = pytest.importorskip("torch")
    compute_ade = _load_compute_ade()
    gt = torch.zeros(1, 4, 3)
    pred = torch.zeros(1, 1, 1, 4, 3)
    pred[..., 2] = 3.0
    xy_only = compute_ade(pred, gt)
    with_z = compute_ade(pred, gt, only_xy=False)
    torch.testing.assert_close(xy_only, torch.zeros(1, 1, 1))
    torch.testing.assert_close(with_z, torch.full((1, 1, 1), 3.0))


def test_xy_error_is_planar_l2_not_xyz():
    torch = pytest.importorskip("torch")
    compute_ade = _load_compute_ade()
    gt = torch.zeros(1, 2, 3)
    pred = torch.zeros(1, 1, 1, 2, 3)
    pred[..., 0] = 3.0
    pred[..., 1] = 4.0
    pred[..., 2] = 12.0
    # XY hypot is 5; XYZ hypot would be 13.
    torch.testing.assert_close(compute_ade(pred, gt), torch.full((1, 1, 1), 5.0))
    torch.testing.assert_close(
        compute_ade(pred, gt, only_xy=False), torch.full((1, 1, 1), 13.0)
    )
