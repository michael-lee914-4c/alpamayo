"""Official ADE horizon skip vs raise, and only_xy=False Z term.

Weight-free. Complementary to open #24 (official ADE / minADE) and #27
(3s horizon / prefix). ``compute_ade`` raises when the horizon is longer
than T. ``compute_minade`` silently drops those horizons. ``only_xy=False``
includes Z, so a Z-only error is invisible under the default XY metric.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_DIST = _SRC / "alpamayo_r1" / "metrics" / "distance_metrics.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _import_metrics():
    sys.path.insert(0, str(_SRC))
    os.environ.setdefault("PYTHONPATH", str(_SRC))
    from alpamayo_r1.metrics.distance_metrics import compute_ade, compute_minade

    return compute_ade, compute_minade


def test_compute_ade_equal_horizon_ok_over_horizon_raises(torch):
    src = _DIST.read_text()
    assert "if timestep_horizon > l2.shape[-1]:" in src
    assert "must be less than" in src

    compute_ade, _ = _import_metrics()
    pred = torch.zeros(1, 1, 1, 4, 3)
    gt = torch.zeros(1, 4, 3)
    got = compute_ade(pred, gt, timestep_horizon=4)
    assert got.shape == (1, 1, 1)
    with pytest.raises(ValueError, match="timestep_horizon"):
        compute_ade(pred, gt, timestep_horizon=5)


def test_compute_minade_skips_horizons_past_t(torch):
    src = _DIST.read_text()
    assert "valid_horizons = [t for t in timestep_horizons if t <= T]" in src

    _, compute_minade = _import_metrics()
    pred = torch.zeros(2, 1, 3, 8, 3)
    gt = torch.zeros(2, 8, 3)
    out = compute_minade(
        pred,
        gt,
        disable_summary=True,
        timestep_horizons=[5, 10, 30, 50],
        time_step=0.1,
    )
    keys = set(out)
    assert "min_ade" in keys
    assert "min_ade/by_t=0.5" in keys
    # T=8, so horizon 10 (1.0s) is dropped; only 5 (0.5s) survives.
    assert "min_ade/by_t=1.0" not in keys
    assert "min_ade/by_t=3.0" not in keys
    assert "min_ade/by_t=5.0" not in keys


def test_compute_ade_only_xy_false_includes_z(torch):
    src = _DIST.read_text()
    assert "if only_xy:" in src
    assert "diff = diff[..., :2]" in src

    compute_ade, _ = _import_metrics()
    pred = torch.zeros(1, 1, 1, 2, 3)
    gt = torch.zeros(1, 2, 3)
    pred[..., 2] = 3.0  # Z-only error
    xy = compute_ade(pred, gt, only_xy=True)
    xyz = compute_ade(pred, gt, only_xy=False)
    assert float(xy) == 0.0
    assert float(xyz) == pytest.approx(3.0)
