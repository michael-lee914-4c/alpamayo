"""``action_to_traj`` builds rotations from ``theta[..., 1:]``, dropping θ0=0.

Weight-free. Complementary to open #45 (trapezoid + ``θ0=0`` +
``κ a dt²/2``). Heading is integrated to length N+1 with a pinned first
zero, then ``rotation_matrix_torch(theta[..., 1:])`` emits N waypoint
rotations. Using the full ``theta`` (including the dummy 0) would
misalign yaw vs XY by one step and feed ``action_to_traj`` a heading
that never existed at t0+dt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def test_source_rotations_use_theta_without_the_pinned_zero():
    src = _UNI.read_text()
    body = src.split("def action_to_traj")[1]
    assert "initial_yaw = torch.zeros_like(v0)" in body
    assert "traj_future_rot = rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))" in body
    assert "rotation_matrix_torch(theta)" not in body.replace(
        "rotation_matrix_torch(theta[..., 1:])", ""
    )


def test_dropping_theta0_keeps_waypoint_count():
    n = 4
    theta = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    assert theta.shape == (n + 1,)
    used = theta[1:]
    assert used.tolist() == [0.1, 0.2, 0.3, 0.4]
    assert used.shape == (n,)
    assert 0.0 not in used.tolist()
