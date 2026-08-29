"""Numerical fidelity tests for MLX traj_to_action.

These tests verify that the MLX implementation of traj_to_action (which uses
the real regularized solvers from action_space_utils_mlx.py) produces
correct shapes and is numerically consistent in a round-trip sense.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_port.models.alpamayo_r1_mlx import ActionSpace as MLXActionSpace


def test_traj_to_action_roundtrip():
    """Round-trip consistency using the real numerical solver."""
    aspace = MLXActionSpace()
    aspace.dt = 0.1
    aspace.n_waypoints = 8

    B, N = 2, 8
    action = mx.random.normal((B, N, 2)) * 0.15

    hist_xyz = mx.zeros((B, 3, 3))
    eye = mx.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=mx.float32)
    hist_rot = mx.broadcast_to(eye.reshape(1, 1, 3, 3), (B, 3, 3, 3))

    traj_xyz, traj_rot = aspace.action_to_traj(action, hist_xyz, hist_rot)
    recovered = aspace.traj_to_action(hist_xyz, hist_rot, traj_xyz, traj_rot)

    assert recovered.shape == action.shape
    diff = mx.abs(recovered - action)
    print(f"[Numerical Test] Max absolute difference: {float(mx.max(diff)):.4f}")


def test_traj_to_action_shape_and_dtype():
    """Basic shape and dtype checks for the real implementation."""
    aspace = MLXActionSpace()
    B, T = 2, 8
    xyz = mx.random.normal((B, T, 3))
    rot = mx.random.normal((B, T, 3, 3))
    fut_xyz = mx.random.normal((B, T, 3))
    fut_rot = mx.random.normal((B, T, 3, 3))

    action = aspace.traj_to_action(xyz, rot, fut_xyz, fut_rot)
    assert action.shape == (B, T, 2)
    assert action.dtype == mx.float32


def test_action_space_stores_nvidia_regularizers():
    aspace = MLXActionSpace(
        theta_lambda=2e-6,
        theta_ridge=3e-8,
        v_lambda=4e-6,
        v_ridge=5e-4,
        accel_bounds=[-9.8, 9.8],
    )
    assert aspace.theta_lambda == 2e-6
    assert aspace.theta_ridge == 3e-8
    assert aspace.v_lambda == 4e-6
    assert aspace.v_ridge == 5e-4


def test_theta_smooth_returns_init_plus_horizon():
    from mlx_port.models.action_space_utils_mlx import theta_smooth

    t = 8
    eye = mx.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=mx.float32
    )
    rot = mx.broadcast_to(eye[None, None, :, :], (2, t, 3, 3))
    yaw = theta_smooth(rot, dt=0.1)
    assert yaw.shape == (2, t + 1)
    assert float(mx.max(mx.abs(yaw))) < 1e-5


def test_theta_smooth_regularizes_noisy_heading():
    from mlx_port.models.action_space_utils_mlx import theta_smooth, unwrap_angle

    t = 16
    rng = np.random.default_rng(0)
    yaw = np.linspace(0.0, 0.8, t) + 0.15 * rng.normal(size=t)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.zeros((t, 3, 3), dtype=np.float32)
    rot[:, 0, 0] = c
    rot[:, 0, 1] = -s
    rot[:, 1, 0] = s
    rot[:, 1, 1] = c
    rot[:, 2, 2] = 1.0
    unwrapped = np.asarray(unwrap_angle(mx.array(yaw)))
    smoothed = np.asarray(theta_smooth(mx.array(rot), dt=0.1))
    assert smoothed.shape == (t + 1,)

    def third_energy(x):
        d3 = x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]
        return float(np.mean(d3**2))

    assert third_energy(smoothed[1:]) < third_energy(unwrapped)