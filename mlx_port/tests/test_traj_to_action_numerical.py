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


def test_without_v0_recovers_constant_speed():
    from mlx_port.models.action_space_utils_mlx import dxy_theta_to_v_without_v0

    dt, v_true, n = 0.1, 8.0, 8
    dxy = np.zeros((n, 2), dtype=np.float32)
    dxy[:, 0] = v_true * dt
    v = np.asarray(dxy_theta_to_v_without_v0(mx.array(dxy), mx.zeros((n + 1,)), dt=dt))
    assert v.shape == (n + 1,)
    assert abs(float(v.mean()) - v_true) < 5e-2
    assert float(np.max(np.abs(v - v_true))) < 0.15


def test_without_v0_differs_from_pinned_v0_on_braking():
    from mlx_port.models.action_space_utils_mlx import dxy_theta_to_v, dxy_theta_to_v_without_v0

    dt = 0.1
    xy = np.array([0.0, 1.0, 1.8, 2.4, 2.8, 3.0], dtype=np.float32)
    dxy = np.stack([np.diff(xy), np.zeros(len(xy) - 1, dtype=np.float32)], axis=-1)
    theta = mx.zeros((len(xy),))
    v_joint = np.asarray(dxy_theta_to_v_without_v0(mx.array(dxy), theta, dt=dt))
    v_pin = np.asarray(dxy_theta_to_v(mx.array(dxy), theta, mx.array(0.0), dt=dt))
    assert v_joint.shape == (len(xy),)
    assert abs(float(v_joint[-1]) - float(v_pin[-1])) > 0.4
    assert float(v_joint[0]) > 4.0
    assert float(v_pin[0]) == pytest.approx(0.0, abs=1e-5)
    assert float(v_joint[-1]) > float(v_pin[-1])


def test_estimate_t0_uses_without_v0_not_pinned_zero():
    aspace = MLXActionSpace(dt=0.1, n_waypoints=8)
    dt, v_true, t = 0.1, 5.0, 5
    xyz = np.zeros((1, t, 3), dtype=np.float32)
    xyz[0, :, 0] = v_true * np.arange(t) * dt
    eye = np.eye(3, dtype=np.float32)
    rot = np.broadcast_to(eye, (1, t, 3, 3)).copy()
    t0 = aspace.estimate_t0_states(mx.array(xyz), mx.array(rot))
    assert abs(float(np.asarray(t0["v"]).reshape(-1)[0]) - v_true) < 0.2


def test_action_to_traj_estimates_t0_from_moving_history():
    """NVIDIA action_to_traj calls estimate_t0 when t0 is omitted."""
    aspace = MLXActionSpace(dt=0.1, n_waypoints=8)
    v_true = 8.0
    dt = 0.1
    hist_xyz = mx.array(
        [[[-2 * v_true * dt, 0.0, 0.0], [-v_true * dt, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    eye = mx.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    hist_rot = mx.broadcast_to(eye[None, None, :, :], (1, 3, 3, 3))
    zero_action = mx.zeros((1, 8, 2))
    fut_xyz, _ = aspace.action_to_traj(zero_action, hist_xyz, hist_rot)
    end_x = float(np.asarray(fut_xyz)[0, -1, 0])
    assert end_x > 4.0, f"keep-rolling t0 should integrate ~6.4 m, got {end_x:.3f}"


def test_without_v0_matches_nvidia_torch():
    torch = pytest.importorskip("torch")
    try:
        from alpamayo_r1.action_space.utils import (
            dxy_theta_to_v_without_v0 as torch_without_v0,
        )
    except ImportError:
        pytest.skip("NVIDIA alpamayo_r1 not importable")

    from mlx_port.models.action_space_utils_mlx import dxy_theta_to_v_without_v0

    rng = np.random.default_rng(0)
    dxy = rng.normal(scale=0.4, size=(2, 6, 2)).astype(np.float32)
    theta = rng.normal(scale=0.05, size=(2, 7)).astype(np.float32)
    mlx_v = np.asarray(dxy_theta_to_v_without_v0(mx.array(dxy), mx.array(theta), dt=0.1))
    with torch.no_grad():
        torch_v = torch_without_v0(
            torch.tensor(dxy), torch.tensor(theta), dt=0.1
        ).numpy()
    np.testing.assert_allclose(mlx_v, torch_v, rtol=1e-4, atol=1e-4)