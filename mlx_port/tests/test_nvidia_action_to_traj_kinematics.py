"""Unicycle ``action_to_traj`` is trapezoidal with ``theta[0] = 0``.

Weight-free. Complementary to open #26 (origin / +x / Z-copy) and #44
(``output_all_states`` uses ``[:, :-1]``). Decode integrates

    v_{t+1} = v0 + cumsum(a dt)
    θ_{t+1} = cumsum(κ v dt + κ a dt²/2)   with θ_0 = 0
    x,y     = trapezoid ½dt (v_t u_t + v_{t+1} u_{t+1})

and copies Z from the last history waypoint. Starting yaw from ``hist_rot``
or using a rectangular ``v dt`` step would warp every Stage-2 / infer sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def test_source_uses_zero_yaw_trapezoid_and_hist_z():
    src = _UNI.read_text()
    assert "initial_yaw = torch.zeros_like(v0)" in src
    assert "half_dt_term = 0.5 * dt" in src
    assert "torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)" in src
    assert "torch.cumsum(kappa * accel * dt_2_term, dim=-1)" in src
    assert "torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt_term" in src
    assert "torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt_term" in src
    assert "traj_future_xyz[..., 2] = traj_history_xyz[..., -1:, 2]" in src
    assert "rot_2d_to_3d(rotation_matrix_torch(theta[..., 1:]))" in src


def _action_to_traj_xy(accel, kappa, v0, dt, hist_z):
    torch = pytest.importorskip("torch")
    velocity = torch.cat(
        [v0.unsqueeze(-1), v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)],
        dim=-1,
    )
    dt_2_term = 0.5 * (dt**2)
    initial_yaw = torch.zeros_like(v0)
    theta = torch.cat(
        [
            initial_yaw.unsqueeze(-1),
            (
                initial_yaw.unsqueeze(-1)
                + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
                + torch.cumsum(kappa * accel * dt_2_term, dim=-1)
            ),
        ],
        dim=-1,
    )
    half_dt = 0.5 * dt
    x = (
        torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1)
        + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt, dim=-1)
    )
    y = (
        torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1)
        + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt, dim=-1)
    )
    z = hist_z.expand_as(x)
    return torch.stack([x, y, z], dim=-1), theta


def test_zero_action_is_straight_plus_x_and_copies_hist_z():
    torch = pytest.importorskip("torch")
    n = 4
    dt = 0.1
    v0 = torch.tensor([2.0])
    accel = torch.zeros(1, n)
    kappa = torch.zeros(1, n)
    hist_z = torch.tensor([7.5])
    xyz, theta = _action_to_traj_xy(accel, kappa, v0, dt, hist_z)
    # Constant v=2, κ=0 → x = v * t, y = 0, θ stays 0.
    t = torch.arange(1, n + 1, dtype=torch.float32) * dt
    torch.testing.assert_close(xyz[0, :, 0], 2.0 * t, atol=1e-5, rtol=0)
    torch.testing.assert_close(xyz[0, :, 1], torch.zeros(n), atol=1e-5, rtol=0)
    torch.testing.assert_close(xyz[0, :, 2], torch.full((n,), 7.5), atol=1e-5, rtol=0)
    torch.testing.assert_close(theta[0], torch.zeros(n + 1), atol=1e-5, rtol=0)


def test_rect_step_does_not_match_trapezoid_when_accelerating():
    torch = pytest.importorskip("torch")
    n = 3
    dt = 0.1
    v0 = torch.tensor([1.0])
    accel = torch.ones(1, n)
    kappa = torch.zeros(1, n)
    xyz, _ = _action_to_traj_xy(accel, kappa, v0, dt, torch.tensor([0.0]))
    # Rectangular ``x += v_t dt`` (no half-step on v_{t+1}) under-integrates.
    v = torch.tensor([1.0, 1.1, 1.2, 1.3])
    rect = torch.cumsum(v[:-1] * dt, dim=0)
    assert float((xyz[0, :, 0] - rect).abs().max()) > 1e-3
