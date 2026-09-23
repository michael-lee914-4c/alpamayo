"""Unicycle ``is_within_bounds`` denormalizes before the physical gates.

Weight-free. Complementary to open #26 (inclusive denorm on encode/decode)
and #25 (``n_waypoints`` raise). Bounds are in physical units; the action
tensor is ``(x-mean)/std``. Comparing the normalized value to
``accel_bounds`` would accept a 2σ overspeed or reject a legal maneuver.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def _is_within_bounds(action, *, accel_mean, accel_std, kappa_mean, kappa_std, accel_bounds, curvature_bounds):
    torch = pytest.importorskip("torch")
    accel = action[..., 0] * accel_std + accel_mean
    kappa = action[..., 1] * kappa_std + kappa_mean
    is_accel = (accel >= accel_bounds[0]) & (accel <= accel_bounds[1])
    is_kappa = (kappa >= curvature_bounds[0]) & (kappa <= curvature_bounds[1])
    return torch.all(is_accel & is_kappa, dim=-1)


def test_source_denormalizes_then_gates_all_waypoints():
    src = _UNI.read_text()
    assert "accel = accel * accel_std + accel_mean" in src
    assert "kappa = kappa * kappa_std + kappa_mean" in src
    assert "is_accel_within_bounds = (accel >= self.accel_bounds[0]) & (accel <= self.accel_bounds[1])" in src
    assert "return torch.all(is_accel_within_bounds & is_kappa_within_bounds, dim=-1)" in src


def test_normalized_overspeed_fails_only_after_denorm():
    torch = pytest.importorskip("torch")
    # std=2 so normalized 5.5 → physical 11.0, which is past 9.8 m/s².
    # Without denorm, 5.5 < 9.8 would incorrectly pass.
    action = torch.zeros(1, 4, 2)
    action[0, 2, 0] = 5.5
    got = _is_within_bounds(
        action,
        accel_mean=0.0,
        accel_std=2.0,
        kappa_mean=0.0,
        kappa_std=0.1,
        accel_bounds=(-9.8, 9.8),
        curvature_bounds=(-0.2, 0.2),
    )
    assert got.tolist() == [False]
    raw_compare = torch.all(
        (action[..., 0] >= -9.8)
        & (action[..., 0] <= 9.8)
        & (action[..., 1] >= -0.2)
        & (action[..., 1] <= 0.2),
        dim=-1,
    )
    assert raw_compare.tolist() == [True]


def test_normalized_curvature_passes_only_after_denorm():
    torch = pytest.importorskip("torch")
    # std=0.1 so normalized 1.5 → physical 0.15, inside ±0.2.
    # Without denorm, 1.5 > 0.2 would incorrectly fail.
    action = torch.zeros(1, 3, 2)
    action[0, 1, 1] = 1.5
    got = _is_within_bounds(
        action,
        accel_mean=0.0,
        accel_std=2.0,
        kappa_mean=0.0,
        kappa_std=0.1,
        accel_bounds=(-9.8, 9.8),
        curvature_bounds=(-0.2, 0.2),
    )
    assert got.tolist() == [True]
    raw_compare = torch.all(
        (action[..., 0] >= -9.8)
        & (action[..., 0] <= 9.8)
        & (action[..., 1] >= -0.2)
        & (action[..., 1] <= 0.2),
        dim=-1,
    )
    assert raw_compare.tolist() == [False]
