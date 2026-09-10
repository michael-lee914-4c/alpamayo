"""NVIDIA unicycle bound check, decode origin/z-copy, unwrap, and D bands.

Weight-free. Complementary to open #25 ``n_waypoints`` raise and the MLX
analytical unicycle suite — this is the CUDA ``is_within_bounds`` /
``action_to_traj`` / smoother-matrix contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_unicycle():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )
        from alpamayo_r1.action_space.utils import (
            first_order_D,
            second_order_D,
            third_order_D,
            unwrap_angle,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA unicycle unavailable: {exc}")
    return (
        UnicycleAccelCurvatureActionSpace,
        first_order_D,
        second_order_D,
        third_order_D,
        unwrap_angle,
    )


def test_is_within_bounds_denormalizes_then_is_inclusive(torch):
    UnicycleAccelCurvatureActionSpace, *_rest = _load_unicycle()
    del _rest
    # Defaults: accel ~ N(0,1) with [-9.8, 9.8]; kappa ~ N(0,1) with [-0.2, 0.2].
    spc = UnicycleAccelCurvatureActionSpace(n_waypoints=4, dt=0.1)
    edge = torch.zeros(1, 4, 2)
    edge[..., 0] = 9.8
    edge[..., 1] = 0.2
    assert bool(spc.is_within_bounds(edge).item()) is True

    over_accel = edge.clone()
    over_accel[..., 0] = 9.81
    assert bool(spc.is_within_bounds(over_accel).item()) is False

    over_kappa = edge.clone()
    over_kappa[..., 1] = 0.21
    assert bool(spc.is_within_bounds(over_kappa).item()) is False

    # Any single timestep out of range fails the whole sequence.
    one = torch.zeros(2, 4, 2)
    one[1, 2, 0] = 10.0
    got = spc.is_within_bounds(one)
    assert got.tolist() == [True, False]


def test_action_to_traj_starts_at_origin_and_copies_hist_z(torch):
    UnicycleAccelCurvatureActionSpace, *_rest = _load_unicycle()
    del _rest
    spc = UnicycleAccelCurvatureActionSpace(n_waypoints=4, dt=0.1)
    hist_xyz = torch.zeros(1, 3, 3)
    hist_xyz[..., 2] = 1.5
    hist_rot = torch.eye(3).expand(1, 3, 3, 3).clone()
    # Normalized (0, 0) → physical accel=0, kappa=0. With v0=2 and dt=0.1,
    # trapezoid integration advances +0.2 m/step along +x.
    xyz, rot = spc.action_to_traj(
        torch.zeros(1, 4, 2),
        hist_xyz,
        hist_rot,
        t0_states={"v": torch.tensor([2.0])},
    )
    assert xyz.shape == (1, 4, 3)
    assert torch.allclose(
        xyz[0, :, 0], torch.tensor([0.2, 0.4, 0.6, 0.8]), atol=1e-5
    )
    assert torch.allclose(xyz[0, :, 1], torch.zeros(4), atol=1e-5)
    assert torch.allclose(xyz[0, :, 2], torch.full((4,), 1.5), atol=1e-6)
    # Zero curvature keeps identity yaw (flat XY → SO(3)).
    assert torch.allclose(rot[0, 0], torch.eye(3), atol=1e-5)


def test_unwrap_angle_closes_a_2pi_jump(torch):
    *_rest, unwrap_angle = _load_unicycle()
    del _rest
    # π-0.1 then -π+0.1 is a +0.2 rad step, not a ~-2π drop.
    phi = torch.tensor([[np.pi - 0.1, -np.pi + 0.1]], dtype=torch.float32)
    out = unwrap_angle(phi)
    assert torch.allclose(out[0, 0], phi[0, 0], atol=1e-6)
    assert float((out[0, 1] - out[0, 0]).item()) == pytest.approx(0.2, abs=1e-5)
    assert float(out[0, 1].item()) > np.pi


def test_smoothing_D_bands_and_short_n(torch):
    (
        _spc,
        first_order_D,
        second_order_D,
        third_order_D,
        _unwrap,
    ) = _load_unicycle()
    del _spc, _unwrap
    d1 = first_order_D(4, ())
    assert d1.shape == (3, 4)
    assert torch.allclose(
        d1,
        torch.tensor(
            [
                [-1.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 1.0],
            ]
        ),
    )
    d2 = second_order_D(4, ())
    assert torch.allclose(
        d2,
        torch.tensor(
            [
                [-1.0, 2.0, -1.0, 0.0],
                [0.0, -1.0, 2.0, -1.0],
            ]
        ),
    )
    d3 = third_order_D(4, ())
    assert torch.allclose(d3, torch.tensor([[-1.0, 3.0, -3.0, 1.0]]))
    # N too short for 3rd order is an empty band, not an error.
    assert tuple(third_order_D(3, ()).shape) == (0, 3)
    batched = first_order_D(3, (2,))
    assert batched.shape == (2, 2, 3)
    assert torch.equal(batched[0], batched[1])
