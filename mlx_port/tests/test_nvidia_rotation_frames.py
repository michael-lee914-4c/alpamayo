"""NVIDIA rotation-frame helpers used by unicycle decode and heading wrap.

Weight-free. Complementary to open #25 ``angle_wrap`` / ``so3_to_yaw`` /
2D rotate — this locks ``ratan2(0,0)``, 2D↔3D embed, Gram-Schmidt, and
the ``round_2pi(π)=π`` vs ``angle_wrap(π)=-π`` split.
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


def _load_rotation():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.geometry.rotation import (
            angle_wrap,
            ratan2,
            rot_2d_to_3d,
            rot_3d_to_2d,
            rotation_matrix_torch,
            round_2pi,
            stable_gramschmidt,
            transform_coords_2d_np,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA rotation unavailable: {exc}")
    return (
        angle_wrap,
        ratan2,
        rot_2d_to_3d,
        rot_3d_to_2d,
        rotation_matrix_torch,
        round_2pi,
        stable_gramschmidt,
        transform_coords_2d_np,
    )


def test_ratan2_zero_over_zero_is_zero_not_nan(torch):
    _angle_wrap, ratan2, *_rest = _load_rotation()
    del _angle_wrap, _rest
    zero = ratan2(torch.tensor(0.0), torch.tensor(0.0))
    assert float(zero.item()) == 0.0
    assert not torch.isnan(zero)
    # Tiny negative cosine is nudged away from the origin, not left at 0/0.
    near = ratan2(torch.tensor(0.0), torch.tensor(-1e-8), eps=1e-4)
    assert abs(abs(float(near.item())) - np.pi) < 1e-4


def test_rot_2d_to_3d_is_flat_xy_and_inverts(torch):
    (
        _aw,
        _ratan,
        rot_2d_to_3d,
        rot_3d_to_2d,
        rotation_matrix_torch,
        *_rest,
    ) = _load_rotation()
    del _aw, _ratan, _rest
    r2 = rotation_matrix_torch(torch.tensor(np.pi / 2, dtype=torch.float32))
    r3 = rot_2d_to_3d(r2)
    assert r3.shape == (3, 3)
    # 90°: (1,0,0) → (0,1,0); z stays the world-up axis.
    x_axis = r3 @ torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(x_axis, torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)
    assert torch.allclose(r3[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6)
    back = rot_3d_to_2d(r3)
    assert torch.allclose(back, r2, atol=1e-5)


def test_stable_gramschmidt_is_right_handed_orthonormal(torch):
    *_rest, stable_gramschmidt, _tx = _load_rotation()
    del _rest, _tx
    # Skewed x/y in the XY plane must be orthonormalized with det +1.
    m = torch.tensor([[2.0, 0.5], [0.3, 1.0], [0.0, 0.0]], dtype=torch.float32)
    r = stable_gramschmidt(m)
    assert r.shape == (3, 3)
    gram = r.T @ r
    assert torch.allclose(gram, torch.eye(3), atol=1e-5)
    assert abs(float(torch.det(r).item()) - 1.0) < 1e-5
    # z is x × y and points +world-z for an XY-plane pair.
    assert float(r[2, 2].item()) > 0.9


def test_transform_coords_rotate_then_offset_and_pi_wrap_split():
    (
        angle_wrap,
        _ratan,
        _r23,
        _r32,
        _rm,
        round_2pi,
        _gs,
        transform_coords_2d_np,
    ) = _load_rotation()
    del _ratan, _r23, _r32, _rm, _gs
    pts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    out = transform_coords_2d_np(pts.copy(), offset=np.array([10.0, 0.0]), angle=np.pi / 2)
    # (1,0) rot90 → (0,1); (0,1) rot90 → (-1,0); then +10 on x.
    np.testing.assert_allclose(out[0], [10.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(out[1], [9.0, 0.0], atol=1e-7)
    # rot_mat wins over angle.
    ident = np.eye(2)
    out_mat = transform_coords_2d_np(
        np.array([1.0, 0.0]), angle=np.pi / 2, rot_mat=ident
    )
    np.testing.assert_allclose(out_mat, [1.0, 0.0], atol=1e-7)

    # angle_wrap is left-closed [-π, π); round_2pi (atan2) keeps +π.
    assert float(angle_wrap(np.array(np.pi))) == pytest.approx(-np.pi)
    assert float(round_2pi(np.array(np.pi))) == pytest.approx(np.pi)
