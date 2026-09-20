"""``transform_coords_2d_np`` rotates first; ``rot_mat`` wins over ``angle``.

Weight-free. Complementary to open #26 (2D↔3D embed / Gram-Schmidt /
``ratan2(0,0)=0``). Local-frame PAI poses and viz overlays both go
through this helper. Applying the offset before the rotation, or using
``angle`` when ``rot_mat`` is set, would rotate about the wrong origin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_ROT = _SRC / "alpamayo_r1" / "geometry" / "rotation.py"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_transform():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.geometry.rotation import (
            rotation_matrix,
            transform_coords_2d_np,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA rotation helpers unavailable: {exc}")
    return rotation_matrix, transform_coords_2d_np


def test_source_prefers_rot_mat_and_applies_offset_last():
    src = _ROT.read_text()
    assert "if rot_mat is None and angle is not None:" in src
    assert "rot_mat = rotation_matrix(angle)" in src
    assert 'coords = np.einsum("...ij,...j->...i", rot_mat, coords)' in src
    assert "coords += offset" in src


def test_rot_mat_overrides_angle_and_offset_is_after_rotation():
    rotation_matrix, transform_coords_2d_np = _load_transform()
    pts = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    # +90° about origin: (1,0)→(0,1), (0,1)→(-1,0). Then + (10, 0).
    rot90 = rotation_matrix(np.pi / 2)
    # If angle were used instead of rot_mat this would be a -90° / identity mix.
    out = transform_coords_2d_np(
        pts.copy(),
        offset=np.array([10.0, 0.0]),
        angle=-np.pi / 2,
        rot_mat=rot90,
    )
    np.testing.assert_allclose(out, np.array([[10.0, 1.0], [9.0, 0.0]]), atol=1e-6)

    # Offset-only is a translation; angle-only rotates about the origin.
    translated = transform_coords_2d_np(pts.copy(), offset=np.array([2.0, 3.0]))
    np.testing.assert_allclose(translated, np.array([[3.0, 3.0], [2.0, 4.0]]))
    rotated = transform_coords_2d_np(pts.copy(), angle=np.pi)
    np.testing.assert_allclose(rotated, np.array([[-1.0, 0.0], [0.0, -1.0]]), atol=1e-6)
