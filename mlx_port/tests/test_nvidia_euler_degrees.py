"""``euler_2_so3`` defaults to degrees, not radians.

Weight-free. Complementary to open #25 (``so3_to_yaw`` first-column /
2D rotate-by-π/2). SciPy ``from_euler(..., degrees=True)`` is the default,
so a π-radian yaw passed without ``degrees=False`` is treated as 180° of
arcminutes and produces a near-identity matrix.
"""

from __future__ import annotations

import importlib.util
import inspect
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


def _load_rotation():
    _ensure_src_on_path()
    pytest.importorskip("scipy")
    pytest.importorskip("torch")
    try:
        from scipy.spatial.transform import Rotation as _R

        del _R
    except Exception as exc:
        pytest.skip(f"scipy.spatial.transform.Rotation unavailable: {exc}")
    try:
        from alpamayo_r1.geometry.rotation import euler_2_so3, so3_to_yaw_np

        return euler_2_so3, so3_to_yaw_np
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_rotation_euler", _ROT)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_ROT}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            pytest.skip(f"NVIDIA rotation.euler_2_so3 unavailable: {exc}")
        return module.euler_2_so3, module.so3_to_yaw_np


def test_source_euler_default_is_degrees():
    src = _ROT.read_text()
    assert "def euler_2_so3(euler_angles: np.ndarray, degrees: bool = True" in src
    assert "R.from_euler(seq=seq, angles=euler_angles, degrees=degrees)" in src


def test_euler_2_so3_90deg_yaw_is_plus_y():
    pytest.importorskip("scipy")
    euler_2_so3, so3_to_yaw_np = _load_rotation()
    sig = inspect.signature(euler_2_so3)
    assert sig.parameters["degrees"].default is True

    yaw_90 = np.array([[0.0, 0.0, 90.0]], dtype=np.float32)
    rot = euler_2_so3(yaw_90)
    assert rot.shape == (1, 3, 3)
    # +X body axis lands on world +Y (xyz Euler, yaw about Z).
    np.testing.assert_allclose(rot[0] @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(so3_to_yaw_np(rot), [np.pi / 2], atol=1e-5)

    # π passed with the default degrees=True is ~0.05 rad, not 180°.
    pi_as_degrees = euler_2_so3(np.array([[0.0, 0.0, np.pi]], dtype=np.float32))
    pi_as_radians = euler_2_so3(
        np.array([[0.0, 0.0, np.pi]], dtype=np.float32), degrees=False
    )
    np.testing.assert_allclose(pi_as_degrees[0], np.eye(3), atol=6e-2)
    np.testing.assert_allclose(pi_as_radians[0] @ np.array([1.0, 0.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-5)
    assert not np.allclose(pi_as_degrees[0], pi_as_radians[0], atol=1e-3)
