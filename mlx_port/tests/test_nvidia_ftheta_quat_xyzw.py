"""f-theta viz builds SciPy rotations as ``[qx, qy, qz, qw]`` (xyzw).

Weight-free. Complementary to open #44 (``(wp - cam_t) @ cam_rot``)
and #29 (on-axis keep / behind+OOB drop). ``Rotation.from_quat`` is
xyzw. Passing wxyz would invert the camera pose and project waypoints
onto the wrong pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_VIZ = _REPO / "src" / "alpamayo_r1" / "visualization" / "viz.py"


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_source_uses_xyzw_scipy_quat():
    src = _VIZ.read_text()
    body = src.split("def viz_waypoints_pai")[1].split("def make_image_grid")[0]
    assert "Rotation.from_quat([qx, qy, qz, qw])" in body
    assert "[qw, qx, qy, qz]" not in body


def test_xyzw_vs_wxyz_disagree_on_a_90deg_yaw():
    # 90 deg about z: xyzw = (0, 0, sin(π/4), cos(π/4))
    s, c = np.sin(np.pi / 4), np.cos(np.pi / 4)
    xyzw = np.array([0.0, 0.0, s, c])
    wxyz_as_if_xyzw = np.array([c, 0.0, 0.0, s])
    m_xyzw = _quat_xyzw_to_matrix(xyzw)
    m_swapped = _quat_xyzw_to_matrix(wxyz_as_if_xyzw)
    assert not np.allclose(m_xyzw, m_swapped)
    x_axis = m_xyzw @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(x_axis, [0.0, 1.0, 0.0], atol=1e-7)
