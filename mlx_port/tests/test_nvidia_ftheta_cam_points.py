"""``project_waypoints_ftheta`` is ``(wp - cam_t) @ cam_rot`` then ``z > 0``.

Weight-free. Complementary to open #29 (f-theta on-axis keep / behind+OOB
drop on another helper). This overlay path multiplies world-minus-t by
``cam_rot`` (not ``cam_rot.T @``), then gates ``z > 0`` and image bounds.
A swapped ``@`` order or ``z >= 0`` leaks points behind the camera onto
the front-wide debug image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_VIZ = _REPO / "src" / "alpamayo_r1" / "visualization" / "viz.py"


def _project_waypoints_ftheta(wp, cam_rot, cam_t, intr):
    np = pytest.importorskip("numpy")
    width, height, cx, cy, fw_poly_0, fw_poly_1, fw_poly_2, fw_poly_3, fw_poly_4 = intr
    cam_points = (wp - cam_t) @ cam_rot
    x, y, z = cam_points.T
    r_xy = np.sqrt(x**2 + y**2)
    theta = np.arctan2(r_xy, z)
    radius = (
        fw_poly_0
        + fw_poly_1 * theta
        + fw_poly_2 * theta**2
        + fw_poly_3 * theta**3
        + fw_poly_4 * theta**4
    )
    scale = radius / (r_xy + 1e-8)
    u = cx + x * scale
    v = cy + y * scale
    projected = np.stack([u, v], axis=1)
    valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return projected[valid]


def test_source_uses_wp_minus_t_at_rot_and_strict_z():
    src = _VIZ.read_text()
    assert "cam_points = (wp - cam_t) @ cam_rot" in src
    assert "valid = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)" in src
    assert "cam_rot.T" not in src.split("def viz_waypoints_pai")[0]


def test_on_axis_point_lands_at_principal_point():
    np = pytest.importorskip("numpy")
    wp = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
    cam_rot = np.eye(3)
    cam_t = np.zeros(3)
    # Linear f-theta: radius = f * theta. On-axis theta=0 → (cx, cy).
    intr = [100.0, 80.0, 50.0, 40.0, 0.0, 100.0, 0.0, 0.0, 0.0]
    got = _project_waypoints_ftheta(wp, cam_rot, cam_t, intr)
    assert got.shape == (1, 2)
    np.testing.assert_allclose(got[0], [50.0, 40.0], atol=1e-6)


def test_behind_camera_is_dropped():
    np = pytest.importorskip("numpy")
    wp = np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    cam_rot = np.eye(3)
    cam_t = np.zeros(3)
    intr = [100.0, 80.0, 50.0, 40.0, 0.0, 100.0, 0.0, 0.0, 0.0]
    got = _project_waypoints_ftheta(wp, cam_rot, cam_t, intr)
    assert got.shape == (1, 2)
    np.testing.assert_allclose(got[0], [50.0, 40.0], atol=1e-6)
