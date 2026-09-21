"""NVIDIA BEV rotate, leftover grid cells, and shared-span limits.

Weight-free. Complementary to open #29 (f-theta keep/drop) and #33
(``frames[7]`` front-wide t0). Reports rotate ``(x, y) → (-y, x)``, pad
leftover grid cells with zeros, and force a shared / min-2 m span so a
tiny segment is not zoomed into noise.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_VIZ = _SRC / "alpamayo_r1" / "visualization" / "viz.py"


def _load_viz():
    if "matplotlib" not in sys.modules:
        mpl = ModuleType("matplotlib")
        mpl.pyplot = ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = mpl.pyplot
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = ModuleType("cv2")
    try:
        import pandas  # noqa: F401
    except Exception:
        pandas = ModuleType("pandas")
        pandas.DataFrame = type("DataFrame", (), {})
        sys.modules["pandas"] = pandas
    try:
        import scipy.spatial.transform  # noqa: F401
    except Exception as exc:
        pytest.skip(f"scipy.spatial.transform unavailable: {exc}")
    spec = importlib.util.spec_from_file_location("nv_viz_rotate_grid", _VIZ)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_VIZ}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA viz helpers unavailable: {exc}")
    return module


def test_source_rotate_grid_and_shared_span():
    src = _VIZ.read_text()
    assert "return np.stack([-xy[1], xy[0]], axis=0)" in src
    assert "grid = np.zeros((rows * h, columns * w, c), dtype=images.dtype)" in src
    assert "span = max(x_span, y_span)" in src
    assert "min_span: float = 2.0" in src
    assert "pad_ratio: float = 0.12" in src


def test_rotate_90cc_maps_plus_x_to_plus_y():
    viz = _load_viz()
    # (2, N): a point on +X must land on +Y after 90° CCW.
    xy = np.array([[2.0, 0.0], [0.0, 1.0]])
    rotated = viz.rotate_90cc(xy)
    np.testing.assert_allclose(rotated, np.array([[0.0, -1.0], [2.0, 0.0]]))
    # A CW swap (x, y) → (y, -x) would send +X to -Y and flip every BEV.
    clockwise = np.stack([xy[1], -xy[0]], axis=0)
    assert not np.allclose(rotated, clockwise)


def test_make_image_grid_leaves_leftover_cells_zero():
    viz = _load_viz()
    images = np.arange(3 * 2 * 2 * 1, dtype=np.uint8).reshape(3, 2, 2, 1)
    grid = viz.make_image_grid(images, columns=2)
    assert grid.shape == (4, 4, 1)
    np.testing.assert_array_equal(grid[:2, :2, 0], images[0, :, :, 0])
    np.testing.assert_array_equal(grid[:2, 2:, 0], images[1, :, :, 0])
    np.testing.assert_array_equal(grid[2:, :2, 0], images[2, :, :, 0])
    # The unused last cell stays background, not a wrap of image 0.
    np.testing.assert_array_equal(grid[2:, 2:, 0], np.zeros((2, 2), dtype=np.uint8))


def test_tight_limits_share_span_and_honor_min_span():
    viz = _load_viz()
    ax = SimpleNamespace(xlim=None, ylim=None, aspect=None)

    def set_xlim(lo, hi):
        ax.xlim = (lo, hi)

    def set_ylim(lo, hi):
        ax.ylim = (lo, hi)

    def set_aspect(value, adjustable=None):
        del adjustable
        ax.aspect = value

    ax.set_xlim = set_xlim
    ax.set_ylim = set_ylim
    ax.set_aspect = set_aspect

    # 0.2 m segment would collapse the plot without min_span=2.
    tiny = np.array([[0.0, 0.2], [0.0, 0.0]])
    viz._set_tight_trajectory_limits(ax, [tiny], pad_ratio=0.12, min_span=2.0)
    x_span = ax.xlim[1] - ax.xlim[0]
    y_span = ax.ylim[1] - ax.ylim[0]
    assert x_span == pytest.approx(y_span)
    # shared span 2.0 plus 12% pad on each side → 2.0 * 1.24
    assert x_span == pytest.approx(2.0 * 1.24)
    assert ax.aspect == "equal"
