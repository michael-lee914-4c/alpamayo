"""4×4 camera/frame image_grid_thw grouping (no weights)."""

import numpy as np

from mlx_port.processor import enforce_alpamayo_temporal_grouping


def test_enforce_groups_sixteen_frames_into_four_cameras():
    grid = np.tile(np.array([1, 20, 36], dtype=np.int64), (16, 1))
    inputs = {"image_grid_thw": grid, "keep": 1}
    out = enforce_alpamayo_temporal_grouping(inputs)
    grouped = np.asarray(out["image_grid_thw"])
    assert grouped.shape == (4, 3)
    assert np.array_equal(grouped, np.tile([4, 20, 36], (4, 1)))
    assert out["keep"] == 1
    assert np.array_equal(inputs["image_grid_thw"], grid)


def test_enforce_leaves_non_16_grid_unchanged():
    grid = np.tile(np.array([1, 20, 36], dtype=np.int64), (15, 1))
    inputs = {"image_grid_thw": grid}
    out = enforce_alpamayo_temporal_grouping(inputs)
    assert out is inputs
    assert np.array_equal(out["image_grid_thw"], grid)


def test_enforce_missing_grid_is_noop():
    inputs = {"input_ids": np.array([[1, 2]])}
    assert enforce_alpamayo_temporal_grouping(inputs) is inputs
