"""``sort_images_by_camera_ids`` return arity depends on optional flags.

Weight-free. Complementary to open #25 (stable camera sort) and #41
(timestamps permute with the sort). Timestamps without
``return_camera_ids=True`` yield a 2-tuple ``(images, timestamps)``.
Callers that unpack ``images, cameras, ts`` would treat timestamps as
camera ids and silently mis-label every view.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _sort_images(image_frames, camera_indices, relative_timestamps=None, return_camera_ids=False):
    sorted_indices = np.argsort(camera_indices, kind="stable")
    result = [image_frames[sorted_indices]]
    if return_camera_ids:
        result.append(camera_indices[sorted_indices])
    if relative_timestamps is not None:
        result.append(relative_timestamps[sorted_indices])
    if len(result) == 1:
        return result[0]
    return tuple(result)


def test_source_appends_optional_outputs_in_flag_order():
    src = _PROC.read_text()
    body = src.split("def sort_images_by_camera_ids")[1].split("def basic_collation_fn")[0]
    cam_idx = body.index("if return_camera_ids:")
    ts_idx = body.index("if relative_timestamps is not None:")
    assert cam_idx < ts_idx
    assert "if len(result) == 1:" in body
    assert "return result[0]" in body
    assert "return tuple(result)" in body


def test_timestamps_without_camera_ids_is_a_two_tuple():
    frames = np.arange(6).reshape(3, 2)
    cams = np.array([2, 0, 1])
    ts = np.array([20.0, 0.0, 10.0])
    out = _sort_images(frames, cams, relative_timestamps=ts, return_camera_ids=False)
    assert isinstance(out, tuple) and len(out) == 2
    images, stamps = out
    np.testing.assert_array_equal(images, frames[[1, 2, 0]])
    np.testing.assert_array_equal(stamps, ts[[1, 2, 0]])


def test_both_flags_return_images_cameras_then_timestamps():
    frames = np.arange(6).reshape(3, 2)
    cams = np.array([2, 0, 1])
    ts = np.array([20.0, 0.0, 10.0])
    images, cameras, stamps = _sort_images(
        frames, cams, relative_timestamps=ts, return_camera_ids=True
    )
    np.testing.assert_array_equal(cameras, np.array([0, 1, 2]))
    np.testing.assert_array_equal(stamps, ts[[1, 2, 0]])
    assert images.shape == frames.shape


def test_images_only_returns_a_tensor_not_a_tuple():
    frames = np.arange(4).reshape(2, 2)
    out = _sort_images(frames, np.array([1, 0]))
    assert not isinstance(out, tuple)
    np.testing.assert_array_equal(out, frames[[1, 0]])
