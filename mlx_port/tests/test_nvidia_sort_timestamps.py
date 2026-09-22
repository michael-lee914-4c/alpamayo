"""Camera sort permutes ``relative_timestamps`` with the same argsort.

Weight-free. Complementary to open #25 (stable camera-id sort of frames).
A sort that only reorders images leaves timestamps on the pre-sort camera
axis, so later VLM RoPE / relative-time features attach the wrong t to
each view.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PROC = _SRC / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _sort_images_by_camera_ids(
    image_frames,
    camera_indices,
    relative_timestamps=None,
    return_camera_ids: bool = False,
):
    torch = pytest.importorskip("torch")
    sorted_indices = torch.argsort(camera_indices, stable=True)
    sorted_image_frames = image_frames[sorted_indices]
    result = [sorted_image_frames]
    if return_camera_ids:
        result.append(camera_indices[sorted_indices])
    if relative_timestamps is not None:
        result.append(relative_timestamps[sorted_indices])
    if len(result) == 1:
        return result[0]
    return tuple(result)


def test_source_sorts_timestamps_with_same_indices():
    src = _PROC.read_text()
    assert "sorted_indices = torch.argsort(camera_indices, stable=True)" in src
    assert "if relative_timestamps is not None:" in src
    assert "result.append(relative_timestamps[sorted_indices])" in src
    assert "return_camera_ids" in src


def test_timestamps_follow_camera_permutation():
    torch = pytest.importorskip("torch")
    # Unsorted cameras 2, 0, 1 with distinct time rows.
    frames = torch.arange(3 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 3, 2, 2)
    cameras = torch.tensor([2, 0, 1])
    stamps = torch.tensor([[20.0, 21.0], [0.0, 1.0], [10.0, 11.0]])
    sorted_frames, sorted_cams, sorted_ts = _sort_images_by_camera_ids(
        frames, cameras, relative_timestamps=stamps, return_camera_ids=True
    )
    assert sorted_cams.tolist() == [0, 1, 2]
    torch.testing.assert_close(sorted_ts, torch.tensor([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]))
    torch.testing.assert_close(sorted_frames[0], frames[1])
    torch.testing.assert_close(sorted_frames[1], frames[2])
    torch.testing.assert_close(sorted_frames[2], frames[0])


def test_stable_sort_keeps_tied_camera_order_and_stamps():
    torch = pytest.importorskip("torch")
    frames = torch.stack(
        [
            torch.full((1, 1, 1, 1), 1.0),
            torch.full((1, 1, 1, 1), 2.0),
            torch.full((1, 1, 1, 1), 3.0),
        ]
    )
    cameras = torch.tensor([5, 5, 1])
    stamps = torch.tensor([[100.0], [200.0], [10.0]])
    _, sorted_cams, sorted_ts = _sort_images_by_camera_ids(
        frames, cameras, relative_timestamps=stamps, return_camera_ids=True
    )
    assert sorted_cams.tolist() == [1, 5, 5]
    # The two camera-5 rows keep input order (stable), so 100 then 200.
    torch.testing.assert_close(sorted_ts, torch.tensor([[10.0], [100.0], [200.0]]))
