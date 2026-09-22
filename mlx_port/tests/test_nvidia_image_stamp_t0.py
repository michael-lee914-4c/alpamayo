"""Camera frames include t0 and walk back ``(num_frames-1)*dt``.

Weight-free. Complementary to open #28 (history includes t0 / future
starts at t0+dt). Image stamps are a different formula:
``t0 - (num_frames-1-i)*dt``. Dropping t0 or starting at t0-dt would
desync the front-wide t0 cell (``frames[7]``) from the ego origin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_LOAD = _REPO / "src" / "alpamayo_r1" / "load_physical_aiavdataset.py"


def _image_timestamps(t0_us: int, num_frames: int, time_step: float) -> np.ndarray:
    return np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )


def test_source_image_stamps_include_t0_and_walk_back():
    src = _LOAD.read_text()
    assert "t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000)" in src
    assert "for i in range(num_frames)" in src
    # History ends at t0; future starts after t0 — images must still land on t0.
    assert "History: [..., t0-0.2s, t0-0.1s, t0]" in src
    assert "Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s]" in src
    assert "[t0-0.3s, t0-0.2s, t0-0.1s, t0]" in src


def test_default_four_frames_end_at_t0():
    t0 = 5_100_000
    stamps = _image_timestamps(t0, num_frames=4, time_step=0.1)
    assert stamps.tolist() == [4_800_000, 4_900_000, 5_000_000, 5_100_000]
    assert stamps[-1] == t0
    assert stamps[0] == t0 - 3 * 100_000
    # Not the future-offset start (t0+dt) and not a t0-exclusive window.
    assert t0 + 100_000 not in stamps.tolist()
    assert stamps.shape == (4,)


def test_single_frame_is_exactly_t0():
    t0 = 2_000_000
    stamps = _image_timestamps(t0, num_frames=1, time_step=0.1)
    assert stamps.tolist() == [t0]
