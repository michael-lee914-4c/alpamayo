"""PAI event-t0 margins are seconds scaled by ``1_000_000`` to microseconds.

Weight-free. Complementary to open #47 (missing ``event_t0s`` column is
a no-op) and #26 (1.6 s / 6.4 s window). ``filter_clips_by_event_t0s``
does ``int(seconds * 1_000_000)``. Treating the constants as already
microseconds (or using milliseconds) shifts the keep-window by 1e3/1e6
and silently drops or keeps the wrong clips.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAI_UTILS = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"


def test_source_converts_seconds_to_microseconds():
    src = _PAI_UTILS.read_text()
    body = src.split("def filter_clips_by_event_t0s")[1].split("def get_all_clip_ids")[0]
    assert "start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)" in body
    assert "end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)" in body
    assert "* 1_000" not in body.replace("* 1_000_000", "")


def test_default_margins_are_1_6s_and_6_4s_in_us():
    start_s, end_s = 1.6, 6.4
    assert int(start_s * 1_000_000) == 1_600_000
    assert int(end_s * 1_000_000) == 6_400_000
    # millisecond mistake: 1.6 * 1000 = 1600 us = 1.6 ms
    assert int(start_s * 1_000) == 1_600
