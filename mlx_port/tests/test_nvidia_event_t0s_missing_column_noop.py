"""PAI ``filter_clips_by_event_t0s`` is a no-op when the column is absent.

Weight-free. Complementary to open #26 (1.6s / 6.4s window) and #27
(``get_clip_key_frame``). A clip index without ``event_t0s`` returns
before converting margins to microseconds, so every row is kept.
Operators who assume the default 1.6s/6.4s gates still apply would
silently train on events that sit inside the history/future pad.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"


def test_source_returns_immediately_without_event_t0s():
    src = _PAI.read_text()
    body = src.split("def filter_clips_by_event_t0s")[1].split("def get_all_clip_ids")[0]
    assert 'if "event_t0s" not in self.clip_index.columns:' in body
    after_guard = body.split('if "event_t0s" not in self.clip_index.columns:')[1]
    before_margins = after_guard.split("start_margin_us")[0]
    assert "return" in before_margins
    assert "1_000_000" not in before_margins


def test_margin_math_only_runs_after_the_column_exists():
    src = _PAI.read_text()
    body = src.split("def filter_clips_by_event_t0s")[1].split("def get_all_clip_ids")[0]
    after_guard = body.split('if "event_t0s" not in self.clip_index.columns:')[1]
    after_return = after_guard.split("return", 1)[1]
    assert "start_margin_us = int(self.start_safe_margin_seconds * 1_000_000)" in after_return
    assert "end_margin_us = int(self.end_safe_margin_seconds * 1_000_000)" in after_return
