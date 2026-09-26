"""``extract_traj_tokens`` keeps tokens strictly between start and end.

Weight-free. Complementary to open #35 (overflow ``cumsum < N``) and #44
(first-end vs last-start). The valid mask is ``start < idx < end``, so the
``<|traj_future_start|>`` / ``<|traj_future_end|>`` delimiters are never
counted toward ``tokens_per_future_traj``. Inclusive bounds would silently
shift the future-band IDs and pad/clamp the wrong tokens into decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def _exclusive_span(seq_len: int, start_pos: int, end_pos: int) -> np.ndarray:
    idx = np.arange(seq_len)
    return (idx > start_pos) & (idx < end_pos)


def test_source_uses_strict_inequalities():
    src = _TOKEN_UTILS.read_text()
    assert "range_tensor > start_positions.unsqueeze(1)" in src
    assert "range_tensor < end_positions.unsqueeze(1)" in src
    assert "range_tensor >= start_positions" not in src
    assert "range_tensor <= end_positions" not in src


def test_start_and_end_tokens_are_excluded():
    # [pad, START, t1, t2, END, pad] → keep only t1, t2
    mask = _exclusive_span(seq_len=6, start_pos=1, end_pos=4)
    assert mask.tolist() == [False, False, True, True, False, False]
    assert int(mask.sum()) == 2


def test_inclusive_bounds_would_count_delimiters():
    seq_len, start_pos, end_pos = 6, 1, 4
    exclusive = _exclusive_span(seq_len, start_pos, end_pos)
    inclusive = (np.arange(seq_len) >= start_pos) & (np.arange(seq_len) <= end_pos)
    assert int(inclusive.sum()) == 4
    assert int(exclusive.sum()) == 2
    assert inclusive[start_pos] and inclusive[end_pos]
    assert not exclusive[start_pos] and not exclusive[end_pos]
