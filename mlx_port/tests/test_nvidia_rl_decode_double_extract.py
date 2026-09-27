"""RL decode strips start/end in text, then ``extract_traj_tokens`` looks again.

Weight-free. Complementary to open #46 (exclusive ``start < idx < end``)
and #32 (``rsplit`` last-start). ``decode_rollout_trajectory`` keeps the
inner span via string split, re-tokenizes it, and only then calls
``extract_traj_tokens``. The inner IDs usually contain neither delimiter,
so missing-start → ``start_positions=-1`` and missing-end → ``seq_len``,
and the exclusive mask keeps *every* token. Re-applying the string
delimiters as token IDs would drop the first/last discrete IDs and shift
the future band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DECODE = _REPO / "finetune" / "rl" / "utils" / "trajectory_decode.py"
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def _strip_delimiters(text: str, start: str, end: str) -> str:
    return text.rsplit(start, maxsplit=1)[-1].split(end, maxsplit=1)[0]


def _exclusive_keep(seq_len: int, start_pos: int, end_pos: int) -> np.ndarray:
    idx = np.arange(seq_len)
    return (idx > start_pos) & (idx < end_pos)


def test_source_strips_then_calls_extract_traj_tokens():
    src = _DECODE.read_text()
    assert 'to_be_evaluated.rsplit(SPECIAL_TOKENS["traj_future_start"], maxsplit=1)' in src
    assert 'SPECIAL_TOKENS["traj_future_end"], maxsplit=1' in src
    assert "traj_token_ids = extract_traj_tokens(" in src
    body = src.split("else:")[1]
    assert "extract_traj_tokens(" in body


def test_source_extract_missing_start_is_minus_one():
    src = _TOKEN_UTILS.read_text()
    body = src.split("def extract_traj_tokens")[1].split("def extract_between_special_tokens")[0]
    assert 'torch.full((batch_size,), -1, dtype=torch.long, device=device)' in body
    assert "torch.full((batch_size,), seq_len, dtype=torch.long, device=device)" in body


def test_stripped_ids_keep_the_whole_sequence():
    start, end = "<|traj_future_start|>", "<|traj_future_end|>"
    inner = _strip_delimiters(f"pre {start} a b c {end} post", start, end)
    assert inner == " a b c "
    # re-tokenized inner has no delimiters → start=-1, end=seq_len
    mask = _exclusive_keep(seq_len=3, start_pos=-1, end_pos=3)
    assert mask.tolist() == [True, True, True]


def test_reinserting_delimiters_would_drop_two_tokens():
    # [START, t0, t1, t2, END] with exclusive span keeps 3; stripped keeps 3
    # if someone put START/END back around 3 tokens, exclusive would keep 3
    # still — the risk is treating the *first/last discrete IDs* as
    # delimiters after a second encode that happens to emit them.
    ids_with_delim = np.array([10, 1, 2, 3, 11])
    mask = _exclusive_keep(5, start_pos=0, end_pos=4)
    assert mask.tolist() == [False, True, True, True, False]
    ids_stripped = np.array([1, 2, 3])
    stripped_mask = _exclusive_keep(3, start_pos=-1, end_pos=3)
    assert ids_stripped[stripped_mask].tolist() == [1, 2, 3]
    assert ids_with_delim[mask].tolist() == [1, 2, 3]
