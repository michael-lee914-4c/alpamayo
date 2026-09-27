"""``replace_padding_after_eos`` is a no-op when a row has no EOS.

Weight-free. Complementary to open #24 (``replace_padding_after_eos``
after first EOS) and #26 (``StopAfterEOS`` one-token-after). Missing EOS
sets ``first_eos_pos = seq_len``, and ``mask_after = positions > seq_len``
is all False, so the generated tokens (including a missing
``<|traj_future_start|>`` fallback) stay intact. Inclusive
``positions >= seq_len`` would wipe the row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def _mask_after(seq_len: int, first_eos_pos: int) -> np.ndarray:
    return np.arange(seq_len) > first_eos_pos


def test_source_missing_eos_uses_seq_len_and_strict_gt():
    src = _TOKEN_UTILS.read_text()
    body = src.split("def replace_padding_after_eos")[1]
    assert "torch.tensor(seq_len, device=token_ids.device)" in body
    assert "first_eos_pos = eos_positions.min(dim=1, keepdim=True)[0]" in body
    assert "mask_after = position_indices > first_eos_pos" in body
    assert "position_indices >= first_eos_pos" not in body


def test_no_eos_keeps_every_token():
    seq_len = 5
    mask = _mask_after(seq_len, first_eos_pos=seq_len)
    assert mask.tolist() == [False] * seq_len


def test_eos_in_middle_pads_after_but_keeps_eos():
    # [a, b, EOS, c, d] → pad c,d ; keep EOS
    mask = _mask_after(5, first_eos_pos=2)
    assert mask.tolist() == [False, False, False, True, True]
