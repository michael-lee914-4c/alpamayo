"""``fill_masks_between_special_tokens`` zips global nonzero rows.

Weight-free. Complementary to open #24 (zip drops an unpaired extra
start) and #34 (``get_role_eos_mask`` cross-row pairing). A start on
row 0 and an end on row 1 still zip: the mask is written on the
*start's row* using the *end's column*. Start-after-end is a silent
empty slice, not an error.
"""

from __future__ import annotations

from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_MASK = _REPO / "src" / "alpamayo_r1" / "utils" / "get_label_mask.py"


class _Tok:
    def __init__(self, mapping):
        self._ids = mapping

    def convert_tokens_to_ids(self, name):
        return self._ids[name]


def _fill_masks(start_token, end_token, input_ids, tokenizer, labels_mask):
    start_idx = (input_ids == tokenizer.convert_tokens_to_ids(start_token)).nonzero()
    end_idx = (input_ids == tokenizer.convert_tokens_to_ids(end_token)).nonzero()
    for start, end in zip(start_idx, end_idx):
        labels_mask[start[0], start[1] : end[1] + 1] = True
    return labels_mask


def test_source_zips_global_nonzero_and_includes_end():
    src = _MASK.read_text()
    assert "start_idx = (input_ids == tokenizer.convert_tokens_to_ids(start_token)).nonzero()" in src
    assert "end_idx = (input_ids == tokenizer.convert_tokens_to_ids(end_token)).nonzero()" in src
    assert "for start, end in zip(start_idx, end_idx):" in src
    assert "labels_mask[start[0], start[1] : end[1] + 1] = True" in src


def test_cross_row_zip_writes_start_row_with_end_column():
    tok = _Tok({"<|s|>": 1, "<|e|>": 2})
    # Row 0 has only a start at col 1. Row 1 has only an end at col 4.
    ids = torch.tensor(
        [
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0],
        ],
        dtype=torch.long,
    )
    mask = torch.zeros_like(ids, dtype=torch.bool)
    got = _fill_masks("<|s|>", "<|e|>", ids, tok, mask)
    assert got[0].tolist() == [False, True, True, True, True, False]
    assert got[1].tolist() == [False, False, False, False, False, False]


def test_inverted_span_is_empty_not_wrapped():
    tok = _Tok({"<|s|>": 1, "<|e|>": 2})
    ids = torch.tensor([[2, 0, 0, 1, 0]], dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    got = _fill_masks("<|s|>", "<|e|>", ids, tok, mask)
    assert got.sum().item() == 0


def test_same_row_includes_end_token():
    tok = _Tok({"<|s|>": 1, "<|e|>": 2})
    ids = torch.tensor([[0, 1, 9, 9, 2, 0]], dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    got = _fill_masks("<|s|>", "<|e|>", ids, tok, mask)
    assert got.tolist() == [[False, True, True, True, True, False]]
