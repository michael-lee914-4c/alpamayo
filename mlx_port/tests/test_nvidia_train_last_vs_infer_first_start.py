"""Stage-2 train crops at the batch-global last start; infer uses first.

Weight-free. Complementary to open #33 (batch-global ``nonzero[-1,1]+1``
KV crop) and #30/#16 (infer first-start offset, missing → ``L-1``).
The same two-marker row therefore crops after the *last* start in train
and blanks attention from the *first* start in infer. Mixing the two
indexers silently shifts expert RoPE / cache length.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
_INFER = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


def _train_crop_idx(input_ids, future_start_token_id):
    last = (input_ids == future_start_token_id).nonzero(as_tuple=False)
    return int(last[-1, 1] + 1)


def _infer_first_offset(sequences, eos_token_id):
    torch = pytest.importorskip("torch")
    start_mask = sequences == eos_token_id
    has_start = start_mask.any(dim=1)
    first = start_mask.int().argmax(dim=1)
    last_token = torch.full((sequences.shape[0],), sequences.shape[1] - 1, dtype=torch.long)
    valid = torch.where(has_start, first, last_token)
    return valid + 1


def test_source_train_last_vs_infer_first():
    sft = _SFT.read_text()
    infer = _INFER.read_text()
    assert "last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1" in sft
    assert "kv_cache.crop(last_traj_future_start_idx)" in sft
    assert "traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)" in infer
    assert "offset = valid_token_pos_id + 1" in infer
    assert "prompt_cache.crop(prefill_seq_len)" in infer


def test_two_markers_crop_and_offset_diverge():
    torch = pytest.importorskip("torch")
    start = 7
    # Row 0 has starts at 1 and 5. Train crop is global last → 6.
    # Infer offset is first-per-row → 2.
    ids = torch.tensor([[0, start, 1, 2, 3, start, 4, 5], [9, 9, 9, 9, 9, 9, 9, 9]])
    assert _train_crop_idx(ids, start) == 6
    infer_off = _infer_first_offset(ids[:1], start)
    torch.testing.assert_close(infer_off, torch.tensor([2]))
    assert int(infer_off[0]) != _train_crop_idx(ids[:1], start)


def test_missing_start_infer_falls_back_to_l_minus_1():
    torch = pytest.importorskip("torch")
    ids = torch.zeros(1, 5, dtype=torch.long)
    torch.testing.assert_close(_infer_first_offset(ids, eos_token_id=99), torch.tensor([5]))
