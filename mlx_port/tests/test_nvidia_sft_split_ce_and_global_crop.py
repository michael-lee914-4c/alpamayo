"""NVIDIA Stage-1 split CE and Stage-2 batch-global KV crop.

Weight-free. Complementary to open #28 (last-start +1 crop / IGNORE=-100)
and #32 (empty-mask CE 0 / token_mask -inf). Stage-1 puts the traj vocab
band *and* the future start/end ids in ``future_traj``, then zeros those
labels before ``others`` so CoC is not double-counted. Stage-2 crop uses
``nonzero(...)[-1, 1] + 1`` over the whole ``[B, L]`` tensor, so a late
start on row 1 moves every row's KV window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_SFT_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
_IGNORE = -100


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_split_ce_traj_mask_includes_band_and_start_end(torch):
    src = _SFT_BASE.read_text()
    assert "(labels >= self.future_token_start_idx)" in src
    assert "(labels < self.future_token_start_idx + self.config.traj_vocab_size)" in src
    assert '(labels == self.special_token_ids["traj_future_start"])' in src
    assert '(labels == self.special_token_ids["traj_future_end"])' in src
    assert "labels[traj_mask] = IGNORE_INDEX" in src
    assert 'losses["others"] = self._compute_next_token_loss' in src

    start_idx, vocab = 200, 10
    start_id, end_id, cot_id = 50, 51, 7
    labels = torch.tensor(
        [
            [_IGNORE, cot_id, start_id, start_idx, start_idx + 3, end_id, cot_id],
        ],
        dtype=torch.long,
    )
    traj_mask = (
        ((labels >= start_idx) & (labels < start_idx + vocab))
        | (labels == start_id)
        | (labels == end_id)
    )
    assert traj_mask.tolist() == [[False, False, True, True, True, True, False]]

    others = labels.clone()
    others[traj_mask] = _IGNORE
    # CoC stays labeled; traj band + markers are dropped before others CE.
    assert others.tolist() == [[_IGNORE, cot_id, _IGNORE, _IGNORE, _IGNORE, _IGNORE, cot_id]]
    assert not ((others != _IGNORE) & traj_mask).any()


def test_stage2_kv_crop_uses_batch_global_last_start(torch):
    src = _SFT_S2.read_text()
    assert "last_traj_future_start_idx = (input_ids == future_start_token_id).nonzero(as_tuple=False)" in src
    assert "last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1" in src
    assert "kv_cache.crop(last_traj_future_start_idx)" in src

    start_id = 9
    # Row 0 start at col 2; row 1 start at col 6. Global last column is 6.
    input_ids = torch.tensor(
        [
            [1, 2, start_id, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, start_id, 8],
        ],
        dtype=torch.long,
    )
    hits = (input_ids == start_id).nonzero(as_tuple=False)
    crop = int(hits[-1, 1] + 1)
    assert crop == 7
    per_row = hits[hits[:, 0] == 0][-1, 1] + 1
    assert int(per_row) == 3
    assert crop != int(per_row)
