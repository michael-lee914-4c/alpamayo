"""NVIDIA empty-rollout policy skip and logprob fallback.

Weight-free. Complementary to open #32 (policy mask = gen+end only).
``get_policy_input`` leaves ``input_ids`` / ``labels_mask`` unchanged
when ``gen_ids.numel() == 0``, but still decodes the empty span for
reward. ``policy_collate_fn`` then builds an all-True logprob mask
with token 0 forced False — GRPO would otherwise score the BOS as a
completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_empty_gen_ids_skip_mask_and_id_extend(torch):
    src = _PACKER.read_text()
    assert "if gen_ids.numel() > 0:" in src
    assert "new_mask[-(gen_ids.numel() + 1) :] = True  # include the end token" in src
    # Decode still runs after the empty-gen skip.
    after = src.split("if gen_ids.numel() > 0:", 1)[1]
    assert "predicted_fut_xyz, predicted_fut_rot = decode_rollout_trajectory(" in after

    ids_row = torch.arange(5, dtype=torch.long)
    gen_ids = torch.tensor([], dtype=torch.long)
    tokenized = {"input_ids": ids_row.unsqueeze(0).clone()}
    if gen_ids.numel() > 0:
        tokenized["input_ids"] = torch.cat(
            [ids_row, gen_ids, torch.tensor([99])]
        ).unsqueeze(0)
        tokenized["labels_mask"] = torch.ones(1, 6, dtype=torch.bool)
    assert tokenized["input_ids"].tolist() == [[0, 1, 2, 3, 4]]
    assert "labels_mask" not in tokenized


def test_logprob_fallback_excludes_first_token(torch):
    src = _PACKER.read_text()
    assert "mask[:, 0] = False" in src
    assert 'raise KeyError("tokenized_data.input_ids is required to construct logprob_masks")' in src

    input_ids = torch.arange(8, dtype=torch.long).view(2, 4)
    batch: dict = {}
    if "labels_mask" in batch:
        batch["logprob_masks"] = batch["labels_mask"]
    else:
        b, t = input_ids.shape
        mask = torch.ones(b, t, dtype=torch.bool)
        mask[:, 0] = False
        batch["logprob_masks"] = mask
    assert batch["logprob_masks"].tolist() == [
        [False, True, True, True],
        [False, True, True, True],
    ]
