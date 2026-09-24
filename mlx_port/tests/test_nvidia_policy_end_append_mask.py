"""Policy packer appends ``traj_future_end`` and supervises gen+end.

Weight-free. Complementary to open #33 (empty ``gen_ids`` skip mask/id
extend) and #32 (policy ``labels_mask`` only gen+end). Non-empty rollouts
``cat`` prompt + gen + end and set ``labels_mask[-(gen+1):]``. Dropping
the extra slot trains without the stop token; including prompt tokens
leaks the prefix into GRPO logprobs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"


def _append_rollout(ids_row, gen_ids, end_id, text=None):
    torch = pytest.importorskip("torch")
    if text is not None:
        text = text + "<rollout>" + "<|traj_future_end|>"
    if gen_ids.numel() == 0:
        return ids_row.unsqueeze(0), None, text
    end_token_tensor = torch.tensor([end_id], dtype=ids_row.dtype, device=ids_row.device)
    new_ids_row = torch.cat([ids_row, gen_ids.to(ids_row.device), end_token_tensor], dim=0)
    new_mask = torch.zeros_like(new_ids_row, dtype=torch.bool)
    new_mask[-(gen_ids.numel() + 1) :] = True
    return new_ids_row.unsqueeze(0), new_mask.unsqueeze(0), text


def test_source_cats_end_and_masks_gen_plus_one():
    src = _PACKER.read_text()
    assert 'SPECIAL_TOKENS["traj_future_end"]' in src
    assert "new_ids_row = torch.cat([ids_row, gen_ids.to(ids_row.device), end_token_tensor], dim=0)" in src
    assert "new_mask[-(gen_ids.numel() + 1) :] = True" in src
    assert 'tokenized["text"] += SPECIAL_TOKENS["traj_future_end"]' in src
    assert "if gen_ids.numel() > 0:" in src


def test_nonempty_gen_appends_end_and_masks_only_tail():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([1, 2, 3, 4])
    gen = torch.tensor([50, 51, 52])
    new_ids, mask, text = _append_rollout(ids, gen, end_id=99, text="prompt")
    torch.testing.assert_close(new_ids, torch.tensor([[1, 2, 3, 4, 50, 51, 52, 99]]))
    assert mask is not None
    assert mask.dtype == torch.bool
    torch.testing.assert_close(
        mask, torch.tensor([[False, False, False, False, True, True, True, True]])
    )
    assert mask.sum().item() == gen.numel() + 1
    assert text == "prompt<rollout><|traj_future_end|>"


def test_empty_gen_skips_id_and_mask_extend():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([1, 2, 3])
    gen = torch.tensor([], dtype=torch.long)
    new_ids, mask, text = _append_rollout(ids, gen, end_id=99, text="prompt")
    torch.testing.assert_close(new_ids, ids.unsqueeze(0))
    assert mask is None
    assert text.endswith("<|traj_future_end|>")
