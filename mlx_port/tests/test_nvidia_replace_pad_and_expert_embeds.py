"""``replace_pad_token`` keeps the first ``pad_count`` ids; Stage-2 expert is action-only.

Weight-free. Complementary to open #25 (scatter walks pads in order) and
#32 (base fuse is hist-only; WithFuture also fuses future pads).
``masked_scatter`` raises when ``new_ids`` is shorter than the pad
count, but extra ids are silently dropped (first ``pad_count`` kept).
A longer hist/future encode would fuse the prefix and hide the rest.
Stage-2 sets ``expert_embeds = action_embeds`` so the expert never
sees hist tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"
_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"


def replace_pad_token(input_ids: torch.Tensor, new_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Replica of NVIDIA ``replace_pad_token``."""
    mask = input_ids == pad_idx
    return input_ids.masked_scatter(mask, new_ids)


def test_source_uses_masked_scatter():
    src = _BASE.read_text()
    assert "mask = input_ids == pad_idx" in src
    assert "return input_ids.masked_scatter(mask, new_ids)" in src


def test_source_stage2_expert_is_action_embeds_only():
    src = _S2.read_text()
    assert "expert_embeds = action_embeds" in src
    assert "hist" not in src.split("expert_embeds = action_embeds")[1].split("kv_cache")[0].lower()
    assert "last_traj_future_start_idx = last_traj_future_start_idx[-1, 1] + 1" in src
    assert "if self.config.expert_non_causal_attention:" in src
    assert 'forward_kwargs["is_causal"] = False' in src


def test_scatter_writes_new_ids_in_pad_order():
    ids = torch.tensor([[9, 1, 9, 2, 9]])
    new = torch.tensor([10, 11, 12])
    got = replace_pad_token(ids, new, pad_idx=9)
    assert torch.equal(got, torch.tensor([[10, 1, 11, 2, 12]]))


def test_scatter_rejects_too_few_new_ids():
    ids = torch.tensor([[9, 1, 9]])
    with pytest.raises(RuntimeError):
        replace_pad_token(ids, torch.tensor([10]), pad_idx=9)


def test_scatter_extra_new_ids_keep_first_n():
    """Too many replacement ids do not raise; only the first pad_count are written."""
    ids = torch.tensor([[9, 1, 9]])
    got = replace_pad_token(ids, torch.tensor([10, 11, 12]), pad_idx=9)
    assert torch.equal(got, torch.tensor([[10, 1, 11]]))
