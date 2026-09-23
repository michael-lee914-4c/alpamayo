"""GRPO must not inject Cosmos-RL's flat (B, L) arange into Qwen3-VL.

HF Qwen3-VL skips get_rope_index when position_ids is set. A rank-2 arange
is expanded to identical T/H/W rows, so camera tokens get text-style RoPE
and policy logprobs diverge from the vLLM rollout.
"""

from __future__ import annotations

import torch

from alpamayo_r1.utils.vlm_mrope import (
    apply_policy_ids_to_tokenized_data,
    vlm_mrope_position_ids,
)


def test_flat_arange_is_dropped():
    flat = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, 8)
    assert vlm_mrope_position_ids(flat) is None
    assert vlm_mrope_position_ids(None) is None


def test_three_row_mrope_is_kept():
    rope = torch.arange(8, dtype=torch.long).view(1, 1, 8).expand(3, 2, 8).clone()
    assert vlm_mrope_position_ids(rope) is rope


def test_four_row_generation_ids_are_kept():
    rope = torch.arange(8, dtype=torch.long).view(1, 1, 8).expand(4, 2, 8).clone()
    assert vlm_mrope_position_ids(rope) is rope


def test_bridge_drops_flat_ids_and_clears_stale_key():
    input_ids = torch.ones(2, 8, dtype=torch.long)
    labels_mask = torch.zeros(2, 8, dtype=torch.bool)
    labels_mask[:, -3:] = True
    flat = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, 8)
    td = {"image_grid_thw": torch.tensor([[1, 2, 3]]), "position_ids": flat.clone()}

    apply_policy_ids_to_tokenized_data(
        td, input_ids=input_ids, labels_mask=labels_mask, position_ids=flat
    )

    assert "position_ids" not in td
    assert torch.equal(td["input_ids"], input_ids)
    assert torch.equal(td["labels_mask"], labels_mask)
    assert td["image_grid_thw"].shape == (1, 3)


def test_bridge_keeps_three_row_mrope():
    rope = torch.arange(8, dtype=torch.long).view(1, 1, 8).expand(3, 1, 8).clone()
    td: dict = {}
    apply_policy_ids_to_tokenized_data(td, position_ids=rope)
    assert td["position_ids"] is rope


def test_hf_expands_flat_ids_to_identical_thw_rows():
    """Contract of transformers 4.57 Qwen3-VL language-model position_ids."""
    flat = torch.arange(6, dtype=torch.long).unsqueeze(0)
    expanded = flat[None, ...].expand(3, flat.shape[0], -1)
    assert expanded.shape == (3, 1, 6)
    assert torch.equal(expanded[0], expanded[1])
    assert torch.equal(expanded[1], expanded[2])
