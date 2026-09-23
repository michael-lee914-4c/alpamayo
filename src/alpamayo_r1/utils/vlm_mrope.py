# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL mRoPE filters for policy / SFT forwards.

Hugging Face Qwen3-VL (4.57 and 5.x) skips ``get_rope_index`` whenever
``position_ids`` is not None. A rank-2 ``(B, L)`` arange is then expanded
to identical temporal/height/width rows, so vision tokens get text-style
RoPE. Official SFT never passes ``position_ids``. Cosmos-RL's
``get_position_ids`` returns that flat arange for sequence-parallel
slicing and must not be forwarded into the VLM.
"""

from __future__ import annotations

from typing import Any

import torch


def vlm_mrope_position_ids(
    position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    """Keep 3+ row mRoPE ids. Drop a flat ``(B, L)`` sequence index."""
    if position_ids is None:
        return None
    if int(position_ids.ndim) >= 3:
        return position_ids
    return None


def apply_policy_ids_to_tokenized_data(
    tokenized_data: dict[str, Any],
    *,
    input_ids: torch.Tensor | None = None,
    labels_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Copy trainer tensors into ``tokenized_data`` without breaking mRoPE."""
    if labels_mask is not None:
        tokenized_data["labels_mask"] = labels_mask
    rope_ids = vlm_mrope_position_ids(position_ids)
    if rope_ids is not None:
        tokenized_data["position_ids"] = rope_ids
    else:
        tokenized_data.pop("position_ids", None)
    if input_ids is not None:
        tokenized_data["input_ids"] = input_ids
    return tokenized_data
