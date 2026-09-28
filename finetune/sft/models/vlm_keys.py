# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Key alignment for Stage-1 Trainer ``vlm.*`` shards."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

VLM_KEY_PREFIX = "vlm."


def align_vlm_state_dict_keys(
    vlm_state_dict: Mapping[str, Any], dest_keys: Iterable[str]
) -> dict[str, Any]:
    """Map Stage-1 ``vlm.*`` checkpoint keys onto the destination module.

    HuggingFace Trainer checkpoints store wrapper names (``vlm.model...``).
    ``Qwen3VLForConditionalGeneration`` uses the same names without ``vlm.``.
    Pass the full Alpamayo/ReasoningVLA wrapper when possible; if the
    destination is the inner VLM, strip the prefix so the tensors apply.
    """
    dest_has_prefix = any(k.startswith(VLM_KEY_PREFIX) for k in dest_keys)
    src_has_prefix = any(k.startswith(VLM_KEY_PREFIX) for k in vlm_state_dict)
    if src_has_prefix and not dest_has_prefix:
        n = len(VLM_KEY_PREFIX)
        return {
            (k[n:] if k.startswith(VLM_KEY_PREFIX) else k): v
            for k, v in vlm_state_dict.items()
        }
    return dict(vlm_state_dict)


def apply_aligned_vlm_state_dict(model: Any, vlm_state_dict: Mapping[str, Any], checkpoint_dir: Any = None):
    """``load_state_dict`` after prefix alignment; raise if nothing matched."""
    dest_keys = model.state_dict().keys()
    aligned = align_vlm_state_dict_keys(vlm_state_dict, dest_keys)
    load_result = model.load_state_dict(aligned, strict=False, assign=True)
    n_applied = len(aligned) - len(load_result.unexpected_keys)
    if n_applied == 0:
        where = f" from checkpoint: {checkpoint_dir}" if checkpoint_dir is not None else ""
        raise ValueError(
            f"No vlm.* tensors matched parameters on {type(model).__name__}{where} "
            f"(unexpected={len(load_result.unexpected_keys)}). "
            "Stage-1 Trainer shards use a 'vlm.' prefix; pass the full "
            "Alpamayo wrapper or an inner VLM whose keys match after strip."
        )
    return n_applied, load_result
