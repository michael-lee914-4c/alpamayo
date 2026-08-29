"""mlx_vlm Qwen3-VL text stack used as the Alpamayo diffusion expert.

Stock ``mlx_lm`` Qwen3 RoPEs at ``cache.offset`` and cannot take NVIDIA's
3-row ``position_ids`` or the pad mask. This wrapper calls the same
``Qwen3VLModel`` attention path as the VLM so those tensors pass through.

Weight keys stay ``language_model.model.layers.*`` to match the existing
``expert.* → expert.language_model.model.*`` remap.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLModel


def text_config_from_vlm_and_overrides(
    vlm_text_config: Any,
    expert_overrides: dict[str, Any],
) -> TextConfig:
    """Copy VLM text config, apply ``expert_cfg``, keep GQA / layer count."""
    if hasattr(vlm_text_config, "to_dict"):
        base = vlm_text_config.to_dict()
    elif isinstance(vlm_text_config, dict):
        base = dict(vlm_text_config)
    else:
        base = dict(vlm_text_config)
    merged = dict(base)
    for key, value in expert_overrides.items():
        if key == "dtype":
            continue
        merged[key] = value
    # Expert never uses token embeddings (action tokens arrive as embeds).
    merged["vocab_size"] = 1
    if "model_type" not in merged:
        merged["model_type"] = "qwen3_vl"
    return TextConfig.from_dict(merged)


def cache_seq_len(cache: list[Any]) -> int:
    """True KV length. mlx_lm ``KVCache.offset`` is updated in ``update_and_fetch``.

    Do not use ``_idx``: AlpamayoLanguageModel sets it at the *start* of a VLM
    call, so after generate it is one token behind ``offset``. mlx_vlm
    attention slices the mask with ``_idx`` and concatenates with ``offset``.
    """
    if not cache or cache[0] is None:
        return 0
    return int(cache[0].offset)


def sync_cache_idx(cache: list[Any]) -> None:
    """Set ``_idx = offset`` so mlx_vlm mask length matches the real KV prefix."""
    if not cache:
        return
    for layer_cache in cache:
        if layer_cache is None:
            continue
        layer_cache._idx = int(layer_cache.offset)


def trim_cache(cache: list[Any], n_tokens: int) -> None:
    """Drop the last ``n_tokens`` from every layer (NVIDIA ``cache.crop``)."""
    from mlx_lm.models.cache import trim_prompt_cache

    if n_tokens <= 0 or not cache:
        return
    trim_prompt_cache(cache, int(n_tokens))
    sync_cache_idx(cache)


def traj_future_start_offsets(
    full_sequences: np.ndarray,
    traj_future_start_id: int,
) -> np.ndarray:
    """Index of the token after ``<|traj_future_start|>`` (NVIDIA ``offset``)."""
    arr = np.asarray(full_sequences)
    if arr.ndim == 1:
        arr = arr[None, :]
    batch, length = arr.shape
    offsets = np.empty((batch,), dtype=np.int32)
    for i in range(batch):
        hits = np.flatnonzero(arr[i] == int(traj_future_start_id))
        offsets[i] = int(hits[0]) + 1 if hits.size else length - 1
    return offsets


def expert_position_ids(
    n_diffusion: int,
    batch: int,
    rope_deltas: Any,
    offsets: np.ndarray,
) -> mx.array:
    """``(3, B, T)`` ids: ``arange(T) + rope_deltas + offset``."""
    base = np.arange(n_diffusion, dtype=np.int32)
    pos = np.broadcast_to(base, (3, batch, n_diffusion)).copy()
    rd = np.asarray(rope_deltas, dtype=np.int64).reshape(-1)
    if rd.size == 1:
        rd = np.broadcast_to(rd, (batch,))
    off = np.asarray(offsets, dtype=np.int64).reshape(-1)
    delta = (rd[:batch] + off[:batch]).reshape(1, batch, 1)
    return mx.array(pos.astype(np.int32) + delta.astype(np.int32))


def expert_attention_mask(
    batch: int,
    n_diffusion: int,
    prefix_len: int,
    offsets: np.ndarray,
) -> mx.array:
    """Additive mask ``(B, 1, T, prefix+T)``. Hide pad between offset and diffusion tokens."""
    kv = prefix_len + n_diffusion
    mask = np.zeros((batch, 1, n_diffusion, kv), dtype=np.float32)
    off = np.asarray(offsets).reshape(-1)
    hide = -1e4
    for i in range(batch):
        start = int(off[i])
        if 0 <= start < prefix_len:
            mask[i, :, :, start:prefix_len] = hide
    return mx.array(mask)


class _ExpertLanguageModel(nn.Module):
    """Holds ``model`` so checkpoint keys stay ``language_model.model.layers.*``."""

    def __init__(self, text_config: TextConfig):
        super().__init__()
        self.model = Qwen3VLModel(text_config)


class AlpamayoExpert(nn.Module):
    """Diffusion expert: action embeds in, last hidden states out."""

    def __init__(self, text_config: TextConfig):
        super().__init__()
        self.text_config = text_config
        self.language_model = _ExpertLanguageModel(text_config)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(
        self,
        inputs_embeds: mx.array,
        position_ids: mx.array | None = None,
        cache: list[Any] | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        batch, tokens, _ = inputs_embeds.shape
        dummy = mx.zeros((batch, tokens), dtype=mx.int32)
        return self.language_model.model(
            dummy,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            mask=mask,
            cache=cache,
        )
