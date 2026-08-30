"""Surgical Alpamayo-specific overrides for mlx_vlm Qwen3-VL classes.

These subclasses fix issues discovered during Alpamayo-R1-10B porting:

1. get_rope_index: HF Qwen3-VL layout (get_vision_position_ids + compact
   max(H,W)//merge advance). Stock mlx_vlm follows older Qwen2-VL indexing.

2. get_input_embeddings / __call__: do not let pixel_values wipe cached
   3-row mRoPE; decode continues with arange(L) + cache_offset + rope_deltas.

Only these two methods are overridden; everything else (vision tower,
weight loading, KV cache, etc.) remains identical to mlx_vlm 0.5.0.
"""

import os
from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_DEBUG = os.environ.get("ALPAMAYO_DEBUG", "0") in ("1", "true", "True")


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(msg)

from mlx_vlm.models.qwen3_vl.language import Attention, LanguageModel
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model, InputEmbeddingsFeatures
from mlx_port.stage_timers import current_clock, time_stage, vlm_step_stage
from mlx_lm.models.base import create_causal_mask

from mlx_port.models.rope_index_mlx import compute_hf_rope_index_mx


class DiagnosticAttentionWrapper(nn.Module):
    """Wraps an Attention module to log the exact mask reaching the kernel.

    Used for Option B deeper diagnostics: reveals what mask shape/dtype the
    base mlx_vlm code constructs from the 'causal' sentinel or from an
    explicit mask we supply.
    """

    def __init__(self, orig_attn: Attention, layer_idx: int):
        super().__init__()
        self._orig_attn = orig_attn
        self._layer_idx = layer_idx
        self._call_count = 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        position_ids: Optional[mx.array] = None,
    ) -> mx.array:
        self._call_count += 1
        step = "DECODE" if (mask is None and cache is not None) else "PREFILL"
        if mask is None:
            mask_info = "None"
        else:
            try:
                m = np.asarray(mask)
                mask_info = f"shape={m.shape} dtype={m.dtype} min={float(m.min()):.2f} max={float(m.max()):.2f}"
                # Show a small corner for triangular inspection
                if m.ndim >= 2:
                    corner = m[..., :3, :3]
                    mask_info += f" corner={corner.tolist()}"
            except Exception:
                mask_info = f"type={type(mask)}"
        cache_info = "None"
        if cache is not None:
            off = getattr(cache, "offset", None)
            idx = getattr(cache, "_idx", None)
            cache_info = f"offset={off} _idx={idx}"
        pos_info = "None"
        if position_ids is not None:
            try:
                p = np.asarray(position_ids)
                pos_info = f"shape={p.shape} T0={int(p[0,0,0])}..{int(p[0,0,-1])}"
            except Exception:
                pos_info = "?"
        print(
            f"[ATTN_KERNEL] L{self._layer_idx:02d} call#{self._call_count:03d} | {step} | "
            f"mask={mask_info} | cache={cache_info} | position_ids={pos_info}"
        )
        return self._orig_attn(x, mask=mask, cache=cache, position_ids=position_ids)


def install_attention_diagnostics(model: "AlpamayoLanguageModel", max_layers: int = None) -> int:
    """Replace each decoder layer's self_attn with a DiagnosticAttentionWrapper.

    Returns the number of layers instrumented. Call this after from_existing
    to enable Option B deeper mask logging for Alpamayo runs.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        print("[ATTN_KERNEL] Could not find model.layers; diagnostics not installed.")
        return 0
    layers = model.model.layers
    n = len(layers) if max_layers is None else min(max_layers, len(layers))
    for i in range(n):
        layer = layers[i]
        if hasattr(layer, "self_attn"):
            orig = layer.self_attn
            wrapper = DiagnosticAttentionWrapper(orig, layer_idx=i)
            layer.self_attn = wrapper
    print(f"[ATTN_KERNEL] Installed DiagnosticAttentionWrapper on {n} layers.")
    return n


def inspect_decode_mask_construction(seq_len: int = 1, cache_offset: int = 32766) -> None:
    """Reproduce the mask that mlx_lm / mlx_vlm would build for a decode step.

    For N=1 (decode), create_attention_mask returns None and the kernel
    constructs causality internally using the cache offset. This helper prints
    what the base code sees so we can compare with our stored full-triangular mask.
    """
    from mlx_lm.models.base import create_attention_mask, create_causal_mask

    # Simulate a 1-token decode input
    h = mx.zeros((1, seq_len, 1))  # dummy hidden states
    # Create a fake cache with the given offset
    class FakeCache:
        def __init__(self, offset):
            self.offset = offset
        def make_mask(self, N, return_array=False, window_size=None):
            # The real KVCache.make_mask would return a (1, kv_len) mask
            kv_len = self.offset + N
            if return_array:
                return create_causal_mask(kv_len, window_size=window_size)
            return "causal"

    cache = FakeCache(cache_offset)
    mask = create_attention_mask(h, cache=cache, return_array=False)
    mask_array = create_attention_mask(h, cache=cache, return_array=True)

    print(f"[MASK_INSPECT] decode step: seq_len={seq_len}, cache_offset={cache_offset}")
    print(f"[MASK_INSPECT]   create_attention_mask(...) -> {mask!r}")
    if mask_array is not None:
        print(f"[MASK_INSPECT]   return_array=True -> shape={mask_array.shape}, dtype={mask_array.dtype}")
        # Show the last few columns that would be used for this decode token
        tail = mask_array[0, -min(8, mask_array.shape[1]):]
        print(f"[MASK_INSPECT]   last 8 cols of row 0: {tail.tolist()}")
    else:
        print("[MASK_INSPECT]   return_array=True -> None (kernel will build internally)")


class AlpamayoLanguageModel(LanguageModel):
    """LanguageModel with corrected vision-aware RoPE index computation and
    guaranteed preservation of multimodal position state across decode steps.
    """

    @classmethod
    def from_existing(cls, base: LanguageModel) -> "AlpamayoLanguageModel":
        """Create an AlpamayoLanguageModel instance from a loaded base.

        This method returns a true AlpamayoLanguageModel (proper type and MRO)
        while preserving all loaded weights and submodules.
        """
        # Use promotion internally for robustness with nn.Module registration,
        # but expose it through an explicit from_existing API as requested.
        base.__class__ = cls
        # Initialize Alpamayo-specific state if not already present
        if not hasattr(base, "_position_ids"):
            base._position_ids = None
        if not hasattr(base, "_rope_deltas"):
            base._rope_deltas = None
        if not hasattr(base, "_attention_mask"):
            base._attention_mask = None
        # Explicit validity flag for the idempotency guard (more robust than
        # checking attribute presence, which can be affected by nn.Module internals)
        if not hasattr(base, "_pos_valid"):
            base._pos_valid = False
        from mlx_port.models.compiled_backbone import install_compiled_prefill

        install_compiled_prefill(base)
        return base

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        # --- DIAGNOSTIC: mask / cache / 3-row mRoPE position_ids interaction (prioritized) ---
        cache_offset0 = 0
        if cache is not None and len(cache) > 0:
            for c in cache:
                if c is None:
                    continue
                off = getattr(c, "offset", 0)
                if hasattr(off, "shape"):
                    off_i = int(np.asarray(off).reshape(-1)[0])
                else:
                    off_i = int(off or 0)
                # Always sync. A stale _idx=0 left from prefill made decode
                # slice position_ids[:, :, 0:1] (the first column, all zeros).
                c._idx = off_i
            if cache[0] is not None:
                cache_offset0 = int(getattr(cache[0], "_idx", 0) or 0)
        is_decode_step = inputs.shape[1] == 1 and bool(cache_offset0)
        step_kind = "DECODE" if is_decode_step else "PREFILL"
        mask_info = "None" if mask is None else f"shape={mask.shape} dtype={mask.dtype}"
        cache_info = "None"
        if cache is not None and len(cache) > 0:
            c0 = cache[0]
            offset = getattr(c0, "offset", None)
            idx = getattr(c0, "_idx", None)
            cache_info = f"len={len(cache)} offset={offset} _idx={idx}"

        # Capture position_ids that will be injected (the 3-row mRoPE state)
        pos_in_kwargs = kwargs.get("position_ids", None)
        pos_info = "None"
        if pos_in_kwargs is not None:
            try:
                pos_np = np.asarray(pos_in_kwargs)
                pos_info = f"shape={pos_np.shape} T=[{int(pos_np[0,0,0])}..{int(pos_np[0,0,-1])}] H=[{int(pos_np[1,0,0])}..{int(pos_np[1,0,-1])}] W=[{int(pos_np[2,0,0])}..{int(pos_np[2,0,-1])}]"
            except Exception:
                pos_info = f"shape={getattr(pos_in_kwargs, 'shape', '?')}"

        _dbg(f"[ATTN_DIAG] {step_kind} | mask={mask_info} | cache={cache_info} | inputs.shape={inputs.shape} | position_ids={pos_info}")

        # Prefill-only: the stored mask is (S, S). Applying it on L=1 decode
        # (when _position_ids was wiped and is_decode_step became False) is what
        # made decode look like another prefill. Let the kernel use cache offset.
        stored = getattr(self, "_attention_mask", None)
        if (
            mask is None
            and stored is not None
            and hasattr(stored, "shape")
            and inputs.shape[1] == stored.shape[-1]
        ):
            mask = stored
            _dbg("[ATTN_DIAG]   using stored prefill mask (seq_len matches)")

        outputs = super().__call__(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        if _DEBUG and is_decode_step:
            try:
                logits = outputs.logits
                mx.eval(logits)
                logits_np = np.asarray(logits[0, 0, :])
                top_idx = int(np.argmax(logits_np))
                top_val = float(np.max(logits_np))
                _dbg(
                    f"[ATTN_DIAG] first-decode logits: shape={logits.shape} "
                    f"max={top_val:.3f} argmax={top_idx}"
                )
            except Exception as e:
                _dbg(f"[ATTN_DIAG] first-decode logits inspect failed: {e}")

        return outputs

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
        mm_token_type_ids: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        # Idempotency: prefill computes 3-row mRoPE once; decode uses
        # arange(L) + cache_offset + _rope_deltas in the base LanguageModel.
        if getattr(self, "_pos_valid", False) and getattr(self, "_position_ids", None) is not None:
            _dbg("[ROPE_DEBUG] guard short-circuit (cached HF RoPE)")
            return self._position_ids, getattr(self, "_rope_deltas", None)
        if getattr(self, "_pos_valid", False) and getattr(self, "_position_ids", None) is None:
            _dbg("[ROPE_DEBUG] guard skipped: _pos_valid but _position_ids is None")
            self._pos_valid = False

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            spatial_merge_size = int(self.config.vision_config.spatial_merge_size)
            position_ids, mrope_position_deltas = compute_hf_rope_index_mx(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                image_token_id=int(self.config.image_token_id),
                video_token_id=int(self.config.video_token_id),
                spatial_merge_size=spatial_merge_size,
            )
            max_pos = int(np.asarray(position_ids).max())
            _dbg(
                f"[ROPE_DEBUG] HF get_rope_index: shape={position_ids.shape} "
                f"max_pos={max_pos} mrope_deltas={np.asarray(mrope_position_deltas).tolist()} "
                f"seq_len={input_ids.shape[1]}"
            )
            try:
                tail_ids = np.asarray(input_ids[0, -8:]).tolist()
                tail_pos = np.asarray(position_ids[:, 0, -8:])
                _dbg(f"[ROPE_DEBUG] last_8_input_ids={tail_ids}")
                _dbg(f"[ROPE_DEBUG] last_8_THW={tail_pos.tolist()}")
            except Exception as e:
                _dbg(f"[ROPE_DEBUG] tail diagnostic failed: {e}")
            self._position_ids = position_ids
            self._rope_deltas = mrope_position_deltas
            self._pos_valid = True
            return position_ids, mrope_position_deltas

        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            position_ids = mx.broadcast_to(position_ids[None, :, :], (3, *position_ids.shape))
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, input_ids.shape[0], input_ids.shape[1]))
            mrope_position_deltas = mx.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


class AlpamayoModel(Model):
    """Model that preserves cached multimodal position state across decode steps."""

    @classmethod
    def from_existing(cls, base: Model) -> "AlpamayoModel":
        """Create an AlpamayoModel instance from a loaded base VLM.

        Returns a true AlpamayoModel (proper type) while preserving all
        loaded weights, vision tower, and language model.
        """
        base.__class__ = cls
        return base

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        # --- FORCED DIAGNOSTIC (get_input_embeddings path, which is always hit) ---
        m = kwargs.get("mask", None)
        mask_info = "None" if m is None else f"shape={m.shape} dtype={m.dtype}"
        _dbg(f"[ATTN_DIAG] get_input_embeddings | mask={mask_info} | pixel_values={'present' if pixel_values is not None else 'None'} | input_ids.shape={input_ids.shape if input_ids is not None else 'None'}")

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        if pixel_values is None:
            # Decode / text-only: do not mutate _position_ids. Official Qwen3-VL
            # continuation is position = arange(L) + cache_offset + _rope_deltas,
            # computed inside LanguageModel.__call__ when those fields survive.
            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # --- vision path unchanged ---
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        with time_stage("encode"):
            if cached is not None:
                hidden_states = cached
                deepstack_visual_embeds = None
            else:
                hidden_states, deepstack_visual_embeds = self.vision_tower(
                    pixel_values, grid_thw
                )
            # Barrier only when T1.1 is on so encode_ms is not lazy-attributed
            # to prefill. Token graph is unchanged.
            if current_clock() is not None:
                mx.eval(hidden_states)
                if deepstack_visual_embeds is not None:
                    mx.eval(deepstack_visual_embeds)

        visual_pos_masks = None
        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        mx.eval(deepstack_visual_embeds)

        if image_grid_thw is not None or video_grid_thw is not None:
            # Explicitly store the ROPE state on the language model so that
            # the base LanguageModel.__call__ sees it immediately (avoiding
            # the "position_ids is None" fallback path).
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                mask,
                kwargs.get("mm_token_type_ids"),
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas
            self.language_model._pos_valid = True

            # Build explicit causal mask here, together with the ROPE state we just computed.
            # This keeps mask construction inside the Alpamayo-managed path so the base
            # class does not trigger a second get_rope_index call when it sees a mask.
            if mask is None:
                causal_mask = create_causal_mask(input_ids.shape[1])
                self.language_model._attention_mask = causal_mask
                _dbg(f"[ATTN_DIAG]   built explicit causal mask inside get_input_embeddings (seq_len={input_ids.shape[1]})")

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ):
        """Merge vision embeds, then run the LM *without* pixel_values.

        mlx_vlm LanguageModel.__call__ treats pixel_values as "new image" and
        sets ``_position_ids = _rope_deltas = None``. Our get_rope_index guard
        then returned (None, None), so attention fell back to a flat arange and
        CoC logits collapsed after step 1.
        """
        feats = self.get_input_embeddings(input_ids, pixel_values, **kwargs)
        kwargs.update({k: v for k, v in feats.to_dict().items() if v is not None})
        kwargs.pop("pixel_values", None)
        lm = self.language_model
        if getattr(lm, "_rope_deltas", None) is not None:
            kwargs["rope_deltas"] = lm._rope_deltas
        _dbg(
            f"[ROPE] LM call without pixel_values | "
            f"_pos_valid={getattr(lm, '_pos_valid', None)} "
            f"_position_ids={'set' if getattr(lm, '_position_ids', None) is not None else 'None'} "
            f"_rope_deltas={getattr(lm, '_rope_deltas', None)}"
        )
        # Prefill = first VLM step (prompt seq). After token 1, one-token
        # calls are decode on a warm KV. Barrier only when T1.1 is on.
        stage = vlm_step_stage(input_ids)
        with time_stage(stage):
            out = lm(input_ids, mask=mask, cache=cache, **kwargs)
            if current_clock() is not None:
                logits = getattr(out, "logits", out)
                mx.eval(logits)
        return out