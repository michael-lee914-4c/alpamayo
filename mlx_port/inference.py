"""MLX-native inference rollout for AlpamayoR1 (clean version after subclass refactor).

This module now relies on the surgical Alpamayo-specific subclasses in
`alpamayo_qwen3vl.py` (AlpamayoLanguageModel + AlpamayoModel) instead of
layering post-processing workarounds after every VLM call.

All previous rope_deltas / position_ids post-processing has been removed.
"""

from typing import Any, Dict, Tuple
import gc
import os
import time

_DEBUG = os.environ.get("ALPAMAYO_DEBUG", "0") in ("1", "true", "True")

import mlx.core as mx
import numpy as np
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.profiling import (
    is_profiling_enabled,
    StepProfiler,
    record_memory_sample,
    get_global_memory_peak,
    MemoryMonitor,
)
from mlx_lm.models.cache import KVCache
from mlx_port.models.token_utils_mlx import (
    ExpertLogitsProcessor,
    StopAfterEOS,
    replace_padding_after_eos,
    extract_text_tokens,
)
from mlx_port.models.alpamayo_qwen3vl import AlpamayoModel, AlpamayoLanguageModel


def apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """HuggingFace-style nucleus mask on already temperature-scaled logits.

    Keeps the smallest prefix of tokens (highest logit first) whose softmax
    mass is at least ``top_p``. The token that crosses the threshold is kept.
    ``top_p >= 1`` is a no-op.
    """
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    arr = np.asarray(logits.astype(mx.float32), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    out = arr.copy()
    for b in range(arr.shape[0]):
        row = arr[b]
        order = np.argsort(-row, kind="stable")
        sorted_logits = row[order]
        finite = np.isfinite(sorted_logits)
        if not finite.any():
            continue
        offset = np.max(sorted_logits[finite])
        exp = np.where(finite, np.exp(sorted_logits - offset), 0.0)
        total = exp.sum()
        if total <= 0:
            continue
        csum = np.cumsum(exp / total)
        remove = csum > top_p
        # Shift right so the token that crosses top_p stays (HF TopPLogitsWarper).
        remove[1:] = remove[:-1]
        remove[0] = False
        out[b, order[remove]] = np.float32("-inf")
    return mx.array(out, dtype=mx.float32)


def sample_next_token(
    logits: mx.array,
    temperature: float = 0.6,
    top_p: float = 0.98,
) -> mx.array:
    """Sample one token id per batch row. ``temperature == 0`` is greedy argmax."""
    if logits.ndim == 1:
        logits = logits[None, :]
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)

    scaled = logits.astype(mx.float32)
    if temperature != 1.0:
        scaled = scaled / temperature
    scaled = apply_top_p(scaled, top_p)
    return mx.random.categorical(scaled)


def _as_numpy_ids(input_ids: Any) -> np.ndarray:
    if isinstance(input_ids, mx.array):
        return np.array(input_ids)
    return np.asarray(input_ids)


def _fuse_and_report(model: AlpamayoR1MLX, input_ids: Any, traj_data: Dict[str, Any]) -> Any:
    """Fuse history traj pads and print before/after counts (step 2 diagnostic)."""
    pad_id = model.traj_token_ids.get("history")
    start = int(getattr(model, "traj_token_start_idx", -1))
    vocab = int(getattr(model, "traj_vocab_size", 0))
    before = _as_numpy_ids(input_ids)
    n_pad_before = int((before == pad_id).sum()) if pad_id is not None else -1
    n_in_before = int(((before >= start) & (before < start + vocab)).sum()) if vocab else -1

    fused = model.fuse_traj_tokens(input_ids, traj_data)
    after = _as_numpy_ids(fused)
    n_pad_after = int((after == pad_id).sum()) if pad_id is not None else -1
    n_in_after = int(((after >= start) & (after < start + vocab)).sum()) if vocab else -1
    if _DEBUG:
        print(
            f"[FUSE] hist_pad_id={pad_id}  pads {n_pad_before}->{n_pad_after}  "
            f"<iN> in prompt {n_in_before}->{n_in_after}  "
            f"(expect 48 pads consumed, 48 <iN> inserted)"
        )
        if n_pad_before != 48 or n_pad_after != 0 or (n_in_after - n_in_before) != 48:
            print("[FUSE] WARNING: history fusion did not replace exactly 48 pads")
        tail = after[0, -20:] if after.ndim == 2 else after[-20:]
        try:
            decoded = model.tokenizer.decode([int(t) for t in tail.tolist()])
        except Exception:
            decoded = "?"
        print(f"[FUSE] last_20_ids={tail.tolist()}")
        print(f"[FUSE] last_20_decoded={decoded!r}")
    elif n_pad_before != 48 or n_pad_after != 0 or (n_in_after - n_in_before) != 48:
        print(
            f"[FUSE] WARNING: history fusion did not replace exactly 48 pads "
            f"({n_pad_before}->{n_pad_after} pads, <iN> {n_in_before}->{n_in_after})"
        )
    return fused


def _decode_id(tokenizer, tid: int) -> str:
    try:
        return tokenizer.decode([int(tid)])
    except Exception:
        return f"<{int(tid)}>"


def _lookup_token_id(tokenizer, text: str) -> int | None:
    tid = tokenizer.convert_tokens_to_ids(text)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is not None and tid != unk and tid >= 0:
        return int(tid)
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        return None
    if len(ids) == 1:
        return int(ids[0])
    return None


def dump_prefill_next_token(
    tokenizer,
    raw_logits: mx.array,
    processed_logits: mx.array | None = None,
    last_token_id: int | None = None,
    k: int = 15,
) -> None:
    """Print P(next | prompt) at the last prefill position (``<|cot_start|>``)."""
    raw = np.asarray(raw_logits.astype(mx.float32))
    if raw.ndim == 2:
        raw = raw[0]
    probs = np.exp(raw - raw.max())
    probs = probs / probs.sum()
    last_s = "?"
    if last_token_id is not None:
        last_s = _decode_id(tokenizer, last_token_id)
    print(f"[PREFILL] last_prompt_token={last_s!r} id={last_token_id}")
    print(f"[PREFILL] logits shape={tuple(np.asarray(raw_logits).shape)} vocab={raw.shape[0]}")

    def _dump_topk(label: str, row_logits: np.ndarray) -> None:
        p = np.exp(row_logits - row_logits.max())
        p = p / p.sum()
        idx = np.argsort(-p)[:k]
        bits = [f"{_decode_id(tokenizer, int(i))!r}({p[int(i)]:.4f} id={int(i)})" for i in idx]
        print(f"[PREFILL] {label} top-{k}:")
        for b in bits:
            print(f"  {b}")

    _dump_topk("raw", raw)
    if processed_logits is not None:
        proc = np.asarray(processed_logits.astype(mx.float32))
        if proc.ndim == 2:
            proc = proc[0]
        _dump_topk("after ExpertLogitsProcessor", proc)

    watch = [
        "<|im_end|>",
        "<|cot_end|>",
        "<|traj_future_start|>",
        "<|cot_start|>",
        "Keep",
        " Slow",
        "Slow",
        " Come",
        "Come",
        " Turn",
        " The",
    ]
    print("[PREFILL] watched tokens (raw softmax):")
    for name in watch:
        tid = _lookup_token_id(tokenizer, name)
        if tid is None or tid >= raw.shape[0]:
            print(f"  {name!r}: not a single token")
            continue
        rank = int((probs > probs[tid]).sum()) + 1
        print(
            f"  {name!r} id={tid} p={probs[tid]:.4f} rank={rank} "
            f"decoded={_decode_id(tokenizer, tid)!r}"
        )


def _image_kwargs_from_tokenized(tokenized_data: Dict[str, Any]) -> Dict[str, mx.array]:
    image_kwargs = {}
    for k, v in tokenized_data.items():
        if k in ("pixel_values", "pixel_values_videos"):
            arr = np.asarray(v)
            if arr.ndim == 5 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (0, 4, 1, 2, 3))
            image_kwargs[k] = mx.array(arr)
        elif k in ("image_grid_thw", "video_grid_thw"):
            image_kwargs[k] = mx.array(v, dtype=mx.int32)
    return image_kwargs


def _rewind_kv_cache(cache: list, offset: int) -> None:
    """Drop decode tokens so the next branch starts from the prefill cache."""
    for c in cache:
        extra = int(c.offset) - int(offset)
        if extra > 0:
            c.trim(extra)
        c._idx = int(offset)


def _greedy_continue_from_first(
    vlm,
    cache,
    first_token_id: int,
    logits_processor: ExpertLogitsProcessor,
    eos_token_id: int,
    max_new_tokens: int,
) -> list[int]:
    generated = [int(first_token_id)]
    stopper = StopAfterEOS(eos_token_id=eos_token_id)
    if stopper(mx.array([generated])):
        return generated
    outputs = vlm(input_ids=mx.array([[first_token_id]]), cache=cache)
    mx.eval(outputs.logits)
    for _ in range(max(0, max_new_tokens - 1)):
        logits = logits_processor(generated, outputs.logits[:, -1, :])
        next_id = int(mx.argmax(logits.astype(mx.float32), axis=-1).item())
        generated.append(next_id)
        if stopper(mx.array([generated])):
            break
        outputs = vlm(input_ids=mx.array([[next_id]]), cache=cache)
        mx.eval(outputs.logits)
    return generated


def generate_top_k_coc(
    model: AlpamayoR1MLX,
    data: Dict[str, Any],
    k: int = 5,
    max_generation_length: int = 256,
) -> list[dict]:
    """Prefill once, then greedy-complete each of the top-k first tokens.

    First-token ranks use softmax after ``ExpertLogitsProcessor`` (traj bins
    masked), matching the greedy policy. Each branch rewinds the KV cache to
    the prefill offset so vision is not recomputed.
    """
    tokenized_data = data["tokenized_data"]
    input_ids = tokenized_data["input_ids"]
    if isinstance(input_ids, list):
        input_ids = mx.array(input_ids)
    image_kwargs = _image_kwargs_from_tokenized(tokenized_data)
    input_ids = _fuse_and_report(
        model,
        input_ids,
        {
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
    )

    eos_token_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if eos_token_id is None:
        eos_token_id = model.tokenizer.eos_token_id
    logits_processor = ExpertLogitsProcessor(
        traj_token_offset=model.traj_token_start_idx,
        traj_vocab_size=model.traj_vocab_size,
        traj_token_ids=getattr(model, "traj_token_id_list", None),
    )

    vlm = model.vlm
    n_layers = len(vlm.language_model.model.layers)
    cache = [KVCache() for _ in range(n_layers)]
    with MemoryMonitor(poll_interval=0.05, label="vlm_prefill"):
        outputs = vlm(input_ids=input_ids, **image_kwargs, cache=cache)
    mx.eval(outputs.logits)
    prefill_offset = int(cache[0].offset)

    raw_last = outputs.logits[:, -1, :].astype(mx.float32)
    processed = logits_processor([], mx.array(np.array(raw_last)))
    if _DEBUG:
        dump_prefill_next_token(
            model.tokenizer,
            raw_last,
            processed,
            last_token_id=int(np.asarray(input_ids)[0, -1]),
            k=max(k, 15),
        )
    probs = np.asarray(mx.softmax(processed.astype(mx.float32), axis=-1)[0])
    top_ids = np.argsort(-probs)[:k]

    rows = []
    for rank, tid in enumerate(top_ids, start=1):
        tid = int(tid)
        _rewind_kv_cache(cache, prefill_offset)
        tokens = _greedy_continue_from_first(
            vlm,
            cache,
            tid,
            logits_processor,
            int(eos_token_id),
            max_generation_length,
        )
        extra = extract_text_tokens(model.tokenizer, mx.array([tokens]))
        raw = extra["cot"][0] if extra and extra.get("cot") else model.tokenizer.decode(tokens)
        first = _decode_id(model.tokenizer, tid)
        print(
            f"[TOP{k}] rank={rank} p0={probs[tid]:.4f} first={first!r} "
            f"coc={raw!r}"
        )
        rows.append(
            {
                "rank": rank,
                "first_token": first,
                "first_token_id": tid,
                "first_p": float(probs[tid]),
                "raw": raw,
                "tokens": tokens,
            }
        )
    return rows


def sample_trajectories_from_data_with_vlm_rollout(
    model: AlpamayoR1MLX,
    data: Dict[str, Any],
    num_traj_samples: int = 1,
    num_traj_sets: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.98,
    vlm_only: bool = False,
    return_extra: bool = False,
    **kwargs,
) -> Tuple[Any, Any, Any]:
    """Clean VLM rollout using the Alpamayo subclasses.

    The heavy post-processing that was previously required for rope_deltas
    and position_ids has been removed. The fixes now live inside
    AlpamayoLanguageModel.get_rope_index and AlpamayoModel.get_input_embeddings.
    """
    n_samples_total = num_traj_samples * num_traj_sets

    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    tokenized_data = data["tokenized_data"]

    input_ids = tokenized_data["input_ids"]
    if isinstance(input_ids, list):
        input_ids = mx.array(input_ids)

    image_kwargs = {}
    for k, v in tokenized_data.items():
        if k in ("pixel_values", "pixel_values_videos"):
            arr = np.asarray(v)
            # Processor flats are HF order C*T*H*W (1536 = 3*2*16*16).
            # Do not reshape as (N, T, H, W, C) — that scrambles the pack.
            # AlpamayoPatchEmbed views 2D as (N, 3, 2, 16, 16).
            if arr.ndim == 5 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (0, 4, 1, 2, 3))
            image_kwargs[k] = mx.array(arr)
        elif k in ("image_grid_thw", "video_grid_thw"):
            image_kwargs[k] = mx.array(v, dtype=mx.int32)

    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = _fuse_and_report(model, input_ids, traj_data_vlm)

    eos_token_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if eos_token_id is None:
        eos_token_id = model.tokenizer.eos_token_id

    logits_processor = ExpertLogitsProcessor(
        traj_token_offset=model.traj_token_start_idx,
        traj_vocab_size=model.traj_vocab_size,
        traj_token_ids=getattr(model, "traj_token_id_list", None),
    )

    stopping_criteria = StopAfterEOS(eos_token_id=eos_token_id)
    max_new_tokens = kwargs.get("max_generation_length") or model.tokens_per_future_traj

    vlm_profiler = StepProfiler(
        enabled=is_profiling_enabled(),
        name="VLM-Gen"
    )

    n_vlm_samples = n_samples_total

    def _run_single_vlm_generation(alpamayo_model, input_ids, image_kwargs, logits_processor,
                                    stopping_criteria, max_new_tokens, temperature, top_p):
        """Single-trajectory manual generation (now relies on fixed subclasses)."""
        generated_tokens = []
        vlm = alpamayo_model.vlm

        # Create KV cache list once before the first forward pass
        n_layers = len(vlm.language_model.model.layers)
        cache = [KVCache() for _ in range(n_layers)]

        # --- Prefill (memory peaks captured by MemoryMonitor) ---
        with MemoryMonitor(poll_interval=0.05, label="vlm_prefill"):
            outputs = vlm(
                input_ids=input_ids,
                **image_kwargs,
                cache=cache,
            )
        mx.eval(outputs.logits)
        record_memory_sample("after_vlm_prefill")

        raw_last = outputs.logits[:, -1, :].astype(mx.float32)
        mx.eval(raw_last)
        processed_last = logits_processor([], mx.array(np.array(raw_last)))
        last_tid = int(np.asarray(input_ids)[0, -1]) if np.asarray(input_ids).ndim == 2 else int(np.asarray(input_ids)[-1])
        if _DEBUG:
            dump_prefill_next_token(
                model.tokenizer,
                raw_last,
                processed_last,
                last_token_id=last_tid,
            )

        decode_profiler = StepProfiler(enabled=is_profiling_enabled(), name="Decode")
        for step in range(max_new_tokens):
            decode_profiler.step_start(step)

            logits = outputs.logits[:, -1, :]
            logits = logits_processor(generated_tokens, logits)
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
            if _DEBUG:
                probs = np.asarray(mx.softmax(logits.astype(mx.float32), axis=-1)[0])
                topk_idx = np.argsort(-probs)[:3]
                topk_bits = []
                for tid in topk_idx:
                    try:
                        tok = model.tokenizer.decode([int(tid)])
                    except Exception:
                        tok = f"<{int(tid)}>"
                    topk_bits.append(f"{tok}({probs[int(tid)]:.3f})")
                sampled_id = int(next_token.item())
                try:
                    sampled_tok = model.tokenizer.decode([sampled_id])
                except Exception:
                    sampled_tok = f"<{sampled_id}>"
                print(
                    f"[STEP {step+1}] sampled={sampled_tok!r} id={sampled_id}  "
                    f"top-3: " + ", ".join(topk_bits)
                )

            if step == 0:
                mx.eval(next_token)
                record_memory_sample(f"after_first_decode_eval")

            generated_tokens.append(int(next_token.item()))

            if stopping_criteria(mx.array([generated_tokens])):
                decode_profiler.step_end()
                break

            outputs = vlm(
                input_ids=next_token[None, :],
                cache=cache,
            )

            mx.eval(outputs.logits)
            decode_profiler.step_end()

        decode_profiler.summary()

        generated = mx.array([generated_tokens])
        record_memory_sample("after_vlm_generation_complete")
        return generated, cache

    if n_vlm_samples <= 1:
        generated, cache = _run_single_vlm_generation(
            model, input_ids, image_kwargs, logits_processor,
            stopping_criteria, max_new_tokens, temperature, top_p
        )
        rope_deltas = 0
    else:
        seq_list = []
        cache_list = []
        for _ in range(n_vlm_samples):
            gen, c = _run_single_vlm_generation(
                model, input_ids, image_kwargs, logits_processor,
                stopping_criteria, max_new_tokens, temperature, top_p
            )
            seq_list.append(gen)
            cache_list.append(c)
        generated = mx.concatenate(seq_list, axis=0)
        cache = cache_list[-1]
        rope_deltas = 0

    class VLMOutputs:
        def __init__(self, sequences, cache, rope_deltas):
            self.sequences = sequences
            self.cache = cache
            self.rope_deltas = rope_deltas

    vlm_outputs = VLMOutputs(sequences=generated, cache=cache, rope_deltas=rope_deltas)

    generated = replace_padding_after_eos(
        generated, eos_token_id=eos_token_id, pad_token_id=model.tokenizer.pad_token_id
    )

    if vlm_only:
        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        return None, None, extra

    if kwargs.get("return_extra", False):
        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        return None, None, extra

    return None, None, None


# ------------------------------------------------------------------
# Convenience wrapper (kept for backward compatibility with tests)
# ------------------------------------------------------------------

def run_vlm_generation(model, input_ids, image_kwargs, **gen_kwargs):
    """Thin wrapper around the clean rollout for unit tests."""
    return sample_trajectories_from_data_with_vlm_rollout(
        model,
        {"tokenized_data": {"input_ids": input_ids, **image_kwargs}},
        vlm_only=True,
        return_extra=True,
        **gen_kwargs,
    )