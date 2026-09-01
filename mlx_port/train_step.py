"""T4.3: SFT train graph — teacher-forced VLM CE + one CFM draw.

This is not inference. Do not call
``sample_trajectories_from_data_with_vlm_rollout``, do not decode token-by-token,
and do not run ``FlowMatching.sample`` (10 Euler steps). NVIDIA Stage 1 is
shifted CE on the backbone; Stage 2 is one VLM forward for KV plus one expert
forward on a single flow-matching timestep.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_port.models.expert_mlx import (
    cache_seq_len,
    expert_non_causal_train_mask,
    trim_cache,
)
from mlx_port.stage_timers import StageClock, bind_clock, compiled_flags, quantized_flags

IGNORE_INDEX = -100
TRAIN_STAGES = ("stage1", "stage2", "joint")
TRAIN_MS_KEYS = (
    "tokenize_ms",
    "encode_cache_ms",
    "encode_ms",
    "backbone_ms",
    "expert_ms",
    "loss_ms",
    "fwd_bwd_ms",
    "adam_ms",
)
TRAIN_DOMINANT = (
    "tokenize",
    "encode_cache",
    "encode",
    "backbone",
    "expert",
    "loss",
    "fwd_bwd",
    "adam",
    "python-overhead",
)


@dataclass
class TrainStepTimes:
    """Wall-clock for one train forward or one Adam step.

    Infer is encode / prefill / decode / FM. Train has no decode and no
    Euler loop. A materialized forward is encode + backbone (+ expert).
    An Adam step is ``value_and_grad`` (fwd+bwd) then ``opt.update``.
    ``encode_cache`` is ``freeze_vision_features`` when vision is off the tape.
    """

    tokenize_ms: float = 0.0
    encode_cache_ms: float = 0.0
    encode_ms: float = 0.0
    backbone_ms: float = 0.0
    expert_ms: float = 0.0
    loss_ms: float = 0.0
    fwd_bwd_ms: float = 0.0
    adam_ms: float = 0.0
    total_ms: float = 0.0
    n_vlm_forwards: int = 0
    n_expert_forwards: int = 0
    n_euler_steps: int = 0
    n_decode_tokens: int = 0

    def as_dict(self) -> dict[str, Any]:
        stages = {
            "tokenize": float(self.tokenize_ms),
            "encode_cache": float(self.encode_cache_ms),
            "encode": float(self.encode_ms),
            "backbone": float(self.backbone_ms),
            "expert": float(self.expert_ms),
            "loss": float(self.loss_ms),
            "fwd_bwd": float(self.fwd_bwd_ms),
            "adam": float(self.adam_ms),
        }
        accounted = sum(stages.values())
        total = float(self.total_ms) if self.total_ms > 0.0 else accounted
        overhead = max(0.0, total - accounted)
        stages["python-overhead"] = overhead
        dominant = max(stages, key=stages.get)
        if stages[dominant] <= 0.0:
            dominant = "python-overhead"
        return {
            "tokenize_ms": round(self.tokenize_ms, 1),
            "encode_cache_ms": round(self.encode_cache_ms, 1),
            "encode_ms": round(self.encode_ms, 1),
            "backbone_ms": round(self.backbone_ms, 1),
            "expert_ms": round(self.expert_ms, 1),
            "loss_ms": round(self.loss_ms, 1),
            "fwd_bwd_ms": round(self.fwd_bwd_ms, 1),
            "adam_ms": round(self.adam_ms, 1),
            "total_ms": round(total, 1),
            "dominant_stage": dominant,
            "n_vlm_forwards": int(self.n_vlm_forwards),
            "n_expert_forwards": int(self.n_expert_forwards),
            "n_euler_steps": int(self.n_euler_steps),
            "n_decode_tokens": int(self.n_decode_tokens),
            "compiled": compiled_flags(),
            "quantized": quantized_flags(),
            "dtype": "bfloat16",
        }


@dataclass
class TrainUpdateOutput:
    loss: float
    times: TrainStepTimes


def print_train_table(times: dict[str, Any]) -> None:
    print(
        "[TRAIN] "
        f"tokenize_ms={times['tokenize_ms']:.1f}  "
        f"encode_cache_ms={times['encode_cache_ms']:.1f}  "
        f"encode_ms={times['encode_ms']:.1f}  "
        f"backbone_ms={times['backbone_ms']:.1f}  "
        f"expert_ms={times['expert_ms']:.1f}  "
        f"loss_ms={times['loss_ms']:.1f}  "
        f"fwd_bwd_ms={times['fwd_bwd_ms']:.1f}  "
        f"adam_ms={times['adam_ms']:.1f}  "
        f"total_ms={times['total_ms']:.1f}  "
        f"dominant={times['dominant_stage']}"
    )


def mean_train_times(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean of train-stage ms. Recomputes dominant from the means."""
    if not trials:
        raise ValueError("mean_train_times requires at least one trial")
    out = dict(trials[0])
    keys = TRAIN_MS_KEYS + ("total_ms",)
    for key in keys:
        out[key] = round(float(sum(t[key] for t in trials) / len(trials)), 1)
    stages = {
        "tokenize": out["tokenize_ms"],
        "encode_cache": out["encode_cache_ms"],
        "encode": out["encode_ms"],
        "backbone": out["backbone_ms"],
        "expert": out["expert_ms"],
        "loss": out["loss_ms"],
        "fwd_bwd": out["fwd_bwd_ms"],
        "adam": out["adam_ms"],
    }
    dominant = max(stages, key=stages.get)
    if stages[dominant] <= 0.0:
        dominant = "python-overhead"
    out["dominant_stage"] = dominant
    return out


def run_value_and_grad_update(
    model: Any,
    loss_fn: Any,
    optimizer: Any,
) -> tuple[float, float, float]:
    """Adam step with a host barrier between VJP and ``opt.update``.

    MLX is lazy: without ``mx.eval(loss, grads)`` the backward is attributed
    to the later ``mx.eval(parameters)``. Same numerics as a single eval.
    """
    if model is None or loss_fn is None or optimizer is None:
        raise ValueError("run_value_and_grad_update requires model, loss_fn, optimizer")
    t0 = time.perf_counter()
    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, grads)
    fwd_bwd_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    adam_ms = (time.perf_counter() - t1) * 1000.0
    return float(loss.item()), fwd_bwd_ms, adam_ms


@dataclass
class TrainStepOutput:
    loss: mx.array
    vlm_ce: mx.array | None
    cfm_mse: mx.array | None
    times: TrainStepTimes
    logits: Any = None
    cache: Any = None
    ce_future: mx.array | None = None
    ce_others: mx.array | None = None
    n_ce: int = 0
    n_future: int = 0
    n_others: int = 0


def assert_train_graph(times: TrainStepTimes) -> None:
    """Raise if the step generated tokens or ran Euler integration."""
    if times is None:
        raise ValueError("assert_train_graph requires times")
    if times.n_euler_steps != 0:
        raise RuntimeError(
            f"train graph ran {times.n_euler_steps} Euler steps; "
            "SFT must use construct_training_data (one CFM draw)"
        )
    if times.n_decode_tokens != 0:
        raise RuntimeError(
            f"train graph decoded {times.n_decode_tokens} tokens; "
            "SFT must teacher-force, not generate"
        )
    if times.n_vlm_forwards < 1:
        raise RuntimeError("train graph did not run a VLM forward")


def shifted_cross_entropy(
    logits: mx.array,
    labels: mx.array,
    ignore_index: int = IGNORE_INDEX,
) -> mx.array:
    """Next-token CE. Matches NVIDIA ``_compute_next_token_loss`` (shift by 1)."""
    if logits is None or labels is None:
        raise ValueError("shifted_cross_entropy requires logits and labels")
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, L, V), got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must be (B, L), got {tuple(labels.shape)}")
    if int(logits.shape[0]) != int(labels.shape[0]) or int(logits.shape[1]) != int(
        labels.shape[1]
    ):
        raise ValueError(
            f"logits {tuple(logits.shape)} does not match labels {tuple(labels.shape)}"
        )
    shift_logits = logits[:, :-1, :].astype(mx.float32)
    shift_labels = labels[:, 1:]
    batch, length, vocab = shift_logits.shape
    flat_logits = shift_logits.reshape((batch * length, vocab))
    flat_labels = shift_labels.reshape((batch * length,))
    mask = flat_labels != int(ignore_index)
    n_valid = int(mx.sum(mask.astype(mx.int32)).item())
    if n_valid < 1:
        return mx.array(0.0, dtype=mx.float32)
    safe = mx.where(mask, flat_labels, mx.zeros_like(flat_labels))
    per = nn.losses.cross_entropy(flat_logits, safe, reduction="none")
    return (per * mask.astype(per.dtype)).sum() / mx.array(n_valid, dtype=per.dtype)


def apply_labels_mask(
    input_ids: mx.array,
    labels_mask: mx.array | None,
    ignore_index: int = IGNORE_INDEX,
) -> mx.array:
    labels = mx.array(input_ids)
    if labels_mask is None:
        return labels
    mask = mx.array(labels_mask).astype(mx.bool_)
    if mask.shape != labels.shape:
        raise ValueError(
            f"labels_mask {tuple(mask.shape)} does not match input_ids {tuple(labels.shape)}"
        )
    return mx.where(mask, labels, mx.full(labels.shape, int(ignore_index), dtype=labels.dtype))


def _as_int32(ids: Any) -> mx.array:
    arr = mx.array(ids)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"input_ids must be (B, L), got {tuple(arr.shape)}")
    return arr.astype(mx.int32)


def get_role_eos_mask(
    input_ids: Any,
    tokenizer: Any,
    bos_token: str = "<|im_start|>",
    eos_token: str = "<|im_end|>",
    role: str = "assistant",
) -> mx.array:
    """True only at assistant ``<|im_end|>``. NVIDIA ``get_role_eos_mask``."""
    if tokenizer is None:
        raise ValueError("get_role_eos_mask requires a tokenizer")
    ids = np.asarray(_as_int32(input_ids))
    bos_id = tokenizer.convert_tokens_to_ids(bos_token)
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    role_id = tokenizer.convert_tokens_to_ids(role)
    if bos_id is None or eos_id is None or role_id is None:
        raise ValueError(
            f"tokenizer missing {bos_token!r} / {eos_token!r} / {role!r}"
        )
    mask = np.zeros(ids.shape, dtype=bool)
    for b in range(ids.shape[0]):
        row = ids[b]
        bos_pos = np.flatnonzero(row == int(bos_id))
        eos_pos = np.flatnonzero(row == int(eos_id))
        n = min(bos_pos.size, eos_pos.size)
        for i in range(n):
            start = int(bos_pos[i])
            if start + 1 >= row.size:
                raise ValueError("BOS is the last token; no role id")
            if int(row[start + 1]) == int(role_id):
                mask[b, int(eos_pos[i])] = True
    return mx.array(mask)


def sft_stage1_labels_mask(input_ids: Any, tokenizer: Any) -> mx.array:
    """NVIDIA Stage 1: ``traj_future`` span plus assistant ``<|im_end|>``.

    ``vla_processor.yaml`` ``label_components: [traj_future]`` plus
    ``get_role_eos_mask``. CoC is not labeled.
    """
    if tokenizer is None:
        raise ValueError("sft_stage1_labels_mask requires a tokenizer")
    start_id = tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    end_id = tokenizer.convert_tokens_to_ids("<|traj_future_end|>")
    if start_id is None or end_id is None:
        raise RuntimeError("tokenizer has no traj_future_start/end")
    span = labels_mask_between(input_ids, int(start_id), int(end_id))
    eos = get_role_eos_mask(input_ids, tokenizer)
    return mx.array(np.asarray(span) | np.asarray(eos))


def future_traj_label_mask(labels: mx.array, model: Any) -> mx.array:
    """NVIDIA ``traj_mask``: discrete future IDs plus start/end specials."""
    start = getattr(model, "future_token_start_idx", None)
    vocab = getattr(model, "traj_vocab_size", None)
    ids = getattr(model, "traj_token_ids", None) or {}
    if start is None or vocab is None:
        raise ValueError("future_traj_label_mask requires future_token_start_idx and traj_vocab_size")
    if "future_start" not in ids or "future_end" not in ids:
        raise ValueError("traj_token_ids must include future_start and future_end")
    lab = mx.array(labels)
    in_bins = (lab >= int(start)) & (lab < int(start) + int(vocab))
    specials = (lab == int(ids["future_start"])) | (lab == int(ids["future_end"]))
    return in_bins | specials


def stage1_two_mean_ce(
    logits: mx.array,
    labels: mx.array,
    model: Any,
) -> tuple[mx.array, mx.array, mx.array, int, int]:
    """Sum of two means: future-traj bins and leftover labeled tokens (im_end)."""
    traj_mask = future_traj_label_mask(labels, model)
    labels_fut = mx.where(traj_mask, labels, mx.full(labels.shape, IGNORE_INDEX, dtype=labels.dtype))
    labels_oth = mx.where(traj_mask, mx.full(labels.shape, IGNORE_INDEX, dtype=labels.dtype), labels)
    ce_future = shifted_cross_entropy(logits, labels_fut)
    ce_others = shifted_cross_entropy(logits, labels_oth)
    n_future = int(mx.sum((labels_fut != IGNORE_INDEX).astype(mx.int32)).item())
    n_others = int(mx.sum((labels_oth != IGNORE_INDEX).astype(mx.int32)).item())
    return ce_future + ce_others, ce_future, ce_others, n_future, n_others


def labels_mask_between(
    input_ids: Any,
    start_id: int,
    end_id: int,
) -> mx.array:
    """Boolean loss mask on the unique ``start_id``…``end_id`` span (inclusive).

    Matches NVIDIA ``fill_masks_between_special_tokens`` for one pair.
    Raises if the sequence does not contain exactly one start and one end,
    or if the end precedes the start.
    """
    arr = np.asarray(_as_int32(input_ids))
    starts = np.flatnonzero(arr[0] == int(start_id))
    ends = np.flatnonzero(arr[0] == int(end_id))
    if starts.size != 1 or ends.size != 1:
        raise ValueError(
            f"need exactly one start id {int(start_id)} and one end id {int(end_id)}, "
            f"found {starts.size} / {ends.size}"
        )
    lo = int(starts[0])
    hi = int(ends[0])
    if hi < lo:
        raise ValueError(f"label span end {hi} precedes start {lo}")
    mask = np.zeros(arr.shape, dtype=bool)
    mask[0, lo : hi + 1] = True
    return mx.array(mask)


def append_traj_future_start(input_ids: Any, future_start_id: int) -> mx.array:
    """Append ``<|traj_future_start|>`` once if the sequence does not have it.

    Infer-style chat prompts stop before CoC, so they have no crop marker.
    A train-graph stage2/joint crop needs that id. Teacher CoC is attached in
    ``create_message(teacher_cot=...)``; this helper only appends the marker.
    """
    ids = _as_int32(input_ids)
    fid = int(future_start_id)
    hits = np.flatnonzero(np.asarray(ids)[0] == fid)
    if hits.size > 0:
        return ids
    marker = mx.array([[fid]], dtype=mx.int32)
    return mx.concatenate([ids, marker], axis=1)


def traj_future_keep_len(input_ids: mx.array, future_start_id: int) -> int:
    """Index after the last ``<|traj_future_start|>`` (NVIDIA crop length)."""
    arr = np.asarray(input_ids)
    if arr.ndim == 1:
        arr = arr[None, :]
    hits = np.flatnonzero(arr[0] == int(future_start_id))
    if hits.size < 1:
        raise ValueError(
            f"input_ids has no traj_future_start id {int(future_start_id)}"
        )
    return int(hits[-1]) + 1


def expert_train_position_ids(
    n_tokens: int,
    batch: int,
    rope_deltas: Any,
    prefix_len: int,
) -> mx.array:
    """NVIDIA SFT: ``arange(T) + rope_deltas + kv_len`` as ``(3, B, T)``."""
    if n_tokens < 1 or batch < 1:
        raise ValueError(f"expert_train_position_ids got n_tokens={n_tokens} batch={batch}")
    base = np.arange(int(n_tokens), dtype=np.int32)
    pos = np.broadcast_to(base, (3, int(batch), int(n_tokens))).copy()
    if rope_deltas is None:
        rd = 0
    else:
        rd = int(np.asarray(rope_deltas).reshape(-1)[0])
    pos = pos + rd + int(prefix_len)
    return mx.array(pos)


def _vlm_of(model: Any) -> Any:
    if model is None:
        raise ValueError("train step requires a model")
    if hasattr(model, "vlm") and model.vlm is not None:
        return model.vlm
    return model


def _rope_deltas_of(vlm: Any) -> Any:
    lm = getattr(vlm, "language_model", None)
    if lm is None:
        return None
    return getattr(lm, "_rope_deltas", None)


def _future_start_id(model: Any) -> int | None:
    ids = getattr(model, "traj_token_ids", None) or {}
    if "future_start" in ids:
        return int(ids["future_start"])
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return None
    tid = tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if tid is None:
        return None
    return int(tid)


def _stop_gradient_cache(cache: Any) -> None:
    """NVIDIA Stage 2: ``stop_grad_from_vlm`` on KV before the expert."""
    if cache is None:
        return
    for layer in cache:
        keys = getattr(layer, "keys", None)
        values = getattr(layer, "values", None)
        if keys is not None:
            layer.keys = mx.stop_gradient(keys)
        if values is not None:
            layer.values = mx.stop_gradient(values)


def freeze_vlm(model: Any) -> None:
    """Freeze the VLM for Stage 2, including any LoRA A/B inside it."""
    if model is None:
        raise ValueError("freeze_vlm requires a model")
    vlm = _vlm_of(model)
    if hasattr(vlm, "freeze"):
        vlm.freeze()
    else:
        raise RuntimeError("VLM has no freeze()")


def unfreeze_expert(model: Any) -> None:
    """Unfreeze the diffusion expert and action projections for Stage 2."""
    if model is None:
        raise ValueError("unfreeze_expert requires a model")
    for name in ("expert", "action_in_proj", "action_out_proj"):
        mod = getattr(model, name, None)
        if mod is None:
            raise RuntimeError(f"unfreeze_expert: model.{name} is missing")
        if not hasattr(mod, "unfreeze"):
            raise RuntimeError(f"unfreeze_expert: model.{name} has no unfreeze()")
        mod.unfreeze()


def assert_stage2_trainables(model: Any) -> None:
    """VLM frozen; expert LoRA or dense expert/action trainable; no packed weight."""
    from mlx.utils import tree_flatten

    flat = dict(tree_flatten(model.trainable_parameters()))
    vlm_keys = [k for k in flat if k.startswith("vlm.")]
    if vlm_keys:
        raise RuntimeError(
            "stage2 VLM must be frozen; "
            f"trainable VLM keys {vlm_keys[:8]}"
        )
    expert_keys = [
        k
        for k in flat
        if k.startswith("expert.")
        or k.startswith("action_in_proj.")
        or k.startswith("action_out_proj.")
    ]
    if not expert_keys:
        raise RuntimeError("stage2 has no trainable expert or action proj parameters")
    for name, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            packed = dict(tree_flatten(mod.trainable_parameters()))
            bad = [k for k in packed if k == "weight" or k.endswith(".weight")]
            if bad:
                raise RuntimeError(
                    f"cannot train packed QuantizedLinear.weight on {name}; "
                    "use dense expert or expert LoRA"
                )


def prepare_stage2_trainables(model: Any, *, train_action_proj: bool = False) -> None:
    """Freeze VLM. Dense FT unfreezes expert+action; LoRA unfreezes A/B, optionally action."""
    from mlx_port.lora import (
        assert_only_lora_trainable,
        freeze_expert_base_unfreeze_lora,
        has_expert_lora,
    )

    freeze_vlm(model)
    if has_expert_lora(model):
        freeze_expert_base_unfreeze_lora(
            model, train_action_proj=bool(train_action_proj)
        )
        if not train_action_proj:
            assert_only_lora_trainable(model)
    else:
        if train_action_proj:
            raise RuntimeError(
                "train_action_proj requires expert LoRA; use unfreeze_expert for full FT"
            )
        unfreeze_expert(model)
    assert_stage2_trainables(model)


def sft_expert_update(
    model: Any,
    batch: dict[str, Any],
    optimizer: Any,
    *,
    train_action_proj: bool = False,
) -> TrainUpdateOutput:
    """One Stage-2 CFM step. VLM (and VLM LoRA) stay frozen. Packed expert weights raise."""
    t_all = time.perf_counter()
    prepare_stage2_trainables(model, train_action_proj=train_action_proj)

    def loss_fn(m: Any) -> mx.array:
        return sft_train_step(m, batch, stage="stage2", materialize=False).loss

    loss, fwd_bwd_ms, adam_ms = run_value_and_grad_update(model, loss_fn, optimizer)
    return TrainUpdateOutput(
        loss=loss,
        times=TrainStepTimes(
            fwd_bwd_ms=fwd_bwd_ms,
            adam_ms=adam_ms,
            total_ms=(time.perf_counter() - t_all) * 1000.0,
            n_vlm_forwards=1,
            n_expert_forwards=1,
        ),
    )


def _crop_cache_to_future_start(cache: Any, input_ids: mx.array, future_start_id: int) -> None:
    if cache is None:
        return
    keep = traj_future_keep_len(input_ids, future_start_id)
    prefix = cache_seq_len(cache)
    if prefix < 1:
        return
    drop = int(prefix) - int(keep)
    if drop > 0:
        trim_cache(cache, drop)


def drop_n_traj_group(xyz: Any, rot: Any) -> tuple[mx.array, mx.array]:
    """NVIDIA ``ego_*[:, -1]`` → ``(B, T, …)`` when the loader adds n_traj."""
    xyz_m = mx.array(xyz)
    rot_m = mx.array(rot)
    if xyz_m.ndim == 4:
        xyz_m = xyz_m[:, -1]
    if rot_m.ndim == 5:
        rot_m = rot_m[:, -1]
    return xyz_m, rot_m


def _action_from_batch(model: Any, batch: dict[str, Any]) -> mx.array:
    if batch.get("action") is not None:
        action = mx.array(batch["action"])
        if action.ndim == 2:
            action = action[None, ...]
        return action
    needed = (
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
    )
    missing = [k for k in needed if batch.get(k) is None]
    if missing:
        raise ValueError(
            "stage2/joint need 'action' or ego history+future xyz/rot; "
            f"missing {missing}"
        )
    if model.action_space is None:
        raise ValueError("stage2/joint requires model.action_space")
    hist_xyz, hist_rot = drop_n_traj_group(
        batch["ego_history_xyz"], batch["ego_history_rot"]
    )
    fut_xyz, fut_rot = drop_n_traj_group(
        batch["ego_future_xyz"], batch["ego_future_rot"]
    )
    action = model.action_space.traj_to_action(
        hist_xyz,
        hist_rot,
        fut_xyz,
        fut_rot,
    )
    action = mx.array(action)
    dims = model.action_space.get_action_space_dims()
    if action.shape[-len(dims) :] != tuple(dims):
        raise RuntimeError(
            f"traj_to_action shape {tuple(action.shape)} does not end with {dims}"
        )
    return action.reshape((-1, *dims))


def _vlm_kwargs(batch: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mask",
        "cached_image_features",
        "cached_deepstack_visual_embeds",
    ):
        if batch.get(key) is not None:
            val = batch[key]
            if key in ("image_grid_thw", "video_grid_thw"):
                kwargs[key] = mx.array(val, dtype=mx.int32)
            elif key == "cached_deepstack_visual_embeds":
                kwargs[key] = val
            elif key == "mask":
                kwargs[key] = mx.array(val)
            elif key == "cached_image_features":
                kwargs[key] = val if isinstance(val, mx.array) else mx.array(val)
            else:
                kwargs[key] = mx.array(val)
    return kwargs


def _new_cache(vlm: Any) -> list[Any] | None:
    lm = getattr(vlm, "language_model", None)
    layers = None
    if lm is not None and hasattr(lm, "model") and hasattr(lm.model, "layers"):
        layers = lm.model.layers
    elif hasattr(vlm, "layers"):
        layers = vlm.layers
    if not layers:
        return None
    from mlx_lm.models.cache import KVCache

    return [KVCache() for _ in range(len(layers))]


def teacher_forced_vlm(
    model: Any,
    input_ids: mx.array,
    *,
    cache: list[Any] | None = None,
    materialize: bool = True,
    **vlm_kwargs: Any,
) -> tuple[Any, TrainStepTimes]:
    """One VLM forward over the full teacher sequence. No decode loop.

    ``materialize=False`` skips ``mx.eval`` so the call can sit inside
    ``nn.value_and_grad`` (T4.1). Timing is only valid when materialized.
    """
    vlm = _vlm_of(model)
    clock = StageClock()
    t0 = time.perf_counter()
    if materialize:
        with bind_clock(clock):
            outputs = vlm(input_ids=input_ids, cache=cache, **vlm_kwargs)
            logits = getattr(outputs, "logits", outputs)
            mx.eval(logits)
    else:
        # Do not bind the stage clock: AlpamayoModel evals logits when a clock
        # is set, which must not sit inside nn.value_and_grad.
        outputs = vlm(input_ids=input_ids, cache=cache, **vlm_kwargs)
        logits = getattr(outputs, "logits", outputs)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    if clock.decode_ms > 0.0 or clock.decode_tok > 0:
        raise RuntimeError(
            "teacher_forced_vlm decoded tokens; pass the full sequence in one call"
        )
    if clock.fm_steps > 0:
        raise RuntimeError("teacher_forced_vlm ran Euler; that is the infer graph")
    times = TrainStepTimes(
        encode_ms=clock.encode_ms,
        backbone_ms=clock.prefill_ms if clock.prefill_ms > 0.0 else wall_ms - clock.encode_ms,
        total_ms=wall_ms,
        n_vlm_forwards=1,
        n_decode_tokens=0,
        n_euler_steps=0,
    )
    return outputs, times


def cfm_expert_forward(
    model: Any,
    action: mx.array,
    *,
    cache: list[Any] | None = None,
    input_ids: mx.array | None = None,
    future_start_id: int | None = None,
    rope_deltas: Any = None,
    materialize: bool = True,
) -> tuple[mx.array, mx.array, TrainStepTimes]:
    """One CFM draw + one expert forward. Does not call ``diffusion.sample``."""
    if model.expert is None or model.action_in_proj is None or model.action_out_proj is None:
        raise ValueError("cfm_expert_forward requires expert, action_in_proj, action_out_proj")
    if model.diffusion is None:
        raise ValueError("cfm_expert_forward requires diffusion")

    t0 = time.perf_counter()
    training = model.diffusion.construct_training_data(action)
    embeds = model.action_in_proj(training["noisy_x"], training["timesteps"])
    if materialize:
        mx.eval(embeds)
    if cache is not None and input_ids is not None and future_start_id is not None:
        _crop_cache_to_future_start(cache, input_ids, future_start_id)
    prefix = cache_seq_len(cache) if cache else 0
    batch, n_tok, _ = embeds.shape
    position_ids = expert_train_position_ids(n_tok, batch, rope_deltas, prefix)
    expert_kwargs: dict[str, Any] = {
        "inputs_embeds": embeds,
        "position_ids": position_ids,
        "cache": cache,
    }
    if getattr(model, "expert_non_causal_attention", True):
        expert_kwargs["mask"] = expert_non_causal_train_mask(batch, n_tok, prefix)
    hidden = model.expert(**expert_kwargs)
    pred = model.action_out_proj(hidden)
    pred = pred.reshape(action.shape)
    loss = model.diffusion.compute_loss_from_pred(training, pred)
    if materialize:
        mx.eval(loss)
    expert_ms = (time.perf_counter() - t0) * 1000.0
    times = TrainStepTimes(
        expert_ms=expert_ms,
        total_ms=expert_ms,
        n_expert_forwards=1,
        n_euler_steps=0,
        n_decode_tokens=0,
    )
    return loss, pred, times


def sft_train_step(
    model: Any,
    batch: dict[str, Any],
    stage: str = "stage1",
    *,
    materialize: bool = True,
) -> TrainStepOutput:
    """Official SFT graphs. ``stage`` is ``stage1``, ``stage2``, or ``joint``.

    ``joint`` is NVIDIA ``cotrain_vlm``: CE + CFM on the same teacher sequence.
    """
    if stage not in TRAIN_STAGES:
        raise ValueError(f"stage must be one of {TRAIN_STAGES}, got {stage!r}")
    if batch is None:
        raise ValueError("sft_train_step requires a batch")
    if "input_ids" not in batch:
        raise ValueError("batch requires input_ids")

    t_all = time.perf_counter()
    input_ids = _as_int32(batch["input_ids"])
    if batch.get("fuse") and hasattr(model, "fuse_traj_tokens"):
        input_ids = model.fuse_traj_tokens(
            input_ids,
            {
                "ego_history_xyz": batch.get("ego_history_xyz"),
                "ego_history_rot": batch.get("ego_history_rot"),
                "ego_future_xyz": batch.get("ego_future_xyz"),
                "ego_future_rot": batch.get("ego_future_rot"),
            },
        )

    labels = batch.get("labels")
    if labels is None:
        labels = apply_labels_mask(input_ids, batch.get("labels_mask"))
    else:
        labels = mx.array(labels)
        if labels.ndim == 1:
            labels = labels[None, :]

    cache = _new_cache(_vlm_of(model))
    vlm_out, vlm_times = teacher_forced_vlm(
        model,
        input_ids,
        cache=cache,
        materialize=materialize,
        **_vlm_kwargs(batch),
    )
    logits = getattr(vlm_out, "logits", vlm_out)

    vlm_ce = None
    cfm_mse = None
    ce_future = None
    ce_others = None
    n_future = 0
    n_others = 0
    expert_times = TrainStepTimes()
    t_ce = time.perf_counter()
    if stage in ("stage1", "joint"):
        can_split = (
            getattr(model, "future_token_start_idx", None) is not None
            and getattr(model, "traj_vocab_size", None) is not None
            and getattr(model, "traj_token_ids", None)
            and "future_start" in (model.traj_token_ids or {})
            and "future_end" in (model.traj_token_ids or {})
        )
        if can_split and batch.get("stage1_two_mean", True):
            vlm_ce, ce_future, ce_others, n_future, n_others = stage1_two_mean_ce(
                logits, labels, model
            )
        else:
            vlm_ce = shifted_cross_entropy(logits, labels)
        if materialize and vlm_ce is not None:
            mx.eval(vlm_ce)
    loss_ms = (time.perf_counter() - t_ce) * 1000.0

    if stage in ("stage2", "joint"):
        if stage == "stage2":
            _stop_gradient_cache(cache)
        action = _action_from_batch(model, batch)
        future_id = _future_start_id(model)
        cfm_mse, _, expert_times = cfm_expert_forward(
            model,
            action,
            cache=cache,
            input_ids=input_ids if future_id is not None else None,
            future_start_id=future_id,
            rope_deltas=_rope_deltas_of(_vlm_of(model)),
            materialize=materialize,
        )

    if stage == "stage1":
        if vlm_ce is None:
            raise RuntimeError("stage1 produced no CE")
        loss = vlm_ce
    elif stage == "stage2":
        if cfm_mse is None:
            raise RuntimeError("stage2 produced no CFM loss")
        loss = cfm_mse
    else:
        if vlm_ce is None or cfm_mse is None:
            raise RuntimeError("joint produced incomplete losses")
        loss = vlm_ce + cfm_mse

    if materialize:
        mx.eval(loss)
    times = TrainStepTimes(
        encode_ms=vlm_times.encode_ms,
        backbone_ms=vlm_times.backbone_ms,
        expert_ms=expert_times.expert_ms,
        loss_ms=loss_ms,
        total_ms=(time.perf_counter() - t_all) * 1000.0,
        n_vlm_forwards=vlm_times.n_vlm_forwards,
        n_expert_forwards=expert_times.n_expert_forwards,
        n_euler_steps=0,
        n_decode_tokens=0,
    )
    if stage == "stage1" and times.n_expert_forwards != 0:
        raise RuntimeError("stage1 must not call the expert")
    if stage in ("stage2", "joint") and times.n_expert_forwards != 1:
        raise RuntimeError(f"{stage} must call the expert once, got {times.n_expert_forwards}")
    assert_train_graph(times)
    n_ce = int(mx.sum((labels != IGNORE_INDEX).astype(mx.int32)).item()) if labels is not None else 0
    return TrainStepOutput(
        loss=loss,
        vlm_ce=vlm_ce,
        cfm_mse=cfm_mse,
        times=times,
        logits=logits,
        cache=cache,
        ce_future=ce_future,
        ce_others=ce_others,
        n_ce=n_ce,
        n_future=n_future,
        n_others=n_others,
    )
