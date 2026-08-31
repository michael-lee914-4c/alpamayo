"""Time one SFT step on the real checkpoint (no generate, no Euler).

Default is a short text-only teacher sequence. ``--from-clip`` is NVIDIA
public SFT Stage 1: 16 frames, fused history, 128 discrete traj-future
tokens, assistant ``<|im_end|>``. ``--teacher-cot`` is paper 5.2 CoC CE.
``--stage stage2`` freezes the VLM and runs one CFM draw.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from mlx_port.gt_eval import DEFAULT_EVAL_CLIP_ID, load_clip_gt
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.models.quantize_lm import resolve_quant_mode
from mlx_port.processor import (
    DEFAULT_FUTURE_TRAJ_TOKENS,
    DEFAULT_NUM_FRAMES,
    alpamayo_apply_chat_template,
    create_message,
    get_processor,
)
from mlx_port.profiling import MemoryMonitor, get_global_memory_peak
from mlx_port.scripts.run_local_coc_sample import LOCAL_DIR
from mlx_port.stage_timers import quantized_flags
from mlx_port.lora import (
    DEFAULT_SAVE_EVERY,
    inject_backbone_lora,
    lora_save_steps,
    packed_weight_fingerprint,
    save_lora_adapters,
    sft_lora_update,
)
from mlx_port.train_step import (
    append_traj_future_start,
    drop_n_traj_group,
    freeze_vlm,
    labels_mask_between,
    sft_expert_update,
    sft_stage1_labels_mask,
    sft_train_step,
)
from mlx_port.traj_sample_plot_utils import quant_path_label

CHECKPOINT = Path("/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B")
LORA_SAVE_DEFAULT = Path("/Users/michaellee/Projects/alpamayo/reports/qlora/time_train_step")


def _dummy_ids(model: AlpamayoR1MLX, seq_len: int) -> mx.array:
    future_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if future_id is None:
        raise RuntimeError("tokenizer has no <|traj_future_start|>")
    if seq_len < 4:
        raise ValueError(f"seq_len must be >= 4, got {seq_len}")
    ids = np.arange(seq_len, dtype=np.int32) % 100 + 10
    ids[-3] = int(future_id)
    return mx.array(ids[None, :])


def _future_start_id(model: AlpamayoR1MLX) -> int:
    tid = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if tid is None:
        raise RuntimeError("tokenizer has no <|traj_future_start|>")
    return int(tid)


def _image_batch_from_tokenized(inputs: dict) -> dict:
    """Same pixel / grid packing as infer (HF C*T*H*W flats, 16×[1,H,W])."""
    out: dict = {}
    for key in ("pixel_values", "pixel_values_videos"):
        if key not in inputs:
            continue
        arr = np.asarray(inputs[key])
        if arr.ndim == 5 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (0, 4, 1, 2, 3))
        out[key] = arr
    for key in ("image_grid_thw", "video_grid_thw"):
        if key in inputs:
            out[key] = np.asarray(inputs[key])
    return out


def _event_coc(gt: dict, t0_us: int) -> str:
    matches = [
        e
        for e in gt["events"]
        if int(e["event_start_timestamp"]) == int(t0_us) and e.get("coc")
    ]
    if not matches:
        raise RuntimeError(f"no CoC label on event t0_us={t0_us}")
    text = str(matches[0]["coc"]).strip()
    if not text:
        raise RuntimeError(f"empty CoC label on event t0_us={t0_us}")
    return text


def _prepare_teacher_inputs(
    processor,
    frames,
    teacher_cot: str | None = None,
    sft_stage: str | None = None,
    num_future_traj_tokens: int = DEFAULT_FUTURE_TRAJ_TOKENS,
) -> dict:
    """Tokenize an SFT teacher turn. Not the infer prefix."""
    messages = create_message(
        frames.flatten(0, 1),
        teacher_cot=teacher_cot,
        sft_stage=sft_stage,
        num_future_traj_tokens=num_future_traj_tokens,
    )
    inputs = alpamayo_apply_chat_template(
        processor,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=False,
        return_dict=True,
        return_tensors="np",
    )
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            arr = inputs[key]
            if hasattr(arr, "shape") and len(arr.shape) == 5 and arr.shape[-1] == 3:
                inputs[key] = np.transpose(arr, (0, 4, 1, 2, 3))
    return inputs


def _token_id(tokenizer, name: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(name)
    if tid is None:
        raise RuntimeError(f"tokenizer has no {name}")
    return int(tid)


def _gt_action(model: AlpamayoR1MLX, data: dict) -> mx.array:
    if model.action_space is None:
        raise RuntimeError("PAI clip batch requires model.action_space")
    needed = (
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
    )
    missing = [k for k in needed if data.get(k) is None]
    if missing:
        raise RuntimeError(f"clip is missing {missing}")
    hist_xyz, hist_rot = drop_n_traj_group(
        data["ego_history_xyz"], data["ego_history_rot"]
    )
    fut_xyz, fut_rot = drop_n_traj_group(
        data["ego_future_xyz"], data["ego_future_rot"]
    )
    action = model.action_space.traj_to_action(
        hist_xyz, hist_rot, fut_xyz, fut_rot
    )
    action = mx.array(action)
    dims = model.action_space.get_action_space_dims()
    if action.shape[-len(dims) :] != tuple(dims):
        raise RuntimeError(
            f"traj_to_action shape {tuple(action.shape)} does not end with {dims}"
        )
    return action.reshape((-1, *dims))


# NVIDIA PAIDataset.DEFAULT_T0_US when use_default_keyframe=true.
DEFAULT_STAGE1_T0_US = 5_100_000


def build_pai_train_batch(
    model: AlpamayoR1MLX,
    clip_id: str,
    local_dir: str | Path,
    *,
    recipe: str = "stage1",
    t0_us: int | None = None,
) -> tuple[dict, dict]:
    """Tokenize a PAI clip for SFT.

    ``recipe="stage1"`` is NVIDIA ``vla_processor.yaml``: discrete traj-future
    pads + assistant ``<|im_end|>``. ``recipe="coc"`` is paper 5.2 teacher CoC.

    ``t0_us`` is the sample time. If omitted, ``recipe="coc"`` (and the
    ``--from-clip`` smoke) uses the first CoC event. Pass
    ``DEFAULT_STAGE1_T0_US`` for Stage-1 traj clips that are not in the CoC set.
    """
    if recipe not in ("stage1", "coc"):
        raise ValueError(f"recipe must be 'stage1' or 'coc', got {recipe!r}")
    teacher_cot = None
    if recipe == "coc":
        if t0_us is not None:
            raise ValueError("recipe='coc' picks t0 from the CoC event; do not pass t0_us")
        gt = load_clip_gt(clip_id)
        if not gt.get("events"):
            raise RuntimeError(f"clip {clip_id} has no CoC events")
        t0_us = int(gt["events"][0]["event_start_timestamp"])
        teacher_cot = _event_coc(gt, t0_us)
    elif t0_us is None:
        gt = load_clip_gt(clip_id)
        if not gt.get("events"):
            raise RuntimeError(f"clip {clip_id} has no CoC events")
        t0_us = int(gt["events"][0]["event_start_timestamp"])
    else:
        t0_us = int(t0_us)
    data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
        local_dir=str(local_dir),
        maybe_stream=True,
        num_frames=DEFAULT_NUM_FRAMES,
    )
    needed = (
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
    )
    missing = [k for k in needed if data.get(k) is None]
    if missing:
        raise RuntimeError(f"clip is missing {missing}")
    processor = get_processor(model.tokenizer)
    n_fut = int(getattr(model, "tokens_per_future_traj", DEFAULT_FUTURE_TRAJ_TOKENS))
    if recipe == "stage1":
        inputs = _prepare_teacher_inputs(
            processor,
            data["image_frames"],
            sft_stage="stage1",
            num_future_traj_tokens=n_fut,
        )
    else:
        inputs = _prepare_teacher_inputs(
            processor, data["image_frames"], teacher_cot=teacher_cot
        )
    raw_ids = mx.array(np.asarray(inputs["input_ids"]), dtype=mx.int32)
    if raw_ids.ndim == 1:
        raw_ids = raw_ids[None, :]
    traj_data = {
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
    }
    if recipe == "stage1":
        fused = model.fuse_traj_tokens(raw_ids, traj_data)
        ids = fused
        labels_mask = sft_stage1_labels_mask(ids, model.tokenizer)
        n_ce = int(np.asarray(labels_mask).sum())
        if n_ce < n_fut + 3:
            raise RuntimeError(
                f"Stage-1 label span is too short: n_ce={n_ce} "
                f"(need start+{n_fut}+end+im_end)"
            )
    else:
        fused = model.fuse_traj_tokens(
            raw_ids,
            {
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
        )
        ids = append_traj_future_start(fused, _future_start_id(model))
        labels_mask = labels_mask_between(
            ids,
            _token_id(model.tokenizer, "<|cot_start|>"),
            _token_id(model.tokenizer, "<|cot_end|>"),
        )
        n_ce = int(np.asarray(labels_mask).sum())
        if n_ce < 2:
            raise RuntimeError(f"CoC label span is too short: n_ce={n_ce}")
    action = _gt_action(model, data)
    image = _image_batch_from_tokenized(inputs)
    if "pixel_values" not in image and "pixel_values_videos" not in image:
        raise RuntimeError("PAI clip tokenize produced no pixel_values")
    if "image_grid_thw" not in image:
        raise RuntimeError("PAI clip tokenize produced no image_grid_thw")
    pixel_key = "pixel_values" if "pixel_values" in image else "pixel_values_videos"
    n_future_pads = int(
        (np.asarray(raw_ids) == _token_id(model.tokenizer, "<|traj_future|>")).sum()
    )
    batch = {
        "input_ids": ids,
        "labels_mask": labels_mask,
        "action": action,
        "fuse": False,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
        "ego_future_xyz": data["ego_future_xyz"],
        "ego_future_rot": data["ego_future_rot"],
        **image,
    }
    meta = {
        "clip_id": clip_id,
        "t0_us": t0_us,
        "recipe": recipe,
        "teacher_cot": teacher_cot,
        "n_ce": n_ce,
        "n_future_pads": n_future_pads,
        "seq_raw": int(raw_ids.shape[-1]),
        "seq_fused": int(fused.shape[-1]),
        "seq": int(ids.shape[-1]),
        "n_images": int(np.asarray(image["image_grid_thw"]).shape[0]),
        "image_grid_thw": np.asarray(image["image_grid_thw"]).tolist(),
        "pixel_key": pixel_key,
        "pixel_shape": tuple(np.asarray(image[pixel_key]).shape),
        "action_shape": tuple(action.shape),
    }
    return batch, meta


def _load_model(args: argparse.Namespace) -> tuple[AlpamayoR1MLX, str, dict]:
    quant_mode = resolve_quant_mode(
        quantize_lm=args.quantize_lm, quantize_all=args.quantize_all
    )
    model = AlpamayoR1MLX.from_pretrained(
        str(args.alpamayo_path),
        load_expert=True,
        quantize_lm=quant_mode == "lm4",
        quantize_all=quant_mode == "all4",
    )
    flags = quantized_flags()
    path = quant_path_label(flags)
    if quant_mode == "all4" and (
        not str(flags.get("vision") or "").startswith("affine-4")
        or not str(flags.get("expert") or "").startswith("affine-4")
    ):
        raise RuntimeError(
            f"requested all4 but quantized flags are {flags}; "
            "mlx_all4 did not install on VLM+expert"
        )
    if quant_mode == "lm4" and not str(flags.get("lm") or "").startswith("affine-4"):
        raise RuntimeError(f"requested lm4 but quantized flags are {flags}")
    return model, path, flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpamayo-path", type=Path, default=CHECKPOINT)
    parser.add_argument("--stage", choices=("stage1", "stage2", "joint"), default="stage1")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument(
        "--from-clip",
        action="store_true",
        help="Greedy e2e PAI clip (16 frames). Ignores --seq-len.",
    )
    parser.add_argument(
        "--teacher-cot",
        action="store_true",
        help="Paper 5.2 CoC teacher string instead of NVIDIA Stage-1 traj-future pads.",
    )
    parser.add_argument(
        "--expert-update",
        action="store_true",
        help="Stage-2 Adam step on expert + action proj. VLM frozen. Packed expert raises.",
    )
    parser.add_argument("--clip-id", default=DEFAULT_EVAL_CLIP_ID)
    parser.add_argument("--local-dir", type=Path, default=LOCAL_DIR)
    parser.add_argument(
        "--quantize-lm",
        action="store_true",
        help="T3.1 affine-4 language tower. Default is dense bf16.",
    )
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        help="all4 affine-4 on the full VLM and diffusion expert.",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "T4.1: QLoRA on decoder q/k/v/o/gate/up/down and vision "
            "(see --lora-vision), then Adam on LoRA only. "
            "Vision LoRA keeps encode on the tape."
        ),
    )
    parser.add_argument(
        "--lora-vision",
        choices=("full", "merger", "none"),
        default="full",
        help=(
            "Vision QLoRA scope with --lora. full=27 blocks + merger + 3 deepstack; "
            "merger=merger + 3 deepstack only; none=language decoder only."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-steps", type=int, default=1)
    parser.add_argument(
        "--lora-lr",
        type=float,
        default=1e-5,
        help="Adam LR on LoRA A/B. Default 1e-5 (NVIDIA Stage-1). 1e-4 overshoots dummy CE.",
    )
    parser.add_argument(
        "--lora-save-dir",
        type=Path,
        default=LORA_SAVE_DEFAULT,
        help="Directory for adapters.safetensors (overwritten every save; requires --lora).",
    )
    parser.add_argument(
        "--lora-save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Overwrite adapters.safetensors every N completed steps (default 10). Last step is always saved.",
    )
    parser.add_argument(
        "--no-lora-save",
        action="store_true",
        help="Do not write LoRA weights.",
    )
    args = parser.parse_args()
    if args.quantize_lm and args.quantize_all:
        parser.error("--quantize-lm and --quantize-all are exclusive")
    if args.lora and args.stage != "stage1":
        parser.error("--lora is the T4.1 Stage-1 smoke; use --stage stage1")
    if args.lora_steps != 1 and not args.lora:
        parser.error("--lora-steps requires --lora")
    if args.lora and args.lora_steps < 1:
        parser.error("--lora-steps must be >= 1")
    if args.lora_vision != "full" and not args.lora:
        parser.error("--lora-vision requires --lora")
    if args.teacher_cot and not args.from_clip:
        parser.error("--teacher-cot requires --from-clip")
    if args.expert_update and args.stage != "stage2":
        parser.error("--expert-update is Stage-2 only; use --stage stage2")
    if args.expert_update and args.lora:
        parser.error("--expert-update and --lora are exclusive")
    if args.no_lora_save and args.lora_save_every != DEFAULT_SAVE_EVERY:
        parser.error("--no-lora-save and --lora-save-every are exclusive")
    if not args.lora and args.lora_save_dir != LORA_SAVE_DEFAULT:
        parser.error("--lora-save-dir requires --lora")
    if args.lora and not args.no_lora_save and args.lora_save_every < 1:
        parser.error("--lora-save-every must be >= 1")

    model, path, flags = _load_model(args)
    if args.from_clip:
        t_prep = time.perf_counter()
        recipe = "coc" if args.teacher_cot else "stage1"
        batch, meta = build_pai_train_batch(
            model, args.clip_id, args.local_dir, recipe=recipe
        )
        prep_ms = (time.perf_counter() - t_prep) * 1000.0
        seq_len = int(meta["seq"])
        print(
            f"[TRAIN] clip={meta['clip_id']} t0_us={meta['t0_us']} "
            f"recipe={meta['recipe']} "
            f"seq_raw={meta['seq_raw']} fused={meta['seq_fused']} "
            f"seq={meta['seq']} n_ce={meta['n_ce']} "
            f"n_future_pads={meta['n_future_pads']} "
            f"images={meta['n_images']} "
            f"grid0={meta['image_grid_thw'][0]} "
            f"{meta['pixel_key']}={meta['pixel_shape']} "
            f"action={meta['action_shape']} prep_ms={prep_ms:.1f}"
        )
        if meta.get("teacher_cot"):
            print(f"[TRAIN] teacher_cot={meta['teacher_cot']!r}")
        else:
            print("[TRAIN] NVIDIA Stage-1: discrete traj_future + assistant im_end")
    else:
        ids = _dummy_ids(model, args.seq_len)
        n_wp = int(model.action_space.get_action_space_dims()[0])
        batch = {"input_ids": ids, "action": mx.zeros((1, n_wp, 2), dtype=mx.float32)}
        seq_len = int(args.seq_len)

    packed_fp0 = None
    if args.lora:
        lora_info = inject_backbone_lora(
            model,
            rank=args.lora_rank,
            scale=args.lora_scale,
            vision=args.lora_vision != "none",
            vision_scope="full" if args.lora_vision == "none" else args.lora_vision,
        )
        if args.quantize_lm or args.quantize_all:
            packed_fp0 = packed_weight_fingerprint(model)
            print(f"[LORA] packed_fp={packed_fp0}")

    with MemoryMonitor(poll_interval=0.05, label="train_step"):
        if args.lora:
            opt = optim.Adam(learning_rate=args.lora_lr)
            losses: list[float] = []
            step_ms: list[float] = []
            t_all = time.perf_counter()
            save_at = (
                set()
                if args.no_lora_save
                else set(lora_save_steps(args.lora_steps, args.lora_save_every))
            )
            for i in range(int(args.lora_steps)):
                t0 = time.perf_counter()
                loss = sft_lora_update(model, batch, opt, stage=args.stage)
                dt = (time.perf_counter() - t0) * 1000.0
                losses.append(loss)
                step_ms.append(dt)
                print(f"[LORA] step={i} loss={loss:.4f} ms={dt:.1f}")
                completed = i + 1
                if completed in save_at:
                    save_lora_adapters(
                        model,
                        args.lora_save_dir,
                        step=completed,
                        rank=int(lora_info["rank"]),
                        scale=float(lora_info["scale"]),
                        vision_scope=str(lora_info["vision_scope"]),
                        extra={"recipe": args.stage},
                    )
            wall_s = time.perf_counter() - t_all
            if packed_fp0 is not None:
                packed_fp1 = packed_weight_fingerprint(model)
                if packed_fp1 != packed_fp0:
                    raise RuntimeError(
                        "packed QuantizedLinear.weight changed after LoRA steps"
                    )
            print(
                f"[LORA] stage={args.stage} seq={seq_len} path={path} "
                f"lm={flags.get('lm')} vision={flags.get('vision')} "
                f"expert={flags.get('expert')} "
                f"steps={args.lora_steps} rank={args.lora_rank} "
                f"lora_vision={args.lora_vision} "
                f"lr={args.lora_lr} "
                f"loss0={losses[0]:.4f} lossN={losses[-1]:.4f} "
                f"step0_ms={step_ms[0]:.1f} stepN_ms={step_ms[-1]:.1f} "
                f"wall_s={wall_s:.1f}"
            )
        elif args.expert_update:
            freeze_vlm(model)
            opt = optim.Adam(learning_rate=args.lora_lr)
            t0 = time.perf_counter()
            loss = sft_expert_update(model, batch, opt)
            dt = (time.perf_counter() - t0) * 1000.0
            print(
                f"[EXPERT] stage=stage2 seq={seq_len} path={path} "
                f"loss={loss:.4f} ms={dt:.1f} lr={args.lora_lr}"
            )
        else:
            if args.stage == "stage2":
                freeze_vlm(model)
            out = sft_train_step(model, batch, stage=args.stage)
            mx.eval(out.loss)
            t = out.times
            ce = None if out.vlm_ce is None else float(out.vlm_ce.item())
            mse = None if out.cfm_mse is None else float(out.cfm_mse.item())
            ce_f = None if out.ce_future is None else float(out.ce_future.item())
            ce_o = None if out.ce_others is None else float(out.ce_others.item())
            print(
                f"[TRAIN] stage={args.stage} seq={seq_len} path={path} "
                f"lm={flags.get('lm')} vision={flags.get('vision')} "
                f"expert={flags.get('expert')} "
                f"loss={float(out.loss.item()):.4f} "
                f"ce={ce if ce is None else f'{ce:.4f}'} "
                f"ce_future={ce_f if ce_f is None else f'{ce_f:.4f}'} "
                f"ce_others={ce_o if ce_o is None else f'{ce_o:.4f}'} "
                f"n_ce={out.n_ce} n_future={out.n_future} n_others={out.n_others} "
                f"cfm={mse if mse is None else f'{mse:.4f}'} "
                f"encode={t.encode_ms:.1f} backbone={t.backbone_ms:.1f} "
                f"expert={t.expert_ms:.1f} total={t.total_ms:.1f} "
                f"vlm={t.n_vlm_forwards} expert_fwd={t.n_expert_forwards} "
                f"euler={t.n_euler_steps} decode_tok={t.n_decode_tokens}"
            )
    peak = get_global_memory_peak()
    if peak["total"] > 0 or peak["metal"] > 0:
        print(
            f"[MEMORY] metal={peak['metal']/1e9:.2f}GB "
            f"rss={peak['resident']/1e9:.2f}GB "
            f"total={peak['total']/1e9:.2f}GB"
        )


if __name__ == "__main__":
    main()
