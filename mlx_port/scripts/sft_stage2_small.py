"""Small-scale NVIDIA Stage-2 SFT: freeze LoRA VLM, Adam on expert CFM.

Loads the Stage-1 language QLoRA adapters, freezes the VLM (including A/B),
and trains the diffusion expert with one flow-matching draw per step.
Same 8-clip 50/50 split as sft_stage1_small (seed 0).

Default: all4 VLM + dense bf16 expert (full FT of expert + action proj).
``--expert-lora``: pack the all4 expert and QLoRA the 36 decoder layers
(q/k/v/o/gate/up/down). Action in/out stay dense and frozen unless
``--train-action-proj`` also Adam-steps them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from mlx_port.gt_eval import LOCAL_PAI_COC
from mlx_port.lora import (
    DEFAULT_SAVE_EVERY,
    freeze_vision_features,
    has_vision_lora,
    inject_backbone_lora,
    inject_expert_lora,
    load_lora_adapters,
    lora_save_steps,
    packed_weight_fingerprint,
    save_dense_trainables,
    save_lora_adapters,
)
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.profiling import MemoryMonitor, get_global_memory_peak
from mlx_port.scripts.sft_stage1_small import (
    LORA_SAVE_DEFAULT,
    resolve_train_steps,
    select_non_coc_clips,
)
from mlx_port.scripts.time_train_step import (
    CHECKPOINT,
    DEFAULT_STAGE1_T0_US,
    _load_model,
    build_pai_train_batch,
)
from mlx_port.paths import REPORTS_DIR
from mlx_port.train_step import (
    mean_train_times,
    prepare_stage2_trainables,
    print_train_table,
    sft_expert_update,
    sft_train_step,
)

REPORT_DEFAULT = REPORTS_DIR / "sft_stage2_small_8clip.json"
REPORT_EXPERT_LORA_DEFAULT = REPORTS_DIR / "sft_stage2_small_8clip_expert_qlora.json"
REPORT_EXPERT_DENSE_DEFAULT = (
    REPORTS_DIR / "sft_stage2_small_8clip_expert_qlora_dense.json"
)
EXPERT_LORA_SAVE_DEFAULT = REPORTS_DIR / "qlora" / "sft_stage2_small"
EVAL_RNG_SEED = 12345


def _prepare_batches(
    model: AlpamayoR1MLX,
    clip_ids: list[str],
    local_dir: Path,
) -> tuple[list[dict], list[dict]]:
    if has_vision_lora(model):
        raise RuntimeError("stage2 smoke freezes the VLM; do not inject vision LoRA")
    batches = []
    prep_rows: list[dict] = []
    for clip_id in clip_ids:
        t0 = time.perf_counter()
        batch, meta = build_pai_train_batch(
            model,
            clip_id,
            local_dir,
            recipe="stage1",
            t0_us=DEFAULT_STAGE1_T0_US,
        )
        tokenize_ms = (time.perf_counter() - t0) * 1000.0
        if meta.get("teacher_cot") is not None:
            raise RuntimeError(f"clip {clip_id} unexpectedly has a CoC teacher string")
        if batch.get("action") is None:
            raise RuntimeError(f"clip {clip_id} has no action for CFM")
        t1 = time.perf_counter()
        batch = freeze_vision_features(model, batch)
        encode_cache_ms = (time.perf_counter() - t1) * 1000.0
        action = np.asarray(batch["action"])
        batches.append(batch)
        prep_rows.append(
            {
                "clip_id": clip_id,
                "tokenize_ms": tokenize_ms,
                "encode_cache_ms": encode_cache_ms,
                "seq": meta["seq"],
            }
        )
        print(
            f"[DATA] clip={clip_id} t0_us={meta['t0_us']} seq={meta['seq']} "
            f"action={tuple(action.shape)} "
            f"amin={action.min():.3f} amax={action.max():.3f} "
            f"amean={action.mean():.3f} "
            f"tokenize_ms={tokenize_ms:.1f} encode_cache_ms={encode_cache_ms:.1f}"
        )
    return batches, prep_rows


def _eval_loss(model: AlpamayoR1MLX, batch: dict) -> tuple[float, dict]:
    out = sft_train_step(model, batch, stage="stage2", materialize=True)
    if out.cfm_mse is None:
        raise RuntimeError("stage2 eval produced no CFM loss")
    return float(out.loss.item()), out.times.as_dict()


def _mean_eval(model: AlpamayoR1MLX, batches: list[dict]) -> tuple[float, dict]:
    if not batches:
        raise ValueError("eval set is empty")
    mx.random.seed(EVAL_RNG_SEED)
    losses: list[float] = []
    times: list[dict] = []
    for b in batches:
        loss, t = _eval_loss(model, b)
        losses.append(loss)
        times.append(t)
    return float(sum(losses) / len(losses)), mean_train_times(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpamayo-path", type=Path, default=CHECKPOINT)
    parser.add_argument("--local-dir", type=Path, default=LOCAL_PAI_COC)
    parser.add_argument(
        "--n-clips",
        type=int,
        default=8,
        help="Even pool, split 50/50. Default 8 (4 train / 4 eval).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Adam steps. Default 10 if --epochs is unset. Exclusive with --epochs.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Full passes over the train split (steps = epochs * n_train). Exclusive with --steps.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lora-adapter-dir",
        type=Path,
        default=LORA_SAVE_DEFAULT,
        help="Stage-1 adapters.safetensors directory (required).",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument(
        "--expert-lr",
        type=float,
        default=1e-4,
        help="Adam LR on expert (dense FT or expert LoRA). NVIDIA sft_base Stage-2 default.",
    )
    parser.add_argument(
        "--expert-lora",
        action="store_true",
        help=(
            "QLoRA the diffusion expert decoder (36×7 q/k/v/o/gate/up/down). "
            "Packs the all4 expert. Action in/out stay dense and frozen "
            "unless --train-action-proj. Default is full dense FT of the expert."
        ),
    )
    parser.add_argument(
        "--train-action-proj",
        action="store_true",
        help=(
            "With --expert-lora, also Adam-step leftover dense action in/out. "
            "Packed QuantizedLinear stays frozen. Requires --expert-lora."
        ),
    )
    parser.add_argument("--expert-lora-rank", type=int, default=8)
    parser.add_argument("--expert-lora-scale", type=float, default=20.0)
    parser.add_argument(
        "--expert-lora-save-dir",
        type=Path,
        default=EXPERT_LORA_SAVE_DEFAULT,
        help="Directory for expert adapters.safetensors (requires --expert-lora).",
    )
    parser.add_argument(
        "--expert-lora-save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Overwrite expert adapters every N completed steps (default 10).",
    )
    parser.add_argument(
        "--no-expert-lora-save",
        action="store_true",
        help="Do not write expert LoRA weights.",
    )
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        default=True,
        help="all4 packed VLM (default on). Expert stays dense unless --expert-lora.",
    )
    parser.add_argument(
        "--no-quantize-all",
        action="store_false",
        dest="quantize_all",
        help="Dense bf16 VLM+expert instead of all4 VLM.",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--expert-dense", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.expert_dense:
        parser.error("--expert-dense was renamed to --train-action-proj")
    if args.n_clips < 2:
        parser.error("--n-clips must be >= 2")
    if args.epochs is not None and args.steps is not None:
        parser.error("--epochs and --steps are exclusive")
    if (
        int(args.expert_lora_rank) != 8 or float(args.expert_lora_scale) != 20.0
    ) and not args.expert_lora:
        parser.error("--expert-lora-rank/--expert-lora-scale require --expert-lora")
    if args.expert_lora_save_dir != EXPERT_LORA_SAVE_DEFAULT and not args.expert_lora:
        parser.error("--expert-lora-save-dir requires --expert-lora")
    if args.no_expert_lora_save and not args.expert_lora:
        parser.error("--no-expert-lora-save requires --expert-lora")
    if args.no_expert_lora_save and args.expert_lora_save_every != DEFAULT_SAVE_EVERY:
        parser.error("--no-expert-lora-save and --expert-lora-save-every are exclusive")
    if args.expert_lora and not args.no_expert_lora_save and args.expert_lora_save_every < 1:
        parser.error("--expert-lora-save-every must be >= 1")
    if args.train_action_proj and not args.expert_lora:
        parser.error("--train-action-proj requires --expert-lora")
    if args.report is None:
        if args.expert_lora and args.train_action_proj:
            args.report = REPORT_EXPERT_DENSE_DEFAULT
        elif args.expert_lora:
            args.report = REPORT_EXPERT_LORA_DEFAULT
        else:
            args.report = REPORT_DEFAULT
    adapter_w = args.lora_adapter_dir / "adapters.safetensors"
    adapter_c = args.lora_adapter_dir / "adapter_config.json"
    if not adapter_w.is_file() or not adapter_c.is_file():
        parser.error(
            f"Stage-1 adapters missing under {args.lora_adapter_dir} "
            "(need adapters.safetensors + adapter_config.json)"
        )

    train_ids, eval_ids = select_non_coc_clips(
        args.local_dir, args.n_clips, args.seed
    )
    try:
        args.steps, args.epochs = resolve_train_steps(
            steps=args.steps, epochs=args.epochs, n_train=len(train_ids)
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"[SPLIT] n={args.n_clips} train={len(train_ids)} eval={len(eval_ids)} "
        f"seed={args.seed} exclude_coc=1 t0_us={DEFAULT_STAGE1_T0_US} "
        f"steps={args.steps}"
        + (f" epochs={args.epochs}" if args.epochs is not None else "")
    )
    print(f"[SPLIT] train={train_ids}")
    print(f"[SPLIT] eval={eval_ids}")

    pack_expert = bool(args.expert_lora and args.quantize_all)
    ns = argparse.Namespace(
        alpamayo_path=args.alpamayo_path,
        quantize_lm=False,
        quantize_all=args.quantize_all,
        quantize_expert=pack_expert,
    )
    model, path, flags = _load_model(ns)
    expert_flag = str(flags.get("expert") or "")
    if pack_expert and not expert_flag.startswith("affine-4"):
        raise RuntimeError(
            f"--expert-lora with --quantize-all needs a packed expert; flags={flags}"
        )
    if not pack_expert and expert_flag.startswith("affine-4"):
        raise RuntimeError(
            f"dense Stage-2 FT needs a dense expert; flags={flags}. "
            "Use --expert-lora to QLoRA a packed expert, or keep quantize_expert=False."
        )
    lora_info = inject_backbone_lora(
        model,
        rank=args.lora_rank,
        scale=args.lora_scale,
        vision=False,
    )
    adapter_cfg = load_lora_adapters(model, args.lora_adapter_dir)
    if int(adapter_cfg.get("rank", -1)) != int(args.lora_rank):
        raise RuntimeError(
            f"adapter rank {adapter_cfg.get('rank')} != --lora-rank {args.lora_rank}"
        )
    if str(adapter_cfg.get("vision_scope", "")) != "none":
        raise RuntimeError(
            f"stage2 smoke expects language-only adapters, got "
            f"vision_scope={adapter_cfg.get('vision_scope')!r}"
        )
    expert_lora_info = None
    if args.expert_lora:
        expert_lora_info = inject_expert_lora(
            model,
            rank=args.expert_lora_rank,
            scale=args.expert_lora_scale,
        )
    prepare_stage2_trainables(model, train_action_proj=bool(args.train_action_proj))
    packed_fp0 = None
    if args.quantize_all:
        packed_fp0 = packed_weight_fingerprint(model)
        print(f"[STAGE2] packed_fp={packed_fp0} path={path}")
    extra = ""
    if expert_lora_info is not None:
        extra = (
            f" expert_lora_leaves={expert_lora_info['n_wrapped']} "
            f"rank={args.expert_lora_rank} scale={args.expert_lora_scale}"
        )
    print(
        f"[STAGE2] adapters={args.lora_adapter_dir} "
        f"step={adapter_cfg.get('step')} "
        f"vlm_lora_leaves={lora_info['n_decoder_wrapped']} "
        f"expert_lr={args.expert_lr} "
        f"expert_mode={'qlora+action' if args.expert_lora and args.train_action_proj else 'qlora' if args.expert_lora else 'dense_ft'}"
        f"{extra}"
    )

    t_prep = time.perf_counter()
    train_batches, train_prep = _prepare_batches(model, train_ids, args.local_dir)
    eval_batches, eval_prep = _prepare_batches(model, eval_ids, args.local_dir)
    prep_s = time.perf_counter() - t_prep

    opt = optim.Adam(learning_rate=args.expert_lr)
    save_at = (
        set()
        if (not args.expert_lora or args.no_expert_lora_save)
        else set(lora_save_steps(int(args.steps), int(args.expert_lora_save_every)))
    )
    rows: list[dict] = []
    with MemoryMonitor(poll_interval=0.05, label="sft_stage2_small"):
        t0_eval = time.perf_counter()
        eval0, eval0_times = _mean_eval(model, eval_batches)
        eval0_ms = (time.perf_counter() - t0_eval) * 1000.0
        print(f"[EVAL] step=-1 mean={eval0:.4f} n={len(eval_batches)} ms={eval0_ms:.1f}")
        print_train_table(eval0_times)
        rows.append(
            {
                "step": -1,
                "train_clip": None,
                "train_loss": None,
                "eval_mean": eval0,
                "train_ms": None,
                "eval_ms": eval0_ms,
                "eval_times": eval0_times,
            }
        )
        t_all = time.perf_counter()
        for i in range(int(args.steps)):
            batch = train_batches[i % len(train_batches)]
            clip_id = train_ids[i % len(train_ids)]
            mx.random.seed(int(args.seed) * 1000 + i)
            t0 = time.perf_counter()
            upd = sft_expert_update(
                model, batch, opt, train_action_proj=bool(args.train_action_proj)
            )
            train_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            ev, ev_times = _mean_eval(model, eval_batches)
            eval_ms = (time.perf_counter() - t1) * 1000.0
            train_times = upd.times.as_dict()
            rows.append(
                {
                    "step": i,
                    "train_clip": clip_id,
                    "train_loss": upd.loss,
                    "eval_mean": ev,
                    "train_ms": train_ms,
                    "eval_ms": eval_ms,
                    "train_times": train_times,
                    "eval_times": ev_times,
                }
            )
            print(
                f"[STEP] {i} clip={clip_id} train={upd.loss:.4f} "
                f"eval={ev:.4f} train_ms={train_ms:.1f} eval_ms={eval_ms:.1f}"
            )
            print_train_table(train_times)
            print_train_table(ev_times)
            completed = i + 1
            if completed in save_at and expert_lora_info is not None:
                save_lora_adapters(
                    model,
                    args.expert_lora_save_dir,
                    step=completed,
                    rank=int(expert_lora_info["rank"]),
                    scale=float(expert_lora_info["scale"]),
                    vision_scope="none",
                    extra={
                        "target": "expert",
                        "recipe": "stage2",
                        "train_action_proj": bool(args.train_action_proj),
                    },
                    allow_extra_trainables=bool(args.train_action_proj),
                )
                if args.train_action_proj:
                    save_dense_trainables(
                        model, args.expert_lora_save_dir, step=completed
                    )
        wall_s = time.perf_counter() - t_all

    if packed_fp0 is not None:
        packed_fp1 = packed_weight_fingerprint(model)
        if packed_fp1 != packed_fp0:
            raise RuntimeError(
                "packed QuantizedLinear.weight changed during Stage-2 "
                "(VLM language/vision and/or expert decoder)"
            )

    train_losses = [r["train_loss"] for r in rows if r["train_loss"] is not None]
    eval_losses = [r["eval_mean"] for r in rows]
    if any(not np.isfinite(x) for x in train_losses + eval_losses):
        raise RuntimeError(f"non-finite loss: train={train_losses} eval={eval_losses}")
    peak = get_global_memory_peak()
    report = {
        "recipe": "stage2",
        "exclude_coc": True,
        "t0_us": DEFAULT_STAGE1_T0_US,
        "lora_vision": "none",
        "lora_adapter_dir": str(args.lora_adapter_dir),
        "lora_adapter_step": adapter_cfg.get("step"),
        "expert_lora": bool(args.expert_lora),
        "train_action_proj": bool(args.train_action_proj),
        "expert_lora_rank": int(args.expert_lora_rank) if args.expert_lora else None,
        "expert_lora_scale": float(args.expert_lora_scale) if args.expert_lora else None,
        "n_expert_wrapped": (
            None if expert_lora_info is None else int(expert_lora_info["n_wrapped"])
        ),
        "expert_lora_save_dir": (
            str(args.expert_lora_save_dir) if args.expert_lora and not args.no_expert_lora_save else None
        ),
        "n_clips": args.n_clips,
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "epochs": args.epochs,
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.expert_lr,
        "path": path,
        "flags": flags,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "prep_s": prep_s,
        "prep_train": train_prep,
        "prep_eval": eval_prep,
        "wall_s": wall_s,
        "loss0": train_losses[0],
        "lossN": train_losses[-1],
        "eval0": eval_losses[0],
        "evalN": eval_losses[-1],
        "rows": rows,
        "metal_gb": peak["metal"] / 1e9,
        "rss_gb": peak["resident"] / 1e9,
        "packed_fp": packed_fp0,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(
        f"[DONE] loss0={train_losses[0]:.4f} lossN={train_losses[-1]:.4f} "
        f"eval0={eval_losses[0]:.4f} evalN={eval_losses[-1]:.4f} "
        f"prep_s={prep_s:.1f} wall_s={wall_s:.1f} "
        f"metal={peak['metal']/1e9:.2f}GB rss={peak['resident']/1e9:.2f}GB "
        f"report={args.report}"
    )


if __name__ == "__main__":
    main()
