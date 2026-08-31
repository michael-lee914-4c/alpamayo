"""Small-scale NVIDIA Stage-1 SFT: language QLoRA, vision frozen, no CoC clips.

50/50 train/eval on local PAI chunks (default 8 clips → 4/4). Default keyframe
t0=5.1 s (not a CoC event). Ten Adam steps on LoRA A/B only. Eval is
forward-only after every step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np

from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    LOCAL_PAI_COC,
    list_local_traj_clips,
    load_coc_table,
    split_train_eval,
)
from mlx_port.lora import (
    DEFAULT_SAVE_EVERY,
    freeze_vision_features,
    has_vision_lora,
    inject_backbone_lora,
    lora_save_steps,
    packed_weight_fingerprint,
    save_lora_adapters,
    sft_lora_update,
)
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.profiling import MemoryMonitor, get_global_memory_peak
from mlx_port.scripts.time_train_step import (
    CHECKPOINT,
    DEFAULT_STAGE1_T0_US,
    _load_model,
    build_pai_train_batch,
)
from mlx_port.train_step import sft_train_step

REPORT_DEFAULT = Path("/Users/michaellee/Projects/alpamayo/reports/sft_stage1_small_10step.json")
LORA_SAVE_DEFAULT = Path("/Users/michaellee/Projects/alpamayo/reports/qlora/sft_stage1_small")


def select_non_coc_clips(
    local_dir: Path,
    n_clips: int,
    seed: int,
    chunk_max: int = 249,
) -> tuple[list[str], list[str]]:
    if n_clips < 2:
        raise ValueError(f"n_clips must be >= 2 for a 50/50 split, got {n_clips}")
    table = list_local_traj_clips(local_dir, chunk_max=chunk_max, exclude_coc=True)
    coc = load_coc_table(local_dir / "reasoning" / "ood_reasoning.parquet")
    coc_ids = set(coc.index.astype(str))
    ids = [str(i) for i in table.index]
    overlap = [c for c in ids if c in coc_ids]
    if overlap:
        raise RuntimeError(f"CoC clips leaked into traj pool: {overlap[:4]}")
    if DEFAULT_EVAL_CLIP_ID in ids:
        raise RuntimeError(
            f"default CoC eval clip {DEFAULT_EVAL_CLIP_ID} is in the traj pool"
        )
    if len(ids) < n_clips:
        raise RuntimeError(
            f"only {len(ids)} non-CoC local clips, need {n_clips}"
        )
    rng = np.random.default_rng(int(seed))
    pick = [ids[int(i)] for i in rng.choice(len(ids), size=n_clips, replace=False)]
    train, eval_ids = split_train_eval(pick, seed=seed, train_frac=0.5)
    leaked = [c for c in train + eval_ids if c in coc_ids]
    if leaked:
        raise RuntimeError(f"split still contains CoC clips: {leaked}")
    return train, eval_ids


def _eval_loss(model: AlpamayoR1MLX, batch: dict) -> float:
    if has_vision_lora(model):
        raise RuntimeError("this smoke freezes the vision tower; do not inject vision LoRA")
    batch = freeze_vision_features(model, batch)
    out = sft_train_step(model, batch, stage="stage1", materialize=True)
    return float(out.loss.item())


def _mean_eval(model: AlpamayoR1MLX, batches: list[dict]) -> float:
    if not batches:
        raise ValueError("eval set is empty")
    losses = [_eval_loss(model, b) for b in batches]
    return float(sum(losses) / len(losses))


def _prepare_batches(
    model: AlpamayoR1MLX,
    clip_ids: list[str],
    local_dir: Path,
) -> list[dict]:
    batches = []
    for clip_id in clip_ids:
        batch, meta = build_pai_train_batch(
            model,
            clip_id,
            local_dir,
            recipe="stage1",
            t0_us=DEFAULT_STAGE1_T0_US,
        )
        if meta.get("teacher_cot") is not None:
            raise RuntimeError(f"clip {clip_id} unexpectedly has a CoC teacher string")
        batch = freeze_vision_features(model, batch)
        batches.append(batch)
        print(
            f"[DATA] clip={clip_id} t0_us={meta['t0_us']} seq={meta['seq']} "
            f"n_ce={meta['n_ce']} n_future={meta['n_future_pads']}"
        )
    return batches


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
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-lr", type=float, default=1e-5)
    parser.add_argument(
        "--quantize-all",
        action="store_true",
        default=True,
        help="all4 packed VLM+expert (default on for this smoke).",
    )
    parser.add_argument(
        "--no-quantize-all",
        action="store_false",
        dest="quantize_all",
        help="Dense bf16 instead of all4.",
    )
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    parser.add_argument(
        "--lora-save-dir",
        type=Path,
        default=LORA_SAVE_DEFAULT,
        help="Directory for adapters.safetensors (overwritten every save).",
    )
    parser.add_argument(
        "--lora-save-every",
        type=int,
        default=DEFAULT_SAVE_EVERY,
        help="Overwrite adapters.safetensors every N completed steps (default 10). The last step is always saved.",
    )
    parser.add_argument(
        "--no-lora-save",
        action="store_true",
        help="Do not write LoRA weights.",
    )
    args = parser.parse_args()
    if args.n_clips < 2:
        parser.error("--n-clips must be >= 2")
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.no_lora_save and args.lora_save_every != DEFAULT_SAVE_EVERY:
        parser.error("--no-lora-save and --lora-save-every are exclusive")
    if not args.no_lora_save and args.lora_save_every < 1:
        parser.error("--lora-save-every must be >= 1")

    train_ids, eval_ids = select_non_coc_clips(
        args.local_dir, args.n_clips, args.seed
    )
    print(
        f"[SPLIT] n={args.n_clips} train={len(train_ids)} eval={len(eval_ids)} "
        f"seed={args.seed} exclude_coc=1 t0_us={DEFAULT_STAGE1_T0_US}"
    )
    print(f"[SPLIT] train={train_ids}")
    print(f"[SPLIT] eval={eval_ids}")

    ns = argparse.Namespace(
        alpamayo_path=args.alpamayo_path,
        quantize_lm=False,
        quantize_all=args.quantize_all,
    )
    model, path, flags = _load_model(ns)
    lora_info = inject_backbone_lora(
        model,
        rank=args.lora_rank,
        scale=args.lora_scale,
        vision=False,
    )
    if has_vision_lora(model):
        raise RuntimeError("vision tower must stay frozen (no vision LoRA)")
    packed_fp0 = None
    if args.quantize_all:
        packed_fp0 = packed_weight_fingerprint(model)
        print(f"[LORA] packed_fp={packed_fp0}")

    t_prep = time.perf_counter()
    train_batches = _prepare_batches(model, train_ids, args.local_dir)
    eval_batches = _prepare_batches(model, eval_ids, args.local_dir)
    prep_s = time.perf_counter() - t_prep

    opt = optim.Adam(learning_rate=args.lora_lr)
    rows: list[dict] = []
    saved: list[str] = []
    save_at = (
        set()
        if args.no_lora_save
        else set(lora_save_steps(args.steps, args.lora_save_every))
    )
    with MemoryMonitor(poll_interval=0.05, label="sft_stage1_small"):
        t0_eval = time.perf_counter()
        eval0 = _mean_eval(model, eval_batches)
        eval0_ms = (time.perf_counter() - t0_eval) * 1000.0
        print(f"[EVAL] step=-1 mean={eval0:.4f} n={len(eval_batches)} ms={eval0_ms:.1f}")
        rows.append(
            {
                "step": -1,
                "train_clip": None,
                "train_loss": None,
                "eval_mean": eval0,
                "train_ms": None,
                "eval_ms": eval0_ms,
            }
        )
        t_all = time.perf_counter()
        for i in range(int(args.steps)):
            batch = train_batches[i % len(train_batches)]
            clip_id = train_ids[i % len(train_ids)]
            t0 = time.perf_counter()
            loss = sft_lora_update(model, batch, opt, stage="stage1")
            train_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            ev = _mean_eval(model, eval_batches)
            eval_ms = (time.perf_counter() - t1) * 1000.0
            rows.append(
                {
                    "step": i,
                    "train_clip": clip_id,
                    "train_loss": loss,
                    "eval_mean": ev,
                    "train_ms": train_ms,
                    "eval_ms": eval_ms,
                }
            )
            print(
                f"[STEP] {i} clip={clip_id} train={loss:.4f} "
                f"eval={ev:.4f} train_ms={train_ms:.1f} eval_ms={eval_ms:.1f}"
            )
            completed = i + 1
            if completed in save_at:
                info = save_lora_adapters(
                    model,
                    args.lora_save_dir,
                    step=completed,
                    rank=int(lora_info["rank"]),
                    scale=float(lora_info["scale"]),
                    vision_scope=str(lora_info["vision_scope"]),
                    extra={
                        "recipe": "stage1",
                        "n_clips": int(args.n_clips),
                        "seed": int(args.seed),
                    },
                )
                saved.append(info["path"])
        wall_s = time.perf_counter() - t_all

    if packed_fp0 is not None:
        packed_fp1 = packed_weight_fingerprint(model)
        if packed_fp1 != packed_fp0:
            raise RuntimeError("packed QuantizedLinear.weight changed during Stage-1 QLoRA")

    train_losses = [r["train_loss"] for r in rows if r["train_loss"] is not None]
    eval_losses = [r["eval_mean"] for r in rows]
    if any(not np.isfinite(x) for x in train_losses + eval_losses):
        raise RuntimeError(f"non-finite loss: train={train_losses} eval={eval_losses}")
    peak = get_global_memory_peak()
    report = {
        "recipe": "stage1",
        "exclude_coc": True,
        "t0_us": DEFAULT_STAGE1_T0_US,
        "lora_vision": "none",
        "n_clips": args.n_clips,
        "n_train": len(train_ids),
        "n_eval": len(eval_ids),
        "steps": args.steps,
        "seed": args.seed,
        "lr": args.lora_lr,
        "rank": args.lora_rank,
        "path": path,
        "flags": flags,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "prep_s": prep_s,
        "wall_s": wall_s,
        "loss0": train_losses[0],
        "lossN": train_losses[-1],
        "eval0": eval_losses[0],
        "evalN": eval_losses[-1],
        "rows": rows,
        "metal_gb": peak["metal"] / 1e9,
        "rss_gb": peak["resident"] / 1e9,
        "lora_save_dir": None if args.no_lora_save else str(args.lora_save_dir),
        "lora_save_every": None if args.no_lora_save else int(args.lora_save_every),
        "lora_snapshots": saved,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(
        f"[DONE] loss0={train_losses[0]:.4f} lossN={train_losses[-1]:.4f} "
        f"eval0={eval_losses[0]:.4f} evalN={eval_losses[-1]:.4f} "
        f"prep_s={prep_s:.1f} wall_s={wall_s:.1f} "
        f"metal={peak['metal']/1e9:.2f}GB rss={peak['resident']/1e9:.2f}GB "
        f"report={args.report} "
        f"lora_save={args.lora_save_dir if not args.no_lora_save else 'off'}"
    )


if __name__ == "__main__":
    main()
