"""Temperature CoC + one expert trajectory on the same 5 local PAI-CoC clips.

Writes plots of pred vs GT egomotion so the XY can be inspected.
Does not download clips. Uses labels + cameras + egomotion already on disk.
"""

from __future__ import annotations

import argparse
import gc
import html
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from mlx_port.gt_eval import (
    _pred_xy_for_ade,
    clean_pred_coc,
    load_clip_gt,
    min_ade_xy,
    score_coc,
)
from mlx_port.inference import sample_trajectories_from_data_with_vlm_rollout
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.processor import DEFAULT_NUM_FRAMES, get_processor
from mlx_port.scripts.run_local_coc_sample import (
    CAM_NAMES,
    CHECKPOINT,
    LOCAL_DIR,
    TIME_LABELS,
    _prepare_inputs,
    _save_contact_sheet,
    _save_frames,
)
from mlx_port.traj_sample_plot_utils import (
    DT_S,
    _as_xy,
    _require_full_xy,
    _speed_mps,
    _xy_for_redraw as _xy_from_cached_record,
)

# Same five clips as reports/coc_sample_5_t06 (seed 42, skip t0 < 1.6s).
CLIP_IDS = [
    "faff17c7-4572-472a-be28-0f035bb88a37",
    "8e83f069-5fc5-4329-8589-4527de19d03e",
    "1946f1c3-8638-4648-8944-506e6bffc4df",
    "8e06f83c-4de6-4cde-82ab-4a747645f80a",
    "36cb5485-5c56-48c5-8219-90095851d627",
]
REPORT_DIR = Path("/Users/michaellee/Projects/alpamayo/reports/traj_sample_5_t06")
SEED = 42
NVIDIA_TEMPERATURE = 0.6
NVIDIA_TOP_P = 0.98


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _save_traj_plots(
    clip_dir: Path,
    hist_xy: np.ndarray,
    gt_xy: np.ndarray,
    pred_xy: np.ndarray,
    min_ade: float,
) -> dict[str, str]:
    """Bird's-eye XY (x forward, y left) plus speed vs time."""
    hist_xy = np.asarray(hist_xy, dtype=np.float64)
    gt_xy = _require_full_xy(gt_xy, "gt_xy")
    pred_xy = _require_full_xy(pred_xy, "pred_xy")

    all_xy = np.concatenate([hist_xy, gt_xy, pred_xy], axis=0)
    pad = 2.0
    ymin, ymax = float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad
    xmin, xmax = float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad
    span = max(ymax - ymin, xmax - xmin, 1.0)
    cy, cx = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span

    w, h, margin = 780, 780, 64
    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    font = _font(14)
    font_sm = _font(12)

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Vehicle frame: x forward, y left. Screen: +x up, +y to the left.
        px = margin + (ymax - y) / span * (w - 2 * margin)
        py = h - margin - (x - xmin) / span * (h - 2 * margin)
        return int(px), int(py)

    n_grid = 6
    for k in range(n_grid + 1):
        t = k / n_grid
        x = xmin + t * span
        y = ymin + t * span
        draw.line([to_px(x, ymin), to_px(x, ymax)], fill=(51, 65, 85), width=1)
        draw.line([to_px(xmin, y), to_px(xmax, y)], fill=(51, 65, 85), width=1)
    draw.line([to_px(xmin, 0), to_px(xmax, 0)], fill=(71, 85, 105), width=1)
    draw.line([to_px(0, ymin), to_px(0, ymax)], fill=(71, 85, 105), width=1)

    def poly(pts: np.ndarray, color: tuple[int, int, int], width: int, r: int) -> None:
        px = [to_px(p[0], p[1]) for p in pts]
        if len(px) >= 2:
            draw.line(px, fill=color, width=width)
        for p in px:
            draw.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=color)

    poly(hist_xy, (56, 189, 248), 3, 3)
    poly(gt_xy, (148, 163, 184), 3, 2)
    poly(pred_xy, (251, 146, 60), 3, 2)
    t0 = to_px(0.0, 0.0)
    draw.ellipse([t0[0] - 7, t0[1] - 7, t0[0] + 7, t0[1] + 7], fill=(244, 63, 94))
    draw.text((14, 12), "Ego-frame XY  (x+ forward ↑, y+ left ←)", fill=(226, 232, 240), font=font)
    draw.text((14, h // 2 - 8), "← y+", fill=(148, 163, 184), font=font_sm)
    draw.text((w // 2 - 10, 12), "x+ ↑", fill=(148, 163, 184), font=font_sm)
    draw.text(
        (14, h - 48),
        f"cyan=history  gray=GT future  amber=pred  red=t0    minADE={min_ade:.2f} m",
        fill=(148, 163, 184),
        font=font_sm,
    )
    xy_name = "traj_xy.png"
    img.save(clip_dir / xy_name)

    # Speed vs time
    sw, sh, sm = 780, 280, 52
    simg = Image.new("RGB", (sw, sh), (15, 23, 42))
    sdraw = ImageDraw.Draw(simg)
    gt_v = _speed_mps(gt_xy)
    pred_v = _speed_mps(pred_xy)
    tmax = DT_S * max(len(gt_v), len(pred_v))
    vmax = max(float(gt_v.max()), float(pred_v.max()), 1.0) * 1.15
    t_gt = np.arange(1, len(gt_v) + 1) * DT_S
    t_pr = np.arange(1, len(pred_v) + 1) * DT_S

    def s_px(t: float, v: float) -> tuple[int, int]:
        px = sm + t / tmax * (sw - 2 * sm)
        py = sh - sm - v / vmax * (sh - 2 * sm)
        return int(px), int(py)

    for k in range(5):
        v = vmax * k / 4
        y = s_px(0, v)[1]
        sdraw.line([(sm, y), (sw - sm, y)], fill=(51, 65, 85), width=1)
        sdraw.text((8, y - 7), f"{v:.1f}", fill=(100, 116, 139), font=font_sm)
    sdraw.line([s_px(t, v) for t, v in zip(t_gt, gt_v)], fill=(148, 163, 184), width=2)
    sdraw.line([s_px(t, v) for t, v in zip(t_pr, pred_v)], fill=(251, 146, 60), width=3)
    sdraw.text((14, 10), "Speed vs time (m/s, 10 Hz waypoints)", fill=(226, 232, 240), font=font)
    sdraw.text((14, sh - 28), "gray=GT  amber=pred", fill=(148, 163, 184), font=font_sm)
    speed_name = "traj_speed.png"
    simg.save(clip_dir / speed_name)

    return {
        "traj_xy": f"{clip_dir.name}/{xy_name}",
        "traj_speed": f"{clip_dir.name}/{speed_name}",
    }


def _xy_stats(xy: np.ndarray) -> dict:
    xy = np.asarray(xy, dtype=np.float64)
    return {
        "start": [round(float(xy[0, 0]), 4), round(float(xy[0, 1]), 4)],
        "end": [round(float(xy[-1, 0]), 4), round(float(xy[-1, 1]), 4)],
        "path_m": round(float(np.linalg.norm(xy[:, :2], axis=-1).max()), 4),
        "first5": np.round(xy[:5], 4).tolist(),
        "last5": np.round(xy[-5:], 4).tolist(),
    }


def run_one_clip(model, processor, clip_id: str, clip_dir: Path, seed: int) -> dict:
    gt = load_clip_gt(clip_id)
    t0_us = int(gt["events"][0]["event_start_timestamp"])
    data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
        local_dir=str(LOCAL_DIR),
        maybe_stream=True,
        num_frames=DEFAULT_NUM_FRAMES,
    )
    inputs = _prepare_inputs(processor, data["image_frames"])
    ids = np.asarray(inputs["input_ids"])
    grid = np.asarray(inputs.get("image_grid_thw"))
    print(f"[traj-sample] tokens={ids.size} image_grid_thw=\n{grid}")
    payload = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    mx.random.seed(seed)
    pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data=payload,
        num_traj_samples=1,
        num_traj_sets=1,
        temperature=NVIDIA_TEMPERATURE,
        top_p=NVIDIA_TOP_P,
        max_generation_length=256,
        return_extra=True,
        vlm_only=False,
    )
    pred_raw = extra["cot"][0] if extra and extra.get("cot") else None
    pred_coc = clean_pred_coc(pred_raw)
    score = score_coc(pred_coc, gt["gt_coc_texts"])

    frames = data["image_frames"]
    frames_np = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
    cam_idx = np.asarray(data["camera_indices"])
    hist_xy = _as_xy(data["ego_history_xyz"])
    gt_xy = _as_xy(data["ego_future_xyz"])
    pred_arr = None if pred_xyz is None else np.asarray(pred_xyz)
    pred_xy = None if pred_arr is None else _pred_xy_for_ade(pred_arr)[0]
    if pred_xy is not None:
        pred_xy = _require_full_xy(pred_xy, "pred_xy")
    gt_xy = _require_full_xy(gt_xy, "gt_xy")
    ade = None if pred_xy is None else min_ade_xy(pred_xy[None, ...], gt_xy)

    clip_dir.mkdir(parents=True, exist_ok=True)
    image_grid = _save_frames(clip_dir, frames_np, cam_idx)
    contact = _save_contact_sheet(clip_dir, frames_np, cam_idx)
    plots = (
        _save_traj_plots(clip_dir, hist_xy, gt_xy, pred_xy, ade)
        if pred_xy is not None
        else {}
    )

    rec = {
        "clip_id": clip_id,
        "chunk": gt["chunk"],
        "split": gt["split"],
        "event_cluster": gt["event_cluster"],
        "t0_us": t0_us,
        "n_events": len(gt["events"]),
        "gt_coc_texts": gt["gt_coc_texts"],
        "pred_coc_raw": pred_raw,
        "pred_coc": pred_coc,
        "readable": bool(score["readable"]),
        "jaccard": float(score["jaccard"]),
        "gt_coverage": float(score["gt_coverage"]),
        "min_ade_m": ade,
        "pred_xyz_shape": None if pred_arr is None else list(pred_arr.shape),
        "gt_xy": _xy_stats(gt_xy),
        "pred_xy": None if pred_xy is None else _xy_stats(pred_xy),
        "hist_xy_full": np.round(hist_xy, 4).tolist(),
        "gt_xy_full": np.round(gt_xy, 4).tolist(),
        "pred_xy_full": None if pred_xy is None else np.round(pred_xy, 4).tolist(),
        "pred_speed_full": None if pred_xy is None else np.round(_speed_mps(pred_xy), 4).tolist(),
        "gt_speed_full": np.round(_speed_mps(gt_xy), 4).tolist(),
        "gt_speed_start_end": [
            round(float(_speed_mps(gt_xy)[0]), 3),
            round(float(_speed_mps(gt_xy)[-1]), 3),
        ],
        "pred_speed_start_end": (
            None
            if pred_xy is None
            else [
                round(float(_speed_mps(pred_xy)[0]), 3),
                round(float(_speed_mps(pred_xy)[-1]), 3),
            ]
        ),
        "expert": (extra or {}).get("expert"),
        "image_grid": image_grid,
        "contact_sheet": contact,
        "traj_xy": plots.get("traj_xy"),
        "traj_speed": plots.get("traj_speed"),
        "cameras": [CAM_NAMES.get(int(i), str(int(i))) for i in cam_idx.tolist()],
        "seed": seed,
        "temperature": NVIDIA_TEMPERATURE,
        "top_p": NVIDIA_TOP_P,
    }
    del data, payload, pred_xyz, pred_rot, extra
    gc.collect()
    mx.clear_cache()
    return rec


def _xy_for_redraw(rec: dict, clip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Hist/GT/pred XY for a plot redraw. Pred is the saved 64-pt path only — no interpolation."""
    if rec.get("hist_xy_full") is not None and rec.get("gt_xy_full") is not None:
        return _xy_from_cached_record(rec, clip_id)
    gt_meta = load_clip_gt(clip_id)
    t0_us = int(gt_meta["events"][0]["event_start_timestamp"])
    data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
        local_dir=str(LOCAL_DIR),
        maybe_stream=True,
        num_frames=DEFAULT_NUM_FRAMES,
    )
    filled = {
        **rec,
        "hist_xy_full": _as_xy(data["ego_history_xyz"]).tolist(),
        "gt_xy_full": _as_xy(data["ego_future_xyz"]).tolist(),
    }
    return _xy_from_cached_record(filled, clip_id)


def _html_report(results: list[dict], generated_at: str) -> str:
    ades = [r["min_ade_m"] for r in results if r.get("min_ade_m") is not None]
    mean_ade = float(np.mean(ades)) if ades else None
    rows = []
    for i, r in enumerate(results):
        pred = html.escape((r["pred_coc"] or "").replace("\n", " "))
        gt = html.escape(r["gt_coc_texts"][0] if r["gt_coc_texts"] else "")
        ade = "—" if r.get("min_ade_m") is None else f"{r['min_ade_m']:.2f}"
        p_end = r.get("pred_xy") or {}
        g_end = r.get("gt_xy") or {}
        gs = r.get("gt_speed_start_end") or [None, None]
        ps = r.get("pred_speed_start_end") or [None, None]
        exp0 = (r.get("expert") or [{}])[0] if r.get("expert") else {}
        accel = exp0.get("accel_mean")
        accel_txt = "—" if accel is None else f"{accel:.2f}"
        rows.append(
            f"""<tr>
              <td class="px-3 py-2 text-slate-400">{i}</td>
              <td class="px-3 py-2 font-mono text-[11px]"><a class="text-cyan-400 hover:underline" href="#clip-{i}">{r['clip_id'][:8]}…</a></td>
              <td class="px-3 py-2 text-xs">{html.escape(str(r['event_cluster']))}</td>
              <td class="px-3 py-2 text-xs tabular-nums text-amber-300">{ade}</td>
              <td class="px-3 py-2 text-[11px] font-mono text-slate-400">
                GT {g_end.get('end', '—')} · pred {p_end.get('end', '—')}
              </td>
              <td class="px-3 py-2 text-[11px] font-mono text-slate-400">
                GT {gs[0]}→{gs[1]} · pred {ps[0]}→{ps[1]}
              </td>
              <td class="px-3 py-2 text-[11px] font-mono text-slate-400">{accel_txt}</td>
              <td class="px-3 py-2 text-xs text-slate-300">{gt}</td>
              <td class="px-3 py-2 text-xs text-amber-200">{pred}</td>
            </tr>"""
        )

    sections = []
    for i, r in enumerate(results):
        gt_block = "".join(f"<li class='mb-1'>{html.escape(t)}</li>" for t in r["gt_coc_texts"])
        pred = html.escape(r["pred_coc"] or "(none)")
        pred_raw = html.escape(r.get("pred_coc_raw") or "")
        ade = "n/a" if r.get("min_ade_m") is None else f"{r['min_ade_m']:.3f} m"
        p = r.get("pred_xy") or {}
        g = r.get("gt_xy") or {}
        exp0 = (r.get("expert") or [{}])[0] if r.get("expert") else {}
        exp_html = ""
        if exp0:
            exp_html = f"""
              <div class="bg-slate-950 border border-slate-800 rounded-2xl p-4 mb-5 text-[11px] font-mono text-slate-300">
                expert t0_v={exp0.get('t0_v')} m/s · accel mean={exp0.get('accel_mean')}
                min={exp0.get('accel_min')} max={exp0.get('accel_max')} ·
                kappa mean={exp0.get('kappa_mean')} |max|={exp0.get('kappa_abs_max')} ·
                pos0_ok={exp0.get('pos0_matches_offset_plus_delta')} hide={exp0.get('hide_start')}:{exp0.get('hide_end')}
              </div>"""
        thumbs = []
        for cam_i, cam_name in enumerate(r["cameras"]):
            cells = []
            for t_i, t_label in enumerate(TIME_LABELS):
                src = r["image_grid"][cam_i][t_i]
                cells.append(
                    f"""<figure class="m-0">
                      <img src="{html.escape(src)}" alt="{html.escape(cam_name)} {t_label}" class="w-full rounded-lg border border-slate-700 object-cover">
                      <figcaption class="mt-1 text-[10px] text-slate-500">{html.escape(cam_name)} · {t_label}</figcaption>
                    </figure>"""
                )
            thumbs.append("".join(cells))
        image_html = "".join(
            f'<div class="grid grid-cols-4 gap-2 mb-3">{row}</div>' for row in thumbs
        )
        plot_html = ""
        if r.get("traj_xy"):
            plot_html = f"""
              <div class="grid md:grid-cols-2 gap-4 mb-5">
                <figure class="m-0">
                  <img src="{html.escape(r['traj_xy'])}" alt="pred vs GT XY" class="w-full rounded-xl border border-slate-700 bg-slate-950">
                  <figcaption class="mt-1 text-[11px] text-slate-500">Bird's-eye XY · x+ forward ↑ · y+ left ← · cyan history · gray GT · amber pred (64 waypoints)</figcaption>
                </figure>
                <figure class="m-0">
                  <img src="{html.escape(r.get('traj_speed') or '')}" alt="speed vs time" class="w-full rounded-xl border border-slate-700 bg-slate-950">
                  <figcaption class="mt-1 text-[11px] text-slate-500">Waypoint speed (m/s) from the 64-pt pred</figcaption>
                </figure>
              </div>"""
        sections.append(
            f"""<section id="clip-{i}" class="bg-slate-900 border border-slate-700 rounded-3xl p-6 mb-8">
              <div class="flex flex-wrap items-baseline justify-between gap-3 mb-4">
                <div>
                  <div class="font-display text-xl text-white">Clip {i} · <span class="font-mono text-sm text-cyan-300">{html.escape(r['clip_id'])}</span></div>
                  <div class="text-xs text-slate-400 mt-1">chunk={r['chunk']} · {html.escape(r['split'])} · {html.escape(str(r['event_cluster']))} · t0_us={r['t0_us']} · seed={r['seed']}</div>
                </div>
                <div class="text-xs text-slate-400">minADE=<span class="text-amber-300 font-semibold">{ade}</span> · jaccard={r['jaccard']:.3f}</div>
              </div>
              <div class="grid md:grid-cols-2 gap-4 mb-5">
                <div class="bg-slate-950 border border-slate-800 rounded-2xl p-4">
                  <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2">GT CoC</div>
                  <ul class="text-sm text-slate-200 list-disc pl-5">{gt_block}</ul>
                </div>
                <div class="bg-slate-950 border border-amber-900/50 rounded-2xl p-4">
                  <div class="text-[11px] uppercase tracking-wider text-amber-500 mb-2">Generated CoC (T=0.6, 1 sample)</div>
                  <p class="text-sm text-amber-100">{pred}</p>
                  {f'<p class="mt-2 text-[11px] text-slate-500 font-mono break-all">{pred_raw}</p>' if pred_raw and pred_raw != pred else ""}
                </div>
              </div>
              <div class="grid md:grid-cols-2 gap-3 mb-5 text-xs font-mono">
                <div class="bg-slate-950 border border-slate-800 rounded-2xl p-4 text-emerald-300">
                  GT XY start {g.get('start')} → end {g.get('end')} · path {g.get('path_m')} m
                  · speed {r.get('gt_speed_start_end')} m/s
                </div>
                <div class="bg-slate-950 border border-slate-800 rounded-2xl p-4 text-amber-200">
                  Pred XY start {p.get('start')} → end {p.get('end')} · path {p.get('path_m')} m
                  · speed {r.get('pred_speed_start_end')} m/s
                </div>
              </div>
              {exp_html}
              {plot_html}
              <div class="text-sm font-semibold text-slate-300 mb-2">16-image history (4 cameras × 4 frames)</div>
              {image_html}
            </section>"""
        )

    mean_txt = "—" if mean_ade is None else f"{mean_ade:.2f} m"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Local traj sample · 5 clips · T=0.6</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@600&amp;display=swap');
    body {{ font-family: Inter, system-ui, sans-serif; }}
    .font-display {{ font-family: 'Space Grotesk', Inter, sans-serif; }}
  </style>
</head>
<body class="bg-slate-950 text-slate-200">
  <div class="max-w-6xl mx-auto px-6 py-10">
    <h1 class="font-display text-3xl text-white mb-2">Local PAI-CoC · CoC + 1 trajectory</h1>
    <p class="text-sm text-slate-400 mb-6">
      Same 5 clips as <a class="text-cyan-400 hover:underline" href="../coc_sample_5_t06/index.html">coc_sample_5_t06</a>
      (seed {SEED}, skip t0 &lt; 1.6 s). NVIDIA sampling: T={NVIDIA_TEMPERATURE}, top_p={NVIDIA_TOP_P},
      1 independent CoC + 1 expert trajectory per clip. Jaccard is cheap word overlap — read the sentences.
      After <code>dxy_theta_to_v_without_v0</code> t0. Plots use the generated 64-waypoint pred
      (<code>pred_xy_full</code>), vehicle frame x+ forward ↑ / y+ left ←.
      Stage 1b is <a class="text-cyan-400 hover:underline" href="../stage1b_progress.html">closed</a>
      (wiring). P2f binds NVIDIA pixel budget (<code>163840–196608</code>, grid
      <code>20×36</code>). Generated {html.escape(generated_at)}. Mean minADE: {mean_txt}.
    </p>
    <div class="overflow-auto bg-slate-900 border border-slate-700 rounded-2xl mb-10">
      <table class="w-full text-left text-sm">
        <thead class="text-[11px] uppercase tracking-wider text-slate-500 border-b border-slate-800">
          <tr>
            <th class="px-3 py-2">#</th><th class="px-3 py-2">Clip</th><th class="px-3 py-2">Cluster</th>
            <th class="px-3 py-2">minADE</th><th class="px-3 py-2">XY end</th>
            <th class="px-3 py-2">Speed start→end</th><th class="px-3 py-2">Accel mean</th>
            <th class="px-3 py-2">GT CoC</th><th class="px-3 py-2">Pred CoC</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="5-clip T=0.6 CoC + one expert trajectory.")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Ignore cached clip result.json and rerun inference.",
    )
    parser.add_argument(
        "--redraw-plots",
        action="store_true",
        help="Reuse cached CoC/traj JSON and only rewrite XY/speed PNGs.",
    )
    args = parser.parse_args()
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[traj-sample] {len(CLIP_IDS)} clips; T={NVIDIA_TEMPERATURE} "
        f"top_p={NVIDIA_TOP_P} num_traj_samples=1 seed={SEED}"
    )
    for i, cid in enumerate(CLIP_IDS):
        print(f"  [{i}] {cid}")

    model = None
    processor = None
    results: list[dict] = []
    for i, cid in enumerate(CLIP_IDS):
        print(f"\n[traj-sample] === clip {i}/{len(CLIP_IDS) - 1} {cid} ===")
        clip_dir = report_dir / f"{i:02d}_{cid[:8]}"
        cached = clip_dir / "result.json"
        rec = None if args.no_reuse else (json.loads(cached.read_text()) if cached.exists() else None)
        if rec is not None and rec.get("pred_xy") is not None:
            print(f"[traj-sample] reuse {cached}")
            if args.redraw_plots:
                hist_xy, gt_xy, pred_xy = _xy_for_redraw(rec, cid)
                if pred_xy is not None:
                    ade = rec.get("min_ade_m") or 0.0
                    clip_dir.mkdir(parents=True, exist_ok=True)
                    plots = _save_traj_plots(clip_dir, hist_xy, gt_xy, pred_xy, float(ade))
                    rec["traj_xy"] = plots.get("traj_xy")
                    rec["traj_speed"] = plots.get("traj_speed")
                    rec["hist_xy_full"] = np.round(hist_xy, 4).tolist()
                    rec["gt_xy_full"] = np.round(gt_xy, 4).tolist()
                    rec.pop("pred_xy_plot", None)
                    rec.pop("pred_xy_plot_note", None)
                    cached.write_text(json.dumps(rec, indent=2) + "\n")
                    print(f"[traj-sample] redrew plots {clip_dir}")
                else:
                    print(
                        "[traj-sample] skip redraw: no pred_xy_full in cache "
                        "(re-run with --no-reuse to persist the 64-pt pred)"
                    )
        else:
            if model is None:
                print("[traj-sample] loading AlpamayoR1MLX (expert on)…")
                model = AlpamayoR1MLX.from_pretrained(
                    str(CHECKPOINT), load_expert=True, dtype=mx.bfloat16
                )
                processor = get_processor(model.tokenizer)
            rec = run_one_clip(model, processor, cid, clip_dir, seed=SEED + i)
            cached.write_text(json.dumps(rec, indent=2) + "\n")
        results.append(rec)
        print(f"[traj-sample] GT:   {rec['gt_coc_texts']}")
        print(f"[traj-sample] PRED: {rec['pred_coc']}")
        print(
            f"[traj-sample] minADE={rec.get('min_ade_m')} m  "
            f"gt_end={rec.get('gt_xy', {}).get('end')}  "
            f"pred_end={(rec.get('pred_xy') or {}).get('end')}"
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    html_path = report_dir / "index.html"
    html_path.write_text(_html_report(results, generated_at))
    print(f"\n[traj-sample] wrote {html_path}")


if __name__ == "__main__":
    main()
