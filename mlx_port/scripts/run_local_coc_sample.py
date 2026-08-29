"""Run greedy CoC inference on 5 random local PAI-CoC clips and write an HTML report.

Does not download clips. Only uses labels + camera/egomotion already on disk
(chunks 0–249). NVIDIA's published test_inference.py clip is skipped because
it is not in the CoC subset.
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
from mlx_port.gt_eval import clean_pred_coc, list_local_coc_clips, load_clip_gt, score_coc
from mlx_port.inference import generate_top_k_coc, sample_n_coc
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.processor import (
    DEFAULT_NUM_FRAMES,
    alpamayo_apply_chat_template,
    create_message,
    get_processor,
)

LOCAL_DIR = Path("/Volumes/MicronSSD/pai_coc")
CHECKPOINT = Path("/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B")
REPORT_DIR = Path("/Users/michaellee/Projects/alpamayo/reports/coc_sample_5")
SEED = 42
N_CLIPS = 5
CAM_NAMES = {
    0: "cross_left_120",
    1: "front_wide_120",
    2: "cross_right_120",
    6: "front_tele_30",
}
TIME_LABELS = ["t-0.3s", "t-0.2s", "t-0.1s", "t+0.0s"]
# NVIDIA loader requires t0 > num_history_steps * 0.1s (1.6s).
MIN_T0_US = 1_600_001


def _first_event_t0(events: object) -> int | None:
    if events is None or (isinstance(events, float) and np.isnan(events)):
        return None
    if isinstance(events, str):
        events = json.loads(events)
    if not events:
        return None
    ev0 = events[0]
    if ev0 is None:
        return None
    ts = ev0.get("event_start_timestamp")
    return int(ts) if ts is not None else None


def select_local_coc_clips(n: int, seed: int) -> list[str]:
    clips = list_local_coc_clips()
    rng = np.random.default_rng(seed)
    ids = clips.index.to_numpy()
    first = rng.choice(ids, size=n, replace=False)
    chosen: list[str] = []
    used = set()
    for cid in first:
        used.add(str(cid))
        t0 = _first_event_t0(clips.loc[cid, "events"])
        if t0 is None or t0 < MIN_T0_US:
            print(f"[coc-sample] skip {cid} (t0_us={t0} < {MIN_T0_US})")
            continue
        chosen.append(str(cid))
    rest = np.array([c for c in ids if str(c) not in used])
    for cid in rng.permutation(rest):
        if len(chosen) >= n:
            break
        t0 = _first_event_t0(clips.loc[cid, "events"])
        if t0 is None or t0 < MIN_T0_US:
            print(f"[coc-sample] skip {cid} (t0_us={t0} < {MIN_T0_US})")
            continue
        chosen.append(str(cid))
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} eligible local CoC clips with t0>1.6s")
    return chosen


def _as_uint8_hwc(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return arr


def _save_frames(clip_dir: Path, frames: np.ndarray, camera_indices: np.ndarray) -> list[list[str]]:
    """Save 16 JPEGs. Returns paths relative to REPORT_DIR, rows=cameras, cols=time."""
    rel_grid: list[list[str]] = []
    for cam_i, cam_id in enumerate(camera_indices.tolist()):
        cam_name = CAM_NAMES.get(int(cam_id), f"cam{int(cam_id)}")
        row: list[str] = []
        for t_i, t_label in enumerate(TIME_LABELS):
            name = f"{cam_i:02d}_{cam_name}_{t_label.replace('+', 'p')}.jpg"
            path = clip_dir / name
            Image.fromarray(_as_uint8_hwc(frames[cam_i, t_i])).save(path, quality=85)
            row.append(f"{clip_dir.name}/{name}")
        rel_grid.append(row)
    return rel_grid


def _save_contact_sheet(clip_dir: Path, frames: np.ndarray, camera_indices: np.ndarray) -> str:
    cell_w, cell_h = 480, 270
    pad, header = 6, 22
    n_cam, n_t = frames.shape[:2]
    sheet = Image.new(
        "RGB",
        (n_t * cell_w + (n_t + 1) * pad, n_cam * (cell_h + header) + (n_cam + 1) * pad),
        (15, 23, 42),
    )
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for cam_i, cam_id in enumerate(camera_indices.tolist()):
        cam_name = CAM_NAMES.get(int(cam_id), f"cam{int(cam_id)}")
        for t_i, t_label in enumerate(TIME_LABELS):
            x = pad + t_i * (cell_w + pad)
            y = pad + cam_i * (cell_h + header + pad)
            draw.text((x, y), f"{cam_name}  {t_label}", fill=(226, 232, 240), font=font)
            thumb = Image.fromarray(_as_uint8_hwc(frames[cam_i, t_i])).resize((cell_w, cell_h))
            sheet.paste(thumb, (x, y + header))
    name = "grid.jpg"
    sheet.save(clip_dir / name, quality=88)
    return f"{clip_dir.name}/{name}"


def _save_ego_plot(clip_dir: Path, xyz: np.ndarray, future_xyz: np.ndarray | None) -> str:
    hist = np.asarray(xyz, dtype=np.float64)
    fut = np.asarray(future_xyz, dtype=np.float64) if future_xyz is not None else None
    pts = [hist[:, :2]]
    if fut is not None:
        pts.append(fut[:, :2])
    all_xy = np.concatenate(pts, axis=0)
    pad = 1.5
    ymin, ymax = float(all_xy[:, 1].min()) - pad, float(all_xy[:, 1].max()) + pad
    xmin, xmax = float(all_xy[:, 0].min()) - pad, float(all_xy[:, 0].max()) + pad
    span = max(ymax - ymin, xmax - xmin, 1.0)
    cy, cx = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    ymin, ymax = cy - 0.5 * span, cy + 0.5 * span
    xmin, xmax = cx - 0.5 * span, cx + 0.5 * span

    w, h, margin = 640, 640, 56
    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> tuple[int, int]:
        # Vehicle frame: x forward, y left. Screen: +x up, +y to the left.
        px = margin + (ymax - y) / span * (w - 2 * margin)
        py = h - margin - (x - xmin) / span * (h - 2 * margin)
        return int(px), int(py)

    for k in range(5):
        t = k / 4
        x = xmin + t * span
        y = ymin + t * span
        draw.line([to_px(x, ymin), to_px(x, ymax)], fill=(51, 65, 85), width=1)
        draw.line([to_px(xmin, y), to_px(xmax, y)], fill=(51, 65, 85), width=1)
    draw.line([to_px(xmin, 0), to_px(xmax, 0)], fill=(71, 85, 105), width=1)
    draw.line([to_px(0, ymin), to_px(0, ymax)], fill=(71, 85, 105), width=1)

    if fut is not None:
        draw.line([to_px(p[0], p[1]) for p in fut], fill=(148, 163, 184), width=2)
    draw.line([to_px(p[0], p[1]) for p in hist], fill=(56, 189, 248), width=3)
    for p in hist:
        xy = to_px(p[0], p[1])
        draw.ellipse([xy[0] - 3, xy[1] - 3, xy[0] + 3, xy[1] + 3], fill=(56, 189, 248))
    o = to_px(hist[0, 0], hist[0, 1])
    draw.ellipse([o[0] - 5, o[1] - 5, o[0] + 5, o[1] + 5], fill=(245, 158, 11))
    t0 = to_px(0.0, 0.0)
    draw.ellipse([t0[0] - 6, t0[1] - 6, t0[0] + 6, t0[1] + 6], fill=(244, 63, 94))
    draw.text((12, 10), "Ego-frame XY  (x+ forward ↑, y+ left ←)", fill=(226, 232, 240))
    draw.text((12, h // 2 - 8), "← y+", fill=(148, 163, 184))
    draw.text((w // 2 - 12, 10), "x+ ↑", fill=(148, 163, 184))
    draw.text((12, h - 36), "cyan=history  amber=oldest  red=t0  gray=GT future", fill=(148, 163, 184))
    name = "ego_history.png"
    img.save(clip_dir / name)
    return f"{clip_dir.name}/{name}"


def _prepare_inputs(processor, frames) -> dict:
    messages = create_message(frames.flatten(0, 1))
    inputs = alpamayo_apply_chat_template(
        processor,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="np",
    )
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            arr = inputs[key]
            if hasattr(arr, "shape") and len(arr.shape) == 5 and arr.shape[-1] == 3:
                inputs[key] = np.transpose(arr, (0, 4, 1, 2, 3))
    return inputs


def _score_top5(top5: list[dict], gt_texts: list[str]) -> list[dict]:
    scored = []
    for row in top5:
        pred = clean_pred_coc(row.get("raw"))
        score = score_coc(pred, gt_texts)
        scored.append(
            {
                "rank": int(row["rank"]),
                "first_token": row["first_token"],
                "first_token_id": int(row["first_token_id"]),
                "first_p": float(row["first_p"]),
                "pred_coc": pred,
                "pred_coc_raw": row.get("raw"),
                "readable": bool(score["readable"]),
                "jaccard": float(score["jaccard"]),
                "gt_coverage": float(score["gt_coverage"]),
            }
        )
    return scored


def _record_from_data(
    clip_id: str,
    clip_dir: Path,
    gt: dict,
    data: dict,
    pred_raw: str | None,
    top5: list[dict] | None = None,
) -> dict:
    frames = data["image_frames"]
    frames_np = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
    cam_idx = np.asarray(data["camera_indices"])
    xyz = np.asarray(data["ego_history_xyz"][0, 0])
    rot = np.asarray(data["ego_history_rot"][0, 0])
    fut = np.asarray(data["ego_future_xyz"][0, 0])
    clip_dir.mkdir(parents=True, exist_ok=True)
    image_grid = _save_frames(clip_dir, frames_np, cam_idx)
    contact = _save_contact_sheet(clip_dir, frames_np, cam_idx)
    ego_plot = _save_ego_plot(clip_dir, xyz, fut)
    pred_coc = clean_pred_coc(pred_raw)
    score = score_coc(pred_coc, gt["gt_coc_texts"])
    yaw_deg = np.degrees(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
    return {
        "clip_id": clip_id,
        "chunk": gt["chunk"],
        "split": gt["split"],
        "event_cluster": gt["event_cluster"],
        "t0_us": int(gt["events"][0]["event_start_timestamp"]),
        "n_events": len(gt["events"]),
        "gt_coc_texts": gt["gt_coc_texts"],
        "pred_coc_raw": pred_raw,
        "pred_coc": pred_coc,
        "readable": bool(score["readable"]),
        "jaccard": float(score["jaccard"]),
        "gt_coverage": float(score["gt_coverage"]),
        "image_grid": image_grid,
        "contact_sheet": contact,
        "ego_plot": ego_plot,
        "ego_history_xyz": xyz.round(4).tolist(),
        "ego_history_yaw_deg": yaw_deg.round(3).tolist(),
        "ego_path_m": float(np.linalg.norm(xyz[:, :2], axis=-1).max()),
        "cameras": [CAM_NAMES.get(int(i), str(int(i))) for i in cam_idx.tolist()],
        "top5_coc": _score_top5(top5 or [], gt["gt_coc_texts"]),
    }


def run_one_clip(
    model,
    processor,
    clip_id: str,
    clip_dir: Path,
    *,
    mode: str = "topk",
    k: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = None,
) -> dict:
    gt = load_clip_gt(clip_id)
    t0_us = int(gt["events"][0]["event_start_timestamp"])
    data = load_physical_aiavdataset(
        clip_id,
        t0_us=t0_us,
        local_dir=str(LOCAL_DIR),
        maybe_stream=True,  # loader reads HF cache, not the flat SSD snapshot
        num_frames=DEFAULT_NUM_FRAMES,
    )
    inputs = _prepare_inputs(processor, data["image_frames"])
    payload = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    if mode == "samples":
        rows = sample_n_coc(
            model=model,
            data=payload,
            n=k,
            temperature=temperature,
            top_p=top_p,
            max_generation_length=256,
            seed=seed,
        )
    else:
        rows = generate_top_k_coc(
            model=model,
            data=payload,
            k=k,
            max_generation_length=256,
        )
    pred_raw = rows[0]["raw"] if rows else None
    return _record_from_data(clip_id, clip_dir, gt, data, pred_raw, top5=rows)


def _html_report(
    results: list[dict],
    generated_at: str,
    *,
    mode: str = "topk",
    k: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> str:
    n_read = sum(1 for r in results if r["readable"])
    rows = []
    for i, r in enumerate(results):
        pred = html.escape((r["pred_coc"] or "").replace("\n", " "))
        gt = html.escape(r["gt_coc_texts"][0] if r["gt_coc_texts"] else "")
        rows.append(
            f"""<tr>
              <td class="px-3 py-2 text-slate-400">{i}</td>
              <td class="px-3 py-2 font-mono text-[11px]"><a class="text-cyan-400 hover:underline" href="#clip-{i}">{r['clip_id'][:8]}…</a></td>
              <td class="px-3 py-2 text-xs">{html.escape(str(r['event_cluster']))}</td>
              <td class="px-3 py-2 text-xs {'text-emerald-400' if r['readable'] else 'text-rose-400'}">{r['readable']}</td>
              <td class="px-3 py-2 text-xs tabular-nums">{r['jaccard']:.3f}</td>
              <td class="px-3 py-2 text-xs text-slate-300">{gt}</td>
              <td class="px-3 py-2 text-xs text-amber-200">{pred}</td>
            </tr>"""
        )

    sections = []
    for i, r in enumerate(results):
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
        ego_rows = []
        for step, (xyz, yaw) in enumerate(zip(r["ego_history_xyz"], r["ego_history_yaw_deg"])):
            dt = -0.1 * (15 - step)
            ego_rows.append(
                f"<tr><td class='px-2 py-0.5'>{step}</td>"
                f"<td class='px-2 py-0.5'>{dt:+.1f}</td>"
                f"<td class='px-2 py-0.5'>{xyz[0]:.3f}</td>"
                f"<td class='px-2 py-0.5'>{xyz[1]:.3f}</td>"
                f"<td class='px-2 py-0.5'>{xyz[2]:.3f}</td>"
                f"<td class='px-2 py-0.5'>{yaw:.2f}</td></tr>"
            )
        gt_block = "".join(f"<li class='mb-1'>{html.escape(t)}</li>" for t in r["gt_coc_texts"])
        pred = html.escape(r["pred_coc"] or "(none)")
        pred_raw = html.escape(r.get("pred_coc_raw") or "")
        top5_rows = []
        for t in r.get("top5_coc") or []:
            top5_rows.append(
                f"<tr>"
                f"<td class='px-2 py-1 text-slate-400'>{t['rank']}</td>"
                f"<td class='px-2 py-1 tabular-nums text-cyan-300'>{t['first_p']:.3f}</td>"
                f"<td class='px-2 py-1 font-mono text-[11px] text-slate-300'>{html.escape(str(t['first_token']))}</td>"
                f"<td class='px-2 py-1 text-amber-100'>{html.escape(t.get('pred_coc') or '')}</td>"
                f"<td class='px-2 py-1 text-xs {'text-emerald-400' if t.get('readable') else 'text-rose-400'}'>{t.get('readable')}</td>"
                f"</tr>"
            )
        top5_html = (
            f"""<div class="mb-5 overflow-auto bg-slate-950 border border-slate-800 rounded-2xl">
              <div class="px-4 pt-3 text-[11px] uppercase tracking-wider text-amber-500">{
                f"{k} independent samples (T={temperature}, top_p={top_p})"
                if mode == "samples"
                else f"Top-{k} CoC (first-token rank, then greedy)"
              }</div>
              <table class="w-full text-left text-sm mt-2">
                <thead class="text-[11px] text-slate-500 border-b border-slate-800">
                  <tr><th class="px-2 py-1">#</th><th class="px-2 py-1">P(first)</th><th class="px-2 py-1">First</th><th class="px-2 py-1">CoC</th><th class="px-2 py-1">Readable</th></tr>
                </thead>
                <tbody>{''.join(top5_rows)}</tbody>
              </table>
            </div>"""
            if top5_rows
            else ""
        )
        sections.append(
            f"""<section id="clip-{i}" class="bg-slate-900 border border-slate-700 rounded-3xl p-6 mb-8">
              <div class="flex flex-wrap items-baseline justify-between gap-3 mb-4">
                <div>
                  <div class="font-display text-xl text-white">Clip {i} · <span class="font-mono text-sm text-cyan-300">{html.escape(r['clip_id'])}</span></div>
                  <div class="text-xs text-slate-400 mt-1">chunk={r['chunk']} · {html.escape(r['split'])} · {html.escape(str(r['event_cluster']))} · t0_us={r['t0_us']} · path {r['ego_path_m']:.1f} m</div>
                </div>
                <div class="text-xs text-slate-400">readable=<span class="{'text-emerald-400' if r['readable'] else 'text-rose-400'}">{r['readable']}</span> · jaccard={r['jaccard']:.3f} · gt_coverage={r['gt_coverage']:.3f}</div>
              </div>
              <div class="grid md:grid-cols-2 gap-4 mb-5">
                <div class="bg-slate-950 border border-slate-800 rounded-2xl p-4">
                  <div class="text-[11px] uppercase tracking-wider text-slate-500 mb-2">GT CoC</div>
                  <ul class="text-sm text-slate-200 list-disc pl-5">{gt_block}</ul>
                </div>
                <div class="bg-slate-950 border border-amber-900/50 rounded-2xl p-4">
                  <div class="text-[11px] uppercase tracking-wider text-amber-500 mb-2">{
                    "Generated CoC (sample 1)"
                    if mode == "samples"
                    else "Generated CoC (greedy = top-1)"
                  }</div>
                  <p class="text-sm text-amber-100">{pred}</p>
                  {f'<p class="mt-2 text-[11px] text-slate-500 font-mono break-all">{pred_raw}</p>' if pred_raw and pred_raw != pred else ""}
                </div>
              </div>
              {top5_html}
              <div class="text-sm font-semibold text-slate-300 mb-2">16-image history (4 cameras × 4 frames)</div>
              {image_html}
              <div class="grid md:grid-cols-[minmax(0,1fr)_280px] gap-4 mt-4">
                <div>
                  <div class="text-sm font-semibold text-slate-300 mb-2">Ego-motion history (16 steps @ 10 Hz, x+ forward ↑, y+ left ←)</div>
                  <img src="{html.escape(r['ego_plot'])}" alt="ego history" class="rounded-xl border border-slate-700 bg-slate-950 max-w-md">
                </div>
                <div class="overflow-auto max-h-80">
                  <table class="w-full text-[11px] tabular-nums text-slate-300">
                    <thead class="text-slate-500"><tr><th class="px-2 text-left">i</th><th class="px-2 text-left">s</th><th class="px-2 text-left">x</th><th class="px-2 text-left">y</th><th class="px-2 text-left">z</th><th class="px-2 text-left">yaw°</th></tr></thead>
                    <tbody>{''.join(ego_rows)}</tbody>
                  </table>
                </div>
              </div>
            </section>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Local CoC sample · 5 clips</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&amp;family=Space+Grotesk:wght@600&amp;display=swap');
    body {{ font-family: Inter, system-ui, sans-serif; }}
    .font-display {{ font-family: 'Space Grotesk', Inter, sans-serif; }}
  </style>
</head>
<body class="bg-slate-950 text-slate-200">
  <div class="max-w-6xl mx-auto px-6 py-10">
    <h1 class="font-display text-3xl text-white mb-2">Local PAI-CoC sample</h1>
    <p class="text-sm text-slate-400 mb-6">
      {N_CLIPS} random clips from the on-disk CoC subset (chunks 0–249, seed={SEED}).
      {
        f"NVIDIA sampling: {k} independent CoCs from one prefill "
        f"(temperature={temperature}, top_p={top_p}). Not first-token top-k."
        if mode == "samples"
        else (
          f"Top-{k} CoC: prefill once, take the {k} most likely first tokens "
          "(after the expert traj-bin mask), then greedy-complete each. Rank 1 is the greedy CoC."
        )
      }
      t0 is the CoC event timestamp, not NVIDIA’s 5.1 s default.
      Jaccard is cheap word overlap — read the sentences.
      Generated {html.escape(generated_at)}. Readable CoC: {n_read}/{len(results)}.
    </p>
    <div class="overflow-auto bg-slate-900 border border-slate-700 rounded-2xl mb-10">
      <table class="w-full text-left text-sm">
        <thead class="text-[11px] uppercase tracking-wider text-slate-500 border-b border-slate-800">
          <tr>
            <th class="px-3 py-2">#</th><th class="px-3 py-2">Clip</th><th class="px-3 py-2">Cluster</th>
            <th class="px-3 py-2">Readable</th><th class="px-3 py-2">Jaccard</th>
            <th class="px-3 py-2">GT</th><th class="px-3 py-2">Pred</th>
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
    parser = argparse.ArgumentParser(description="Run CoC on random local PAI-CoC clips.")
    parser.add_argument(
        "--mode",
        choices=("topk", "samples"),
        default="topk",
        help="topk = first-token rank + greedy. samples = NVIDIA T/top_p rollouts.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k first tokens or number of samples")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Write HTML/JSON here (default: reports/coc_sample_5)",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Ignore cached clip result.json and rerun inference.",
    )
    parser.add_argument(
        "--redraw-plots",
        action="store_true",
        help="Reuse cached CoC JSON and only rewrite ego_history.png.",
    )
    args = parser.parse_args()

    n_local = len(list_local_coc_clips())
    chosen = select_local_coc_clips(N_CLIPS, SEED)
    print(
        f"[coc-sample] {n_local} local CoC clips; seed={SEED}; "
        f"mode={args.mode} k={args.k} T={args.temperature} top_p={args.top_p}"
    )
    for i, cid in enumerate(chosen):
        print(f"  [{i}] {cid}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    model = None
    processor = None
    results = []
    for i, cid in enumerate(chosen):
        print(f"\n[coc-sample] === clip {i}/{N_CLIPS-1} {cid} ===")
        clip_dir = report_dir / f"{i:02d}_{cid[:8]}"
        cached = clip_dir / "result.json"
        rec = None if args.no_reuse else (json.loads(cached.read_text()) if cached.exists() else None)
        if rec is not None and rec.get("top5_coc"):
            print(f"[coc-sample] reuse {cached}")
            if args.redraw_plots:
                gt = load_clip_gt(str(cid))
                t0_us = int(gt["events"][0]["event_start_timestamp"])
                data = load_physical_aiavdataset(
                    str(cid),
                    t0_us=t0_us,
                    local_dir=str(LOCAL_DIR),
                    maybe_stream=True,
                    num_frames=DEFAULT_NUM_FRAMES,
                )
                xyz = np.asarray(data["ego_history_xyz"][0, 0])
                fut = np.asarray(data["ego_future_xyz"][0, 0])
                rec["ego_plot"] = _save_ego_plot(clip_dir, xyz, fut)
                cached.write_text(json.dumps(rec, indent=2) + "\n")
                print(f"[coc-sample] redrew ego plot {clip_dir}")
        else:
            if model is None:
                print("[coc-sample] loading AlpamayoR1MLX…")
                model = AlpamayoR1MLX.from_pretrained(
                    str(CHECKPOINT), load_expert=False, dtype=mx.bfloat16
                )
                processor = get_processor(model.tokenizer)
            rec = run_one_clip(
                model,
                processor,
                str(cid),
                clip_dir,
                mode=args.mode,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=SEED + i,
            )
            cached.write_text(json.dumps(rec, indent=2) + "\n")
            gc.collect()
            mx.clear_cache()
        results.append(rec)
        print(f"[coc-sample] GT:   {rec['gt_coc_texts']}")
        print(f"[coc-sample] PRED: {rec['pred_coc']}")
        for t in rec.get("top5_coc") or []:
            print(
                f"[coc-sample]   #{t['rank']} p={t['first_p']:.3f} "
                f"{t['first_token']!r} → {t['pred_coc']}"
            )
        print(
            f"[coc-sample] readable={rec['readable']} jaccard={rec['jaccard']:.3f} "
            f"gt_coverage={rec['gt_coverage']:.3f}"
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    html_path = report_dir / "index.html"
    html_path.write_text(
        _html_report(
            results,
            generated_at,
            mode=args.mode,
            k=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )
    print(f"\n[coc-sample] wrote {html_path}")


if __name__ == "__main__":
    main()
