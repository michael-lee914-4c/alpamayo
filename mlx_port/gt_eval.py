"""Stage 1 ground-truth comparison for the local PAI-CoC subset.

Stay on Stage 1 until both of these pass on a local clip:

1. CoC — generated reasoning is readable English and aligns with the
   human label in ``reasoning/ood_reasoning.parquet``.
2. Action expert — predicted XY vs ``ego_future_xyz`` minADE is in a
   physical range (NVIDIA reports meters, not hundreds of m/s²).

NVIDIA's ``test_inference.py`` only scores trajectory (minADE) and prints
CoC. This module also scores CoC against the human labels that shipped
with the CoC subset.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOCAL_PAI_COC = Path("/Volumes/MicronSSD/pai_coc")
REASONING_PATH = LOCAL_PAI_COC / "reasoning" / "ood_reasoning.parquet"
CLIP_INDEX_PATH = LOCAL_PAI_COC / "clip_index.parquet"

# Clip used in earlier PAI-CoC runs: chunk 0, local cameras + egomotion + CoC label.
DEFAULT_EVAL_CLIP_ID = "0abe118e-aa79-41f6-a719-f2df8abaf1ea"

# Downloaded camera/egomotion chunks on this machine.
LOCAL_CHUNK_MAX = 249

_WORD_RE = re.compile(r"[a-z0-9']+")
_SPECIAL_RE = re.compile(r"<\|[^|>]+?\|>")


def clean_pred_coc(text: str | None) -> str:
    """Strip Alpamayo special tokens and leftover single-letter prefixes."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", _SPECIAL_RE.sub(" ", text)).strip()
    return re.sub(r"^[A-Z]\s+", "", cleaned)


def load_coc_table(reasoning_path: Path = REASONING_PATH) -> pd.DataFrame:
    if not reasoning_path.exists():
        raise FileNotFoundError(f"CoC labels not found: {reasoning_path}")
    return pd.read_parquet(reasoning_path)


def list_local_coc_clips(
    local_dir: Path = LOCAL_PAI_COC,
    chunk_max: int = LOCAL_CHUNK_MAX,
    split: str | None = None,
) -> pd.DataFrame:
    """CoC-labeled clips whose chunk is on the local disk (0..chunk_max)."""
    coc = load_coc_table(local_dir / "reasoning" / "ood_reasoning.parquet")
    index = pd.read_parquet(local_dir / "clip_index.parquet")
    joined = coc.join(index[["chunk", "clip_is_valid"]], how="left")
    mask = (
        joined["clip_is_valid"].fillna(False)
        & joined["chunk"].notna()
        & (joined["chunk"] >= 0)
        & (joined["chunk"] <= chunk_max)
    )
    out = joined.loc[mask]
    if split is not None:
        out = out[out["split"] == split]
    return out


def load_clip_gt(clip_id: str, local_dir: Path = LOCAL_PAI_COC) -> dict[str, Any]:
    """Human CoC events + chunk/split for one clip."""
    coc = load_coc_table(local_dir / "reasoning" / "ood_reasoning.parquet")
    if clip_id not in coc.index:
        raise KeyError(f"{clip_id} has no CoC label in {REASONING_PATH}")
    row = coc.loc[clip_id]
    events = json.loads(row["events"]) if isinstance(row["events"], str) else row["events"]
    if isinstance(events, str):
        events = json.loads(events)
    index = pd.read_parquet(local_dir / "clip_index.parquet")
    chunk = int(index.loc[clip_id, "chunk"]) if clip_id in index.index else None
    return {
        "clip_id": clip_id,
        "split": str(row["split"]),
        "event_cluster": str(row["event_cluster"]),
        "events": list(events),
        "gt_coc_texts": [e["coc"] for e in events if e.get("coc")],
        "chunk": chunk,
    }


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def score_coc(pred_coc: str | None, gt_texts: list[str]) -> dict[str, Any]:
    """Cheap lexical overlap vs human CoC. Not a substitute for reading the text."""
    pred = clean_pred_coc(pred_coc)
    pred_toks = _tokenize(pred)
    letter_words = [w for w in pred_toks if any(c.isalpha() for c in w) and len(w) >= 3]
    readable = len(letter_words) >= 3 and not pred.strip().startswith("<")
    best = {
        "jaccard": 0.0,
        "pred_coverage": 0.0,
        "gt_coverage": 0.0,
        "matched_gt": None,
    }
    for gt in gt_texts:
        gt_toks = _tokenize(gt)
        if not gt_toks:
            continue
        inter = pred_toks & gt_toks
        union = pred_toks | gt_toks
        jaccard = len(inter) / len(union) if union else 0.0
        pred_cov = len(inter) / len(pred_toks) if pred_toks else 0.0
        gt_cov = len(inter) / len(gt_toks) if gt_toks else 0.0
        if jaccard >= best["jaccard"]:
            best = {
                "jaccard": jaccard,
                "pred_coverage": pred_cov,
                "gt_coverage": gt_cov,
                "matched_gt": gt,
            }
    return {
        "readable": readable,
        "pred_len": len(pred),
        **best,
    }


def min_ade_xy(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    """NVIDIA test_inference.py minADE.

    pred_xy: (num_samples, 2, T) or (num_samples, T, 2)
    gt_xy:   (2, T) or (T, 2)
    """
    pred = np.asarray(pred_xy, dtype=np.float64)
    gt = np.asarray(gt_xy, dtype=np.float64)

    if gt.ndim == 2 and gt.shape[0] == 2:
        gt_t = gt.T  # (T, 2)
    else:
        gt_t = gt  # (T, 2)

    if pred.ndim == 3 and pred.shape[1] == 2:
        pred_t = np.transpose(pred, (0, 2, 1))  # (S, T, 2)
    elif pred.ndim == 2:
        pred_t = pred[None, ...]
    else:
        pred_t = pred  # (S, T, 2)

    t = min(pred_t.shape[1], gt_t.shape[0])
    diff = np.linalg.norm(pred_t[:, :t, :2] - gt_t[None, :t, :2], axis=-1).mean(-1)
    return float(diff.min())


def _pred_xy_for_ade(pred: np.ndarray) -> np.ndarray:
    """Normalize pred_xyz to (num_samples, T, 2) for min_ade_xy.

    Accepted layouts:
      (B, num_traj_sets, num_samples, T, 3)  NVIDIA rollout
      (B or sets, num_samples, T, 3)
      (num_samples, T, 3) or (num_samples, T, 2)
      (T, 3) or (T, 2)
    """
    pred = np.asarray(pred)
    if pred.ndim == 5:
        return np.asarray(pred[0, 0, :, :, :2])
    if pred.ndim == 4:
        return np.asarray(pred[0, :, :, :2])
    if pred.ndim == 3:
        return np.asarray(pred[:, :, :2])
    if pred.ndim == 2:
        return np.asarray(pred[None, :, :2])
    return pred


def format_gt_report(
    gt: dict[str, Any],
    pred_coc: str | None = None,
    pred_xyz: Any = None,
    ego_future_xyz: Any = None,
) -> str:
    lines = [
        f"clip_id={gt['clip_id']}  split={gt['split']}  chunk={gt['chunk']}",
        f"event_cluster={gt['event_cluster']}",
        "GT CoC:",
    ]
    for i, text in enumerate(gt["gt_coc_texts"]):
        lines.append(f"  [{i}] {text}")

    if pred_coc is not None:
        coc_score = score_coc(pred_coc, gt["gt_coc_texts"])
        lines.append("Pred CoC:")
        lines.append(f"  {pred_coc[:500]!r}")
        lines.append(
            f"  readable={coc_score['readable']}  jaccard={coc_score['jaccard']:.3f}  "
            f"gt_coverage={coc_score['gt_coverage']:.3f}"
        )

    if pred_xyz is not None and ego_future_xyz is not None:
        pred = np.asarray(pred_xyz)
        gt_xyz = np.asarray(ego_future_xyz)
        # NVIDIA: gt_xy = ego_future_xyz[0, 0, :, :2].T  → (2, T)
        if gt_xyz.ndim == 4:
            gt_xy = gt_xyz[0, 0, :, :2].T
        elif gt_xyz.ndim == 3:
            gt_xy = gt_xyz[0, :, :2].T
        else:
            gt_xy = gt_xyz[:, :2].T if gt_xyz.shape[-1] >= 2 else gt_xyz
        pred_xy = _pred_xy_for_ade(pred)
        ade = min_ade_xy(pred_xy, gt_xy)
        lines.append(f"minADE={ade:.3f} m  (pred shape={pred.shape})")
    elif pred_xyz is None:
        lines.append("minADE=n/a  (action expert not run)")

    return "\n".join(lines)
