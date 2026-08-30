"""Numpy helpers for traj-sample plots. Safe to import without dataset extras."""

from __future__ import annotations

import numpy as np

DT_S = 0.1
N_WAYPOINTS = 64


def _as_xy(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 4:
        a = a[0, 0]
    elif a.ndim == 3:
        a = a[0]
    return a[:, :2]


def _speed_mps(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    prev = np.concatenate([np.zeros((1, 2), dtype=np.float64), xy[:-1]], axis=0)
    return np.linalg.norm(xy - prev, axis=-1) / DT_S


def _require_full_xy(xy: np.ndarray, name: str) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[0] != N_WAYPOINTS or xy.shape[1] < 2:
        raise ValueError(
            f"{name} must be ({N_WAYPOINTS}, 2+), got {getattr(xy, 'shape', None)}"
        )
    return xy


def _xy_for_redraw(
    rec: dict, clip_id: str = ""
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Hist/GT/pred XY from a cached record. Pred is the saved 64-pt path only."""
    del clip_id
    hist = rec.get("hist_xy_full")
    gt = rec.get("gt_xy_full")
    if hist is None or gt is None:
        raise ValueError("record missing hist_xy_full or gt_xy_full")
    hist_xy, gt_xy = np.asarray(hist), np.asarray(gt)
    pred_full = rec.get("pred_xy_full")
    if pred_full is None:
        return hist_xy, gt_xy, None
    return hist_xy, gt_xy, _require_full_xy(pred_full, "pred_xy_full")


def quant_path_label(flags: dict | None) -> str:
    """HTML/log stamp for the load path. Safe without dataset extras."""
    lm = str((flags or {}).get("lm") or "bf16")
    if lm.startswith("affine-4"):
        return "T3.1 W4 LM"
    if lm == "bf16":
        return "bf16"
    return lm


def load_path_stamp(run_meta: dict | None = None) -> dict[str, str]:
    """Title + flag line for the traj-sample HTML banner."""
    meta = run_meta or {}
    flags = meta.get("quantized") or {}
    path = str(meta.get("quant_path") or quant_path_label(flags))
    flag_txt = (
        f"lm={flags.get('lm', 'bf16')} · vision={flags.get('vision', 'bf16')} · "
        f"expert={flags.get('expert', 'bf16')}"
    )
    return {"quant_path": path, "flag_txt": flag_txt}
