"""Unit tests for Stage 1 GT comparison (no model load)."""

import json

import numpy as np
import pandas as pd
import pytest

from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    REASONING_PATH,
    _pred_xy_for_ade,
    clean_pred_coc,
    format_gt_report,
    list_local_coc_clips,
    list_local_traj_clips,
    load_clip_gt,
    min_ade_xy,
    score_coc,
)


def _write_pai(tmp_path, index: dict, coc: dict | None = None):
    """Minimal clip_index + optional CoC parquet tree."""
    root = tmp_path / "pai"
    (root / "reasoning").mkdir(parents=True)
    pd.DataFrame.from_dict(index, orient="index").to_parquet(root / "clip_index.parquet")
    if coc is not None:
        pd.DataFrame.from_dict(coc, orient="index").to_parquet(
            root / "reasoning" / "ood_reasoning.parquet"
        )
    return root

_HAS_PAI_COC = REASONING_PATH.exists()


@pytest.mark.skipif(not _HAS_PAI_COC, reason="PAI-CoC labels not on this machine")
def test_local_coc_subset_has_labeled_clips():
    clips = list_local_coc_clips()
    assert len(clips) >= 1
    assert DEFAULT_EVAL_CLIP_ID in clips.index


@pytest.mark.skipif(not _HAS_PAI_COC, reason="PAI-CoC labels not on this machine")
def test_default_clip_has_human_coc():
    gt = load_clip_gt(DEFAULT_EVAL_CLIP_ID)
    assert gt["gt_coc_texts"]
    assert "pedestrian" in gt["gt_coc_texts"][0].lower() or "yield" in gt["gt_coc_texts"][0].lower()


def test_clean_pred_coc_strips_specials_and_prefix():
    raw = "Y<|im_end|>Slow yield to the pedestrian.<|cot_end|><|traj_future_start|>"
    assert clean_pred_coc(raw) == "Slow yield to the pedestrian."


def test_clean_pred_coc_drops_token_after_cot_end():
    raw = (
        "Stop to yield to the pedestrian in the crosswalk."
        "<|cot_end|><|traj_future_start|> experience"
    )
    assert clean_pred_coc(raw) == "Stop to yield to the pedestrian in the crosswalk."


def test_score_coc_perfect_match():
    gt = ["Slow down for the pedestrian in the crosswalk."]
    s = score_coc(gt[0], gt)
    assert s["readable"]
    assert s["jaccard"] == 1.0


def test_score_coc_garbage_is_not_readable():
    s = score_coc("<alpamayo_ext_155683> __(", ["Yield to the pedestrian."])
    assert s["readable"] is False
    assert s["jaccard"] < 0.2


def test_score_coc_punctuation_is_not_readable():
    s = score_coc('.!"!!!!!!!!!!!!"!"!', ["Yield to the pedestrian."])
    assert s["readable"] is False


def test_min_ade_zero_when_identical():
    gt = np.zeros((2, 64), dtype=np.float32)
    gt[0] = np.linspace(0, 6.4, 64)
    pred = gt[None, ...]  # (1, 2, T)
    assert min_ade_xy(pred, gt) == 0.0


def _toy_gt_report_inputs(t: int = 4):
    gt = {
        "clip_id": "toy",
        "split": "train",
        "chunk": 0,
        "event_cluster": "unit",
        "gt_coc_texts": ["Yield to the pedestrian."],
    }
    gt_xyz = np.zeros((t, 3), dtype=np.float64)
    gt_xyz[:, 0] = np.arange(t, dtype=np.float64)
    bad = gt_xyz + 10.0
    good = gt_xyz.copy()
    return gt, gt_xyz, np.stack([bad, good], axis=0)


def test_pred_xy_for_ade_rank3_is_samples_t_2():
    _, _, pred = _toy_gt_report_inputs()
    xy = _pred_xy_for_ade(pred)
    assert xy.shape == (2, 4, 2)


def test_format_gt_report_rank3_pred_uses_best_sample():
    gt, gt_xyz, pred = _toy_gt_report_inputs()
    report = format_gt_report(gt, pred_xyz=pred, ego_future_xyz=gt_xyz)
    assert "minADE=0.000 m" in report


def test_format_gt_report_rank5_pred_matches_nvidia_layout():
    gt, gt_xyz, pred = _toy_gt_report_inputs()
    pred5 = pred.reshape(1, 1, 2, 4, 3)
    report = format_gt_report(gt, pred_xyz=pred5, ego_future_xyz=gt_xyz)
    assert "minADE=0.000 m" in report


def test_format_gt_report_rank2_single_traj():
    gt, gt_xyz, pred = _toy_gt_report_inputs()
    report = format_gt_report(gt, pred_xyz=pred[1], ego_future_xyz=gt_xyz)
    assert "minADE=0.000 m" in report


def test_list_local_traj_clips_excludes_coc_and_invalid(tmp_path):
    root = _write_pai(
        tmp_path,
        index={
            "keep": {"clip_is_valid": True, "chunk": 1},
            "coc_clip": {"clip_is_valid": True, "chunk": 2},
            "invalid": {"clip_is_valid": False, "chunk": 3},
            "far_chunk": {"clip_is_valid": True, "chunk": 300},
            "neg_chunk": {"clip_is_valid": True, "chunk": -1},
        },
        coc={
            "coc_clip": {
                "split": "val",
                "event_cluster": "yield",
                "events": [{"coc": "Yield."}],
            }
        },
    )
    table = list_local_traj_clips(root, chunk_max=249, exclude_coc=True)
    assert list(table.index.astype(str)) == ["keep"]
    kept = list_local_traj_clips(root, chunk_max=249, exclude_coc=False)
    assert set(kept.index.astype(str)) == {"keep", "coc_clip"}


def test_list_local_traj_clips_empty_or_missing_coc_raises(tmp_path):
    only_coc = _write_pai(
        tmp_path / "only_coc",
        index={"coc_clip": {"clip_is_valid": True, "chunk": 0}},
        coc={"coc_clip": {"split": "val", "event_cluster": "x", "events": []}},
    )
    with pytest.raises(RuntimeError, match="no traj clips"):
        list_local_traj_clips(only_coc, exclude_coc=True)

    no_coc = _write_pai(
        tmp_path / "no_coc",
        index={"keep": {"clip_is_valid": True, "chunk": 0}},
        coc=None,
    )
    with pytest.raises(FileNotFoundError, match="CoC labels"):
        list_local_traj_clips(no_coc, exclude_coc=True)
    kept = list_local_traj_clips(no_coc, exclude_coc=False)
    assert list(kept.index.astype(str)) == ["keep"]


def test_list_local_coc_clips_filters_chunk_and_split(tmp_path):
    root = _write_pai(
        tmp_path,
        index={
            "train_ok": {"clip_is_valid": True, "chunk": 1},
            "val_ok": {"clip_is_valid": True, "chunk": 2},
            "invalid": {"clip_is_valid": False, "chunk": 3},
            "far": {"clip_is_valid": True, "chunk": 300},
        },
        coc={
            "train_ok": {"split": "train", "event_cluster": "a", "events": []},
            "val_ok": {"split": "val", "event_cluster": "b", "events": []},
            "invalid": {"split": "val", "event_cluster": "c", "events": []},
            "far": {"split": "val", "event_cluster": "d", "events": []},
            "missing_index": {"split": "val", "event_cluster": "e", "events": []},
        },
    )
    all_local = list_local_coc_clips(root, chunk_max=249)
    assert set(all_local.index.astype(str)) == {"train_ok", "val_ok"}
    val_only = list_local_coc_clips(root, chunk_max=249, split="val")
    assert list(val_only.index.astype(str)) == ["val_ok"]


def test_load_clip_gt_parses_events_and_missing_index(tmp_path):
    events = [
        {"coc": "Yield to the pedestrian.", "event_start_timestamp": 10},
        {"coc": "", "event_start_timestamp": 20},
        {"event_start_timestamp": 30},
    ]
    root = _write_pai(
        tmp_path,
        index={"labeled": {"clip_is_valid": True, "chunk": 7}},
        coc={
            "labeled": {
                "split": "val",
                "event_cluster": "yield",
                "events": json.dumps(events),
            },
            "no_index": {
                "split": "train",
                "event_cluster": "other",
                "events": json.dumps(json.dumps([{"coc": "Stop."}])),
            },
        },
    )
    gt = load_clip_gt("labeled", local_dir=root)
    assert gt["clip_id"] == "labeled"
    assert gt["split"] == "val"
    assert gt["chunk"] == 7
    assert gt["gt_coc_texts"] == ["Yield to the pedestrian."]

    orphan = load_clip_gt("no_index", local_dir=root)
    assert orphan["chunk"] is None
    assert orphan["gt_coc_texts"] == ["Stop."]

    with pytest.raises(KeyError, match="no CoC label"):
        load_clip_gt("missing", local_dir=root)


def test_score_coc_readable_needs_three_letter_words():
    gt = ["Yield to the pedestrian in the crosswalk."]
    short = score_coc("ok go", gt)
    assert short["readable"] is False
    sentence = score_coc("stop the car", gt)
    assert sentence["readable"] is True
    angled = score_coc("<stop the car now", gt)
    assert angled["readable"] is False
