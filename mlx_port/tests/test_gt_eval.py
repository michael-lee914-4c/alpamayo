"""Unit tests for Stage 1 GT comparison (no model load)."""

import numpy as np
import pytest

from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    REASONING_PATH,
    _pred_xy_for_ade,
    clean_pred_coc,
    format_gt_report,
    list_local_coc_clips,
    load_clip_gt,
    load_coc_table,
    min_ade_xy,
    score_coc,
)

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


def test_min_ade_accepts_t_by_2_gt_without_transposing():
    """(T, 2) must stay waypoint-major. Transposing it would look like a 2-step path."""
    t = 8
    gt_t2 = np.stack([np.linspace(0.0, 7.0, t), np.zeros(t)], axis=-1)
    pred = gt_t2[None, ...]
    assert min_ade_xy(pred, gt_t2) == 0.0
    # NVIDIA (2, T) layout of the same XY still scores 0.
    assert min_ade_xy(np.transpose(pred, (0, 2, 1)), gt_t2.T) == 0.0


def test_pred_xy_for_ade_rank2_is_one_sample():
    xyz = np.zeros((6, 3), dtype=np.float64)
    xyz[:, 0] = np.arange(6)
    xy = _pred_xy_for_ade(xyz)
    assert xy.shape == (1, 6, 2)
    np.testing.assert_array_equal(xy[0, :, 0], np.arange(6))


def test_load_coc_table_missing_file(tmp_path):
    try:
        load_coc_table(tmp_path / "missing.parquet")
    except FileNotFoundError as exc:
        assert "CoC labels" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when the CoC table is absent")


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
