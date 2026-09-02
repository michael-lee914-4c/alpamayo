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
    min_ade_xy,
    score_coc,
    split_train_eval,
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


def test_clean_pred_coc_empty_and_none_are_blank():
    assert clean_pred_coc(None) == ""
    assert clean_pred_coc("") == ""
    assert clean_pred_coc("   ") == ""


def test_clean_pred_coc_prefers_cot_end_even_if_traj_start_is_earlier():
    raw = (
        "hello<|traj_future_start|>leaked tokens"
        "<|cot_end|>this must not remain"
    )
    assert clean_pred_coc(raw) == "hello leaked tokens"


def test_clean_pred_coc_splits_on_traj_future_start_when_no_cot_end():
    raw = "Y Slow yield.<|traj_future_start|><alpamayo_ext_1>"
    assert clean_pred_coc(raw) == "Slow yield."


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


def test_min_ade_picks_best_sample_and_accepts_layouts():
    gt_t2 = np.stack([np.arange(4, dtype=np.float64), np.zeros(4)], axis=-1)
    far = gt_t2 + 10.0
    near = gt_t2.copy()
    near[:, 0] += 0.5
    pred_st2 = np.stack([far, near], axis=0)
    ade = min_ade_xy(pred_st2, gt_t2)
    assert abs(ade - 0.5) < 1e-9

    pred_s2t = np.transpose(pred_st2, (0, 2, 1))
    gt_2t = gt_t2.T
    assert abs(min_ade_xy(pred_s2t, gt_2t) - 0.5) < 1e-9


def test_min_ade_truncates_to_shorter_horizon():
    gt = np.zeros((3, 2), dtype=np.float64)
    gt[:, 0] = [0.0, 1.0, 100.0]
    pred = np.zeros((1, 2, 2), dtype=np.float64)
    pred[0, 0] = [0.0, 1.0]
    assert min_ade_xy(pred, gt) == 0.0


def test_split_train_eval_clamps_extreme_fracs_and_rejects_bounds():
    ids = ["a", "b", "c", "d"]
    train, ev = split_train_eval(ids, seed=0, train_frac=0.25)
    assert len(train) == 1
    assert len(ev) == 3
    assert not (set(train) & set(ev))
    train01, ev01 = split_train_eval(["x", "y"], seed=0, train_frac=0.01)
    assert len(train01) == 1 and len(ev01) == 1
    train99, ev99 = split_train_eval(["x", "y"], seed=0, train_frac=0.99)
    assert len(train99) == 1 and len(ev99) == 1
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            split_train_eval(ids, seed=0, train_frac=bad)
        except ValueError as exc:
            assert "train_frac" in str(exc)
        else:
            raise AssertionError(f"expected ValueError for train_frac={bad}")


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
