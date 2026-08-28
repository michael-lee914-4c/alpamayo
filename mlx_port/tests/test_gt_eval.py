"""Unit tests for Stage 1 GT comparison (no model load)."""

import numpy as np

from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    clean_pred_coc,
    list_local_coc_clips,
    load_clip_gt,
    min_ade_xy,
    score_coc,
)


def test_local_coc_subset_has_labeled_clips():
    clips = list_local_coc_clips()
    assert len(clips) >= 1
    assert DEFAULT_EVAL_CLIP_ID in clips.index


def test_default_clip_has_human_coc():
    gt = load_clip_gt(DEFAULT_EVAL_CLIP_ID)
    assert gt["gt_coc_texts"]
    assert "pedestrian" in gt["gt_coc_texts"][0].lower() or "yield" in gt["gt_coc_texts"][0].lower()


def test_clean_pred_coc_strips_specials_and_prefix():
    raw = "Y<|im_end|>Slow yield to the pedestrian.<|cot_end|><|traj_future_start|>"
    assert clean_pred_coc(raw) == "Slow yield to the pedestrian."


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
