"""Aggregated reward subtracts 1.0 from the mean comfort score.

Weight-free. Complementary to open #25 (ADE ``< 3.0`` hard gate) and #13
(per-metric ``mean-1`` onto ``[-1, 0]`` — that shift lives in comfort
scoring comments; ``compute_comfort`` itself still returns ``[0, 1]``
fractions). ``compute_reward`` then does ``mean(comfort_dict) - 1.0`` before
mixing with ADE. Dropping this second shift would inflate every GRPO reward
by ``comfort_weight``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_AGG = _REPO / "finetune" / "rl" / "rewards" / "aggregated_reward.py"
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"


def test_source_comfort_returns_unshifted_fractions():
    src = _COMFORT.read_text()
    assert "comfort_metric_dict[name] = _within_bound" in src
    assert "comfort_metric_dict[name] = comfort_metric_dict[name].mean(dim=-1)" in src
    # No second shift inside compute_comfort.
    assert "comfort_score = comfort_score - 1.0" not in src


def test_source_aggregated_reward_means_then_subtracts_one():
    src = _AGG.read_text()
    assert "comfort_score = float(sum(comfort_dict_t.values()) / len(comfort_dict_t))" in src
    assert "comfort_score = comfort_score - 1.0" in src
    assert "ade_threshold = 3.0" in src
    assert "if l2_dist < ade_threshold:" in src
    assert 'final_reward = -1.0' in src or "final_reward = -1.0" in src


def test_mean_then_minus_one_maps_unit_interval_onto_minus_one_zero():
    comfort_dict = {
        "comfort/lon_accel": 1.0,
        "comfort/lat_accel": 1.0,
        "comfort/jerk": 0.0,
        "comfort/lon_jerk": 1.0,
        "comfort/yaw_accel": 1.0,
        "comfort/yaw_rate": 1.0,
    }
    comfort_score = float(sum(comfort_dict.values()) / len(comfort_dict))
    comfort_score = comfort_score - 1.0
    # Five of six metrics pass → 5/6 - 1 = -1/6.
    assert abs(comfort_score - (5.0 / 6.0 - 1.0)) < 1e-12
    all_pass = {k: 1.0 for k in comfort_dict}
    assert float(sum(all_pass.values()) / len(all_pass)) - 1.0 == 0.0
    all_fail = {k: 0.0 for k in comfort_dict}
    assert float(sum(all_fail.values()) / len(all_fail)) - 1.0 == -1.0
