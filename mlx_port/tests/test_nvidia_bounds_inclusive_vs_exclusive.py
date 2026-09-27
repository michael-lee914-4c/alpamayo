"""Unicycle physical gates are inclusive; comfort gates are exclusive.

Weight-free. Complementary to open #42 (unicycle denormalizes before the
physical check) and #25 (comfort exclusive min-bound). A value sitting
exactly on ``accel_bounds`` / ``MAX_LON_ACCEL`` is *in-bounds* for
``is_within_bounds`` (``>=`` / ``<=``) and *out-of-bounds* for
``_within_bound`` (``>`` / ``<``). Swapping the inequalities would
silently flip SFT action filtering vs RL comfort reward on the same
threshold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"


def _unicycle_within(accel: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (accel >= lo) & (accel <= hi)


def _comfort_within(metric: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (metric > lo) & (metric < hi)


def test_source_unicycle_uses_inclusive_comparisons():
    src = _UNI.read_text()
    body = src.split("def is_within_bounds")[1].split("def _v_to_a")[0]
    assert "accel >= self.accel_bounds[0]" in body
    assert "accel <= self.accel_bounds[1]" in body
    assert "kappa >= self.curvature_bounds[0]" in body
    assert "kappa <= self.curvature_bounds[1]" in body
    assert "accel > self.accel_bounds[0]" not in body


def test_source_comfort_uses_exclusive_comparisons():
    src = _COMFORT.read_text()
    body = src.split("def _within_bound")[1].split("def gather_dynamics")[0]
    assert "metric > min_bound" in body
    assert "metric < max_bound" in body
    assert "metric >= min_bound" not in body
    assert "metric <= max_bound" not in body


def test_exact_bound_passes_unicycle_and_fails_comfort():
    lo, hi = -4.05, 2.40
    on_hi = np.array([hi])
    assert _unicycle_within(on_hi, lo, hi).all()
    assert not _comfort_within(on_hi, lo, hi).any()
    inside = np.array([(lo + hi) / 2.0])
    assert _unicycle_within(inside, lo, hi).all()
    assert _comfort_within(inside, lo, hi).all()
