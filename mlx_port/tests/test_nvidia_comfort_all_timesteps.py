"""Comfort ``_within_bound`` is all-timesteps AND, not a mean over time.

Weight-free. Complementary to open #47 (exclusive ``>`` / ``<``) and
#16 (``.mean(2)`` then ``.mean(-1)`` over K then N). After the
per-timestep exclusive check, ``torch.all(..., axis=-1)`` fails the
whole sample if any timestep sits on or outside the gate. Averaging
the boolean mask would tolerate a single violent step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"


def test_source_uses_all_over_the_last_axis():
    src = _COMFORT.read_text()
    body = src.split("def _within_bound")[1].split("def gather_dynamics")[0]
    assert "torch.all(metric_within_bound, axis=-1)" in body
    assert "metric_within_bound.mean" not in body
    assert "metric_within_bound.any" not in body


def test_one_violation_fails_the_sample():
    lo, hi = -4.05, 2.40
    metric = np.array([0.0, 0.0, 2.40, 0.0])  # exclusive: 2.40 is out
    inside = (metric > lo) & (metric < hi)
    assert inside.tolist() == [True, True, False, True]
    assert bool(inside.all()) is False
    assert float(inside.mean()) == 0.75
