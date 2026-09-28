"""Comfort yaw wrap happens on the raw Δθ, then ``* PLANNING_FREQ``.

Weight-free. Complementary to open #27 (hold-last × 10 Hz) and #45
(comfort uses ``R[1, 0]``). ``_diff_yaw`` wraps at ±π *before*
scaling by 10 Hz. Wrapping the already-scaled rate would treat a 2π
jump as 20π rad/s and fail every yaw-rate gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"


def test_source_wraps_then_scales():
    src = _COMFORT.read_text()
    body = src.split("def _diff_yaw")[1].split("def _within_bound")[0]
    assert "yaw_diff = torch.diff(yaw, dim=-1)" in body
    assert "yaw_diff > torch.pi" in body
    assert "yaw_diff - 2 * torch.pi" in body
    assert "yaw_diff < -torch.pi" in body
    assert "yaw_diff + 2 * torch.pi" in body
    assert "yaw_rate = yaw_diff * PLANNING_FREQ" in body
    wrap_at = body.index("yaw_diff > torch.pi")
    scale_at = body.index("yaw_rate = yaw_diff * PLANNING_FREQ")
    assert wrap_at < scale_at


def test_twopi_jump_is_zero_rate_after_wrap():
    # headings [0, 2π-ε] look like a huge positive jump before wrap.
    eps = 1e-6
    d = (2 * np.pi - eps) - 0.0
    assert d > np.pi
    wrapped = d - 2 * np.pi
    rate = wrapped * 10.0
    assert abs(rate) < 1e-4
    unwrapped_rate = d * 10.0
    assert unwrapped_rate > 60.0
