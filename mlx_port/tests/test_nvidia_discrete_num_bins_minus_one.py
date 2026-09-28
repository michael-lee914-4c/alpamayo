"""Discrete traj encode/decode scales by ``num_bins - 1``, not ``num_bins``.

Weight-free. Complementary to open #42 (``<i0>``…``<i{V-1}>``) and
discrete clamp. Encode is
``round((x - min) / (max - min) * (num_bins - 1)).clamp(0, num_bins - 1)``
and decode divides by the same ``num_bins - 1``. Using ``num_bins``
makes the last bin unreachable and stretches decoded actions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DISC = _REPO / "src" / "alpamayo_r1" / "action_space" / "discrete_action_space.py"


def test_source_scales_by_num_bins_minus_one():
    src = _DISC.read_text()
    encode = src.split("def encode(")[1].split("def decode(")[0]
    decode = src.split("def decode(")[1]
    assert "action = (action * (self.num_bins - 1)).round().long()" in encode
    assert "action = action.clamp(0, self.num_bins - 1)" in encode
    assert "action = action / (self.num_bins - 1)" in decode
    assert "self.num_bins)" not in encode.split("clamp")[0]


def test_max_value_lands_on_last_bin():
    num_bins = 5
    lo, hi = 0.0, 1.0
    x = np.array([1.0])
    token = np.rint((x - lo) / (hi - lo) * (num_bins - 1)).astype(np.int64)
    token = np.clip(token, 0, num_bins - 1)
    assert token.tolist() == [4]
    recovered = token / (num_bins - 1) * (hi - lo) + lo
    assert recovered.tolist() == [1.0]
    wrong = np.rint((x - lo) / (hi - lo) * num_bins).astype(np.int64)
    assert wrong.tolist() == [5]
