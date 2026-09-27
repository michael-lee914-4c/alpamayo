"""``FourierEncoderV2`` uses ``dim // 2`` frequencies, so an odd ``dim`` drops a channel.

Weight-free. Complementary to open #29 (logspace ``[sin, cos] * √2``).
``half = dim // 2`` then ``cat([sin, cos], -1)`` always emits ``2 * half``
features. ``dim=5`` therefore stores a 4-wide buffer while
``self.out_dim`` still claims 5, and the MLP ``num_input_feats`` sum
drifts vs the actual concat. Even dims are the only safe contract.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_PROJ = _REPO / "src" / "alpamayo_r1" / "models" / "action_in_proj.py"


def test_source_uses_integer_half_and_sin_cos_cat():
    src = _PROJ.read_text()
    body = src.split("class FourierEncoderV2")[1].split("class PerWaypointActionInProjV2")[0]
    assert "half = dim // 2" in body
    assert "freqs = torch.logspace(0, math.log10(max_freq), steps=half)" in body
    assert "self.out_dim = dim" in body
    assert "torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)" in body


def test_odd_dim_emits_even_width():
    dim = 5
    half = dim // 2
    assert half == 2
    emitted = 2 * half
    assert emitted == 4
    assert emitted != dim
    # even dim is lossless
    assert 2 * (20 // 2) == 20


def test_logspace_length_is_half_not_dim():
    max_freq = 100.0
    half = 3
    freqs = np.logspace(0, math.log10(max_freq), num=half)
    assert freqs.shape == (half,)
    assert freqs[0] == 1.0
    np.testing.assert_allclose(freqs[-1], max_freq)
