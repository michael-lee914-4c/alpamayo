"""``FourierEncoderV2`` logspace starts at ``10**0 = 1``, not 0.

Weight-free. Complementary to open #29 (``[sin, cos] * √2``) and #47
(odd ``dim`` drops a channel). Frequencies are
``logspace(0, log10(max_freq), steps=half)``, so the first bin is 1 Hz
and the last is ``max_freq``. ``logspace(-inf, …)`` or a linear
``linspace(0, max_freq)`` would put a DC bin in the action-in features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_ACTION_IN = _REPO / "src" / "alpamayo_r1" / "models" / "action_in_proj.py"


def test_source_logspace_starts_at_decade_zero():
    src = _ACTION_IN.read_text()
    body = src.split("class FourierEncoderV2")[1].split("class PerWaypointActionInProjV2")[0]
    assert "freqs = torch.logspace(0, math.log10(max_freq), steps=half)" in body
    assert "torch.linspace" not in body
    assert "logspace(-" not in body


def test_first_and_last_frequencies():
    half = 4
    max_freq = 100.0
    freqs = np.logspace(0, np.log10(max_freq), num=half)
    assert freqs[0] == 1.0
    assert np.isclose(freqs[-1], 100.0)
    linear = np.linspace(0.0, max_freq, num=half)
    assert linear[0] == 0.0
    assert not np.isclose(linear[1], freqs[1])
