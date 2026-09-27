"""Euler integration evaluates the vector field at the *interval start* t.

Weight-free. Complementary to open #44 (``linspace(0, 1, N+1)``) and #28
(Euler N+1 trace). Each step uses ``t_start = time_steps[i]`` then
``x = x + dt * v``. A midpoint / Heun step would query ``t_i + dt/2``
and change every sampled action for the same ``num_inference_steps``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_FM = _REPO / "src" / "alpamayo_r1" / "diffusion" / "flow_matching.py"


def test_source_euler_queries_interval_start_not_midpoint():
    src = _FM.read_text()
    body = src.split("def _euler")[1].split("def construct_training_data")[0]
    assert "t_start = time_steps[i].view" in body
    assert "v = step_fn(x=x, t=t_start)" in body
    assert "x = x + dt * v" in body
    assert "time_steps[i] + dt" not in body
    assert "0.5" not in body


def test_interval_start_differs_from_midpoint():
    n = 10
    time_steps = np.linspace(0.0, 1.0, n + 1)
    starts = time_steps[:-1]
    mids = time_steps[:-1] + 0.5 * np.diff(time_steps)
    assert starts[0] == 0.0
    assert not np.allclose(starts, mids)
    assert starts[0] != mids[0]
