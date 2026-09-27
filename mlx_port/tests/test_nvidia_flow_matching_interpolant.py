"""Flow-matching interpolant is ``t*x + (1-t)*noise``; loss target is ``x-noise``.

Weight-free. Complementary to open #46 (guidance knobs unused +
``is_drop_guidance: None``) and #28 (beta ``t=0.999*(1-u)``). At t=0 the
training point is pure noise; at t=1 it is data. The MSE target is the
constant velocity ``x - noise``, *not* ``x - noisy_x`` and not a
t-dependent residual. Flipping the mix or the target would train the
expert on the wrong vector field while eval Euler still integrates
``x += dt * v``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_FM = _REPO / "src" / "alpamayo_r1" / "diffusion" / "flow_matching.py"


def _noisy_x(x: np.ndarray, noise: np.ndarray, t: np.ndarray) -> np.ndarray:
    return t * x + (1.0 - t) * noise


def test_source_mixes_data_at_t1_and_noise_at_t0():
    src = _FM.read_text()
    body = src.split("def construct_training_data")[1].split("def compute_loss_from_pred")[0]
    assert "noisy_x = t * x + (1 - t) * noise" in body
    assert "t * noise + (1 - t) * x" not in body


def test_source_loss_target_is_x_minus_noise():
    src = _FM.read_text()
    body = src.split("def compute_loss_from_pred")[1]
    assert "target = (x - noise).to(dtype=pred.dtype)" in body
    assert "target = x - training_data" not in body
    assert "timesteps" not in body


def test_endpoints_and_target_are_t_independent():
    x = np.array([4.0])
    noise = np.array([1.0])
    assert _noisy_x(x, noise, np.array([0.0])) == noise
    assert _noisy_x(x, noise, np.array([1.0])) == x
    target = x - noise
    assert target == np.array([3.0])
    # x - noisy_x at t=0.5 is 1.5, not the FM velocity 3.0
    mid = _noisy_x(x, noise, np.array([0.5]))
    assert (x - mid) != target
