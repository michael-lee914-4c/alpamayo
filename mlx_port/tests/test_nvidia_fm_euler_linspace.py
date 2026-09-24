"""Flow-matching Euler uses ``linspace(0, 1, inference_step + 1)``.

Weight-free. Complementary to open #28 (Euler N+1 trace) and #29
(uniform sampler / invalid int-method). The integrator takes N steps
across N+1 knots so the last ``dt`` lands on t=1. Using ``inference_step``
knots would stop at ``(N-1)/N`` and leave the expert one Euler step short.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_FM = _REPO / "src" / "alpamayo_r1" / "diffusion" / "flow_matching.py"


def _euler(x, inference_step, step_fn):
    torch = pytest.importorskip("torch")
    time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=x.device)
    n_dim = x.ndim - 1
    all_steps = [x]
    for i in range(inference_step):
        dt = time_steps[i + 1] - time_steps[i]
        dt = dt.view(1, *[1] * n_dim).expand_as(x)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand_as(x)
        v = step_fn(x=x, t=t_start)
        x = x + dt * v
        all_steps.append(x)
    return torch.stack(all_steps, dim=1), time_steps


def test_source_linspace_includes_endpoint():
    src = _FM.read_text()
    assert "time_steps = torch.linspace(0.0, 1.0, inference_step + 1, device=device)" in src
    assert "for i in range(inference_step):" in src
    assert "dt = time_steps[i + 1] - time_steps[i]" in src
    assert "x = x + dt * v" in src


def test_n_steps_produce_n_plus_one_knots_ending_at_one():
    torch = pytest.importorskip("torch")
    x0 = torch.zeros(2, 3)
    steps, times = _euler(x0, inference_step=4, step_fn=lambda x, t: torch.ones_like(x))
    assert steps.shape == (2, 5, 3)
    assert times.shape == (4 + 1,)
    torch.testing.assert_close(times[0], torch.tensor(0.0))
    torch.testing.assert_close(times[-1], torch.tensor(1.0))
    # Constant v=1 over dt=0.25 → x ends at 1.0 after 4 steps.
    torch.testing.assert_close(steps[:, -1], torch.ones(2, 3))
    torch.testing.assert_close(times[1] - times[0], torch.tensor(0.25))
