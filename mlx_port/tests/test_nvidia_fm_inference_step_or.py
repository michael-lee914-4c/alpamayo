"""``FlowMatching.sample`` treats ``inference_step=0`` as missing.

Weight-free. Complementary to open #44 (``linspace(0, 1, N+1)``) and #47
(Euler evaluates at interval-start ``t``). The override is
``inference_step = inference_step or self.num_inference_steps``, so a
coerced ``0`` silently runs the default step count instead of a no-op
or a one-step sample. Same ``or`` swallows a falsy ``int_method``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_FM = _REPO / "src" / "alpamayo_r1" / "diffusion" / "flow_matching.py"


def test_source_uses_or_for_inference_step_and_int_method():
    src = _FM.read_text()
    body = src.split("def sample(")[1].split("def _euler(")[0]
    assert "inference_step = inference_step or self.num_inference_steps" in body
    assert "int_method = int_method or self.int_method" in body
    assert "if inference_step is None" not in body


def test_zero_is_falsy_so_default_steps_win():
    default = 10
    override = 0
    n_steps = override or default
    assert n_steps == default
    assert (1 or default) == 1
