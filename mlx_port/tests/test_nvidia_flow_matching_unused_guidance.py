"""Flow-matching stores guidance knobs but never reads them.

Weight-free. Complementary to open #28 (beta ``t=0.999*(1-u)``) and #29
(uniform sampler / invalid int-method). ``train_ignore_guidance_rate`` and
``inference_guidance_weight`` are assigned in ``__init__`` only;
``construct_training_data`` always writes ``is_drop_guidance: None`` and
``sample`` never scales by the weight. A config-only "guidance" toggle
would look like a train/eval fork while changing nothing.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_FM = _REPO / "src" / "alpamayo_r1" / "diffusion" / "flow_matching.py"


def _attr_uses(name: str) -> list[str]:
    tree = ast.parse(_FM.read_text())
    uses: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == name:
            uses.append(type(node.ctx).__name__)
    return uses


def test_guidance_fields_are_only_stored():
    ignore_uses = _attr_uses("train_ignore_guidance_rate")
    weight_uses = _attr_uses("inference_guidance_weight")
    assert ignore_uses == ["Store"]
    assert weight_uses == ["Store"]


def test_training_data_always_drops_guidance_as_none():
    src = _FM.read_text()
    assert '"is_drop_guidance": None' in src
    assert "is_drop_guidance" in src
    # No branch ever sets a boolean drop mask.
    assert "is_drop_guidance\": True" not in src
    assert "is_drop_guidance\": False" not in src


def test_loss_is_unweighted_mse_on_x_minus_noise():
    src = _FM.read_text()
    assert "target = (x - noise).to(dtype=pred.dtype)" in src
    assert "torch.nn.functional.mse_loss(target, pred)" in src
    assert "inference_guidance_weight" not in src.split("def compute_loss_from_pred")[1]
