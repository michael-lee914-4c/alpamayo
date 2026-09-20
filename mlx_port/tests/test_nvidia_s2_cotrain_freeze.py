"""Stage-2 ``cotrain_vlm=False`` freezes the VLM and drops its CE.

Weight-free. Complementary to open #36 (``expert_embeds = action_embeds``)
and #20 (MLX freeze/unfreeze). NVIDIA ``TrainableAlpamayoR1`` freezes
every VLM param, runs the tower under ``no_grad``, and only adds
``vlm_outputs.loss`` when ``cotrain_vlm`` is True. Flipping that flag
would backprop through the 8B tower or drop CoC CE while co-training.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SFT_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"


def test_source_freezes_vlm_and_gates_ce_on_cotrain_flag():
    src = _SFT_S2.read_text()
    assert "if not self.cotrain_vlm:" in src
    assert "for param in self.vlm.parameters():" in src
    assert "param.requires_grad = False" in src
    assert "context = torch.no_grad()" in src
    assert "context = nullcontext()" in src
    assert "loss = future_traj_loss" in src
    assert "if self.cotrain_vlm:" in src
    assert "loss += vlm_outputs.loss" in src
    assert "expert_embeds = action_embeds" in src


def _stage2_loss(cotrain_vlm: bool, future_traj_loss: float, vlm_loss: float) -> float:
    """Replica of NVIDIA Stage-2 loss composition."""
    loss = future_traj_loss
    if cotrain_vlm:
        loss += vlm_loss
    return loss


def test_expert_only_loss_ignores_vlm_ce():
    assert _stage2_loss(False, future_traj_loss=0.4, vlm_loss=9.0) == pytest.approx(0.4)


def test_cotrain_loss_adds_vlm_ce():
    assert _stage2_loss(True, future_traj_loss=0.4, vlm_loss=9.0) == pytest.approx(9.4)


def test_freeze_loop_clears_requires_grad():
    class _P:
        def __init__(self):
            self.requires_grad = True

    class _VLM:
        def parameters(self):
            return [self.a, self.b]

        def __init__(self):
            self.a = _P()
            self.b = _P()

    vlm = _VLM()
    cotrain_vlm = False
    if not cotrain_vlm:
        for param in vlm.parameters():
            param.requires_grad = False
    assert vlm.a.requires_grad is False
    assert vlm.b.requires_grad is False
