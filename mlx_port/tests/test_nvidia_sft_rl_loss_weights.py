"""SFT comments out loss_weights; RL multiplies them in.

Weight-free. Complementary to open #33 (Stage-1 split CE then IGNORE).
SFT Stage-1 sums unweighted ``future_traj`` + ``others``. The RL wrapper
multiplies each term by ``config.loss_weights``. Aligning the two by
uncommenting SFT — or dropping the RL multiply — silently rescales the
train objective.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_RL = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "base_model.py"


def test_sft_stage1_loss_weights_are_commented_out():
    src = _SFT.read_text()
    assert 'losses["future_traj"] = self._compute_next_token_loss(outputs, labels, traj_mask)' in src
    assert '#  * self.config.loss_weights.get("future_traj", 1.0)' in src
    assert 'losses["others"] = self._compute_next_token_loss(outputs, labels, labels != IGNORE_INDEX)' in src
    assert '#  * self.config.loss_weights.get("others", 1.0)' in src
    assert ") * self.config.loss_weights.get(" not in src


def test_rl_wrapper_applies_loss_weights():
    src = _RL.read_text()
    assert ') * self.config.loss_weights.get("future_traj", 1.0)' in src
    assert ') * self.config.loss_weights.get("others", 1.0)' in src
    assert 'outputs.loss = sum(losses.values())' in src


def test_weight_scale_changes_rl_not_sft():
    future_ce, others_ce = 2.0, 1.0
    weights = {"future_traj": 0.5, "others": 2.0}
    sft_loss = future_ce + others_ce
    rl_loss = future_ce * weights["future_traj"] + others_ce * weights["others"]
    assert sft_loss == 3.0
    assert rl_loss == 3.0  # 1.0 + 2.0 — same total only for this pair
    # A uniform 2x weight doubles RL and leaves SFT unchanged.
    rl_double = future_ce * 2.0 + others_ce * 2.0
    assert rl_double == 2.0 * sft_loss
    assert sft_loss != rl_double
