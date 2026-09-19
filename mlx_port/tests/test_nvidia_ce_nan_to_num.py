"""Stage-1 / RL CE maps an all-IGNORE remaining slice to 0 via ``nan_to_num``.

Weight-free. Complementary to open #32 (empty shift-mask returns 0 before
CE; ``~token_mask`` logits are ``-inf``). After the traj band is written
to ``IGNORE_INDEX``, the leftover ``others`` mask can still be True on
positions whose labels are already -100. Mean CE is then NaN and is
forced to 0. Dropping ``nan_to_num`` poisons the sum with NaN.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

_REPO = Path(__file__).resolve().parents[2]
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_RL = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "base_model.py"
IGNORE_INDEX = -100


def _compute_next_token_loss(outputs, labels, labels_mask=None, token_mask=None):
    """Replica of NVIDIA ``TrainableReasoningVLA._compute_next_token_loss``."""
    if labels_mask is None:
        labels_mask = torch.ones_like(labels, dtype=torch.bool)
    if labels_mask[:, 1:].sum() == 0:
        return torch.tensor(0.0, device=labels.device)
    shift_labels = labels[..., 1:]
    shift_logits = outputs.logits[..., :-1, :].clone()
    shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
    shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
    if token_mask is not None:
        shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
    return torch.nan_to_num(
        torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
        ),
        nan=0.0,
    )


def test_source_uses_nan_to_num_after_mean_ce():
    for path in (_SFT, _RL):
        src = path.read_text()
        assert "if labels_mask[:, 1:].sum() == 0:" in src
        assert "return torch.tensor(0.0, device=labels.device)" in src
        assert "torch.nan_to_num(" in src
        assert "nan=0.0" in src
        assert "ignore_index=IGNORE_INDEX" in src
        assert "reduction=\"mean\"" in src


def test_empty_shift_mask_returns_zero_without_ce():
    logits = torch.zeros(1, 4, 8)
    labels = torch.tensor([[1, 2, 3, 4]])
    mask = torch.zeros_like(labels, dtype=torch.bool)
    got = _compute_next_token_loss(SimpleNamespace(logits=logits), labels, mask)
    assert float(got) == 0.0


def test_all_ignore_remaining_labels_become_zero_not_nan():
    """Mask is non-empty but every selected label is IGNORE_INDEX → NaN CE → 0."""
    logits = torch.zeros(1, 4, 8)
    labels = torch.full((1, 4), IGNORE_INDEX)
    mask = torch.ones_like(labels, dtype=torch.bool)
    raw = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 8),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="mean",
    )
    assert torch.isnan(raw)
    got = _compute_next_token_loss(SimpleNamespace(logits=logits), labels, mask)
    assert not torch.isnan(got)
    assert float(got) == 0.0
