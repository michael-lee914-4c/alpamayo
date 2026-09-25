"""GRPO advantages expand to a padded ``computed_max_len``; collate does not.

Weight-free. Complementary to open #36 (policy ids stay ``[L]`` / ``[1, L]``)
and #35 (skip when ≤1 non-empty completion). The trainer rounds
``policy_compute_max_len`` up to ``seq_len_multiple`` and expands advantages
to that width, but ``policy_collate_fn`` never reads ``computed_max_len`` —
it ``torch.cat``s the real tokenized rows. A later pad-to-max change that
forgets the advantage tensor (or the reverse) mis-weights GRPO tokens.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACK = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"
_TRAIN = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "trainer.py"


def _round_up(length: int, multiple: int) -> int:
    return (length + multiple - 1) // multiple * multiple


def test_source_trainer_pads_advantages_to_rounded_max_len():
    src = _TRAIN.read_text()
    assert "self.data_packer.policy_compute_max_len(" in src
    assert "(computed_max_len + self.seq_len_multiple - 1)" in src
    assert "// self.seq_len_multiple" in src
    assert "* self.seq_len_multiple" in src
    assert ".unsqueeze(1)" in src
    assert ".expand(-1, computed_max_len)" in src
    assert "current_advantages = logprob_masks * minibatched_advantages" in src


def test_source_collate_never_uses_computed_max_len():
    src = _PACK.read_text()
    assert "def policy_collate_fn(" in src
    assert "computed_max_len: int," in src
    # Parameter is accepted for the Cosmos-RL signature but unused in the body.
    body = src.split("def policy_collate_fn(", 1)[1]
    body = body.split("\n    def ", 1)[0]
    assert body.count("computed_max_len") == 1
    assert 'torch.cat([row[k] for row in batch["tokenized_data"]])' in src


def test_round_up_leaves_true_max_shorter_than_advantage_width():
    torch = pytest.importorskip("torch")
    lengths = [100, 130]
    computed = max(lengths)
    rounded = _round_up(computed, 64)
    assert computed == 130
    assert rounded == 192
    advantages = torch.tensor([1.5, -0.5]).unsqueeze(1).expand(-1, rounded)
    # Collate keeps the true max length (no pad to seq_len_multiple).
    logprob_masks = torch.zeros(2, computed, dtype=torch.bool)
    assert advantages.shape == (2, 192)
    assert logprob_masks.shape == (2, 130)
    assert advantages.shape[1] != logprob_masks.shape[1]
