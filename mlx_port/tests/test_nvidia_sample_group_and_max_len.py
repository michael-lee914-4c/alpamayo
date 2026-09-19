"""Infer / SFT sample paths reject ``n_traj_group != 1`` and short max-gen.

Weight-free. Complementary to open #30 (hist last-group repeat) and #34
(Stage-2 VLM ``num_return_sequences = num_traj_samples``). A loader that
keeps ``n_traj>1`` would pass the last-group squeeze tests but still
reshape ``(B*n_traj)`` into ``(B, sets, samples)``. A
``max_generation_length < tokens_per_future_traj`` would truncate the
future span before extract.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
_RL = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "base_model.py"
_INFER = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


def test_sft_and_rl_sample_require_single_traj_group():
    for path in (_SFT, _RL, _INFER):
        src = path.read_text()
        assert "B, n_traj_group, _, _ = ego_history_xyz.shape" in src
        assert 'assert n_traj_group == 1, "Only one trajectory group is supported for inference."' in src


def test_sft_and_rl_reject_max_gen_shorter_than_future():
    for path in (_SFT, _RL):
        src = path.read_text()
        assert "max_generation_length = kwargs.get(" in src
        assert "assert max_generation_length >= self.config.tokens_per_future_traj" in src
    # Stage-2 VLM rollout does not assert the future-token floor.
    s2 = _S2.read_text()
    assert "assert max_generation_length >= self.config.tokens_per_future_traj" not in s2
    infer = _INFER.read_text()
    assert "assert max_generation_length >= self.config.tokens_per_future_traj" not in infer


def test_max_gen_default_is_tokens_per_future_traj():
    for path in (_SFT, _RL, _INFER):
        src = path.read_text()
        assert '"max_generation_length", self.config.tokens_per_future_traj' in src
