"""Stage-2 VLM return-sequences vs diffusion batch when sets > 1.

Weight-free. Complementary to open #32 (Stage-1 ``samples * sets``) and
#30 (infer ``num_return_sequences = num_traj_samples``). Stage-2 VLM
rollout keeps ``num_return_sequences = num_traj_samples`` while diffusion
samples ``B * samples * sets``. Extra text is then reshaped with the
original batch ``B``, not ``b_star``. Changing either side alone breaks
set>1 inference or CoC extract.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ALPAMAYO_R1 = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_RL_BASE = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "base_model.py"


def test_stage2_vlm_return_sequences_is_samples_not_sets():
    src = _ALPAMAYO_R1.read_text()
    assert "n_samples_total = num_traj_samples * num_traj_sets" in src
    assert "generation_config.num_return_sequences = num_traj_samples" in src
    assert "generation_config.num_return_sequences = n_samples_total" not in src
    assert "total_batch = B * n_samples_total" in src
    assert 'einops.rearrange(\n            pred_xyz, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples' in src


def test_stage1_and_rl_use_samples_times_sets():
    s1 = _SFT_BASE.read_text()
    rl = _RL_BASE.read_text()
    assert "generation_config.num_return_sequences = n_samples_total" in s1
    assert "generation_config.num_return_sequences = n_samples_total" in rl


def test_sets_gt_1_vlm_count_does_not_match_extra_reshape():
    """VLM emits B*samples sequences; extra reshape wants B*sets*samples."""
    src = _ALPAMAYO_R1.read_text()
    assert "b_star = vlm_outputs.sequences.shape[0]" in src
    assert 'extra[text_tokens] = np.array(extra[text_tokens]).reshape(\n                    [input_ids.shape[0], num_traj_sets, num_traj_samples]' in src

    B, samples, sets = 2, 6, 2
    vlm_seq_count = B * samples
    extra_reshape_count = B * sets * samples
    diffusion_batch = B * samples * sets
    assert vlm_seq_count == 12
    assert extra_reshape_count == 24
    assert diffusion_batch == 24
    assert vlm_seq_count != extra_reshape_count


def test_stage2_extra_reshape_uses_original_b_not_bstar():
    src = _ALPAMAYO_R1.read_text()
    # input_ids is popped from the original batch before generate, so
    # input_ids.shape[0] is B, not B*num_traj_samples.
    assert "input_ids = tokenized_data.pop(\"input_ids\")" in src
    assert "[input_ids.shape[0], num_traj_sets, num_traj_samples]" in src
    assert "reshape(\n                    [b_star," not in src
