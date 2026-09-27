"""Train and infer ``pop`` ``input_ids`` out of ``tokenized_data``.

Weight-free. Complementary to open #38 (SFT ``torch.where(labels_mask)``)
and #20 (mixed collate). Both ``TrainableAlpamayoR1.forward`` and
``sample_trajectories_from_data_with_vlm_rollout`` mutate the batch dict
in place. A second forward / a caller that reuses the same
``tokenized_data`` would ``KeyError`` after the first pop; copying the
dict first is *not* the NVIDIA contract.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
_INFER = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


def test_source_sft_forward_pops_input_ids():
    src = _SFT.read_text()
    body = src.split("def forward")[1].split("def sample_trajectories_from_data")[0]
    assert 'input_ids = tokenized_data.pop("input_ids")' in body
    assert 'tokenized_data["input_ids"]' not in body
    assert "**tokenized_data" in body


def test_source_infer_rollout_pops_input_ids():
    src = _INFER.read_text()
    body = src.split("def sample_trajectories_from_data_with_vlm_rollout")[1]
    assert 'input_ids = tokenized_data.pop("input_ids")' in body
    assert 'tokenized_data["input_ids"]' not in body
    assert "**tokenized_data" in body


def test_pop_removes_the_key_for_a_second_caller():
    tokenized_data = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    popped = tokenized_data.pop("input_ids")
    assert popped == [1, 2, 3]
    assert "input_ids" not in tokenized_data
    assert tokenized_data == {"attention_mask": [1, 1, 1]}
