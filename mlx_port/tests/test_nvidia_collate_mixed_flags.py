"""Qwen/RL collate reject mixed ``generation_mode`` / ``label_components``.

Weight-free. Complementary to open #32 (train assistant-EOS OR + gen-mode
all-False labels). A batch that mixes train and gen rows, or mixes
``traj_future`` vs ``cot`` label sets, would OR the wrong spans or zero
the CE mask for only some rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"
_RL_PACK = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"


def _assert_same(values, name: str):
    """Replica of NVIDIA collate same-flag asserts."""
    first = values[0]
    assert all(item == first for item in values), (
        f"{name} is not the same for all instances in the batch."
    )
    return first


def test_source_asserts_homogeneous_label_and_gen_flags():
    qwen = _QWEN.read_text()
    assert "label_components is not the same for all instances in the batch." in qwen
    assert "generation_mode is not the same for all instances in the batch." in qwen
    assert "if not generation_mode:" in qwen
    assert "batched_data[\"labels_mask\"] |= eos_mask_assistant" in qwen
    pack = _RL_PACK.read_text()
    assert "label_components is not the same for all instances in the batch." in pack


def test_matching_flags_keep_the_first_value():
    assert _assert_same(["traj_future", "traj_future"], "label_components") == "traj_future"
    assert _assert_same([False, False], "generation_mode") is False
    assert _assert_same([True, True], "generation_mode") is True


def test_mixed_label_components_raise():
    with pytest.raises(AssertionError, match="label_components is not the same"):
        _assert_same(["traj_future", "cot"], "label_components")


def test_mixed_generation_mode_raise():
    with pytest.raises(AssertionError, match="generation_mode is not the same"):
        _assert_same([False, True], "generation_mode")
