"""RL ``_prepare_policy_sample`` accepts ``[L]`` / ``[1, L]`` and rejects B>1.

Weight-free. Complementary to open #32 (rank-3 ego unsqueeze to
``(B, 1, T, …)``). Policy ``input_ids`` must be a single row so later
``get_policy_input`` can append the rollout + ``traj_future_end``. A
``[B, L]`` tensor with B≠1 would concatenate the wrong axis and label
every prompt token.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "finetune" / "rl" / "base_data_packer.py"
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"


def _prepare_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Replica of NVIDIA ``BaseRLDataPacker._prepare_policy_sample`` shape guards."""
    if input_ids.ndim == 1:
        return input_ids.unsqueeze(0)
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be shape [1, L] or [L], got {tuple(input_ids.shape)}"
            )
        return input_ids
    raise ValueError(f"input_ids must be shape [1, L] or [L], got {tuple(input_ids.shape)}")


def test_source_rejects_batch_greater_than_one():
    src = _BASE.read_text()
    assert "if input_ids.ndim == 1:" in src
    assert "input_ids = input_ids.unsqueeze(0)" in src
    assert "if input_ids.shape[0] != 1:" in src
    assert "input_ids must be shape [1, L] or [L]" in src
    packer = _PACKER.read_text()
    assert "tokenized_data.input_ids must be a torch.Tensor of shape [1, L]" in packer
    assert "n_ignore_prefix_tokens" in packer
    # Accepted on the signature, never applied to the mask / ids.
    assert "n_ignore_prefix_tokens:" in packer
    assert "n_ignore_prefix_tokens +" not in packer
    assert "n_ignore_prefix_tokens]" not in packer


def test_rank1_promotes_to_single_row():
    got = _prepare_input_ids(torch.arange(5, dtype=torch.long))
    assert tuple(got.shape) == (1, 5)
    assert torch.equal(got[0], torch.arange(5))


def test_single_row_is_unchanged():
    ids = torch.arange(4, dtype=torch.long).unsqueeze(0)
    got = _prepare_input_ids(ids)
    assert got is ids


def test_batch_greater_than_one_raises():
    with pytest.raises(ValueError, match=r"\[1, L\]"):
        _prepare_input_ids(torch.zeros(2, 4, dtype=torch.long))


def test_rank3_raises():
    with pytest.raises(ValueError, match=r"\[1, L\]"):
        _prepare_input_ids(torch.zeros(1, 1, 4, dtype=torch.long))
