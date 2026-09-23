"""``extract_traj_tokens`` subtracts the vocab offset then clamps IDs.

Weight-free. Complementary to open #35 (count overflow keeps first N via
``cumsum < tokens_per_future_traj``) and #33 (missing-start ``-1`` scoop).
A token below ``future_token_start_idx`` or above the traj vocab must not
survive as a raw VLM id — decode would treat it as a huge accel bin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOK = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def _extract_traj_tokens(
    output_tokens,
    special_token_ids,
    tokens_per_future_traj,
    future_token_start_idx,
    traj_tokenizer_vocab_size,
):
    torch = pytest.importorskip("torch")
    batch_size, seq_len = output_tokens.shape
    device = output_tokens.device
    traj_tokens = torch.zeros(
        (batch_size, tokens_per_future_traj), dtype=output_tokens.dtype, device=device
    )
    end_mask = output_tokens == special_token_ids["traj_future_end"]
    end_positions = torch.where(
        end_mask.any(dim=1),
        end_mask.int().argmax(dim=1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=device),
    )
    start_mask = output_tokens == special_token_ids["traj_future_start"]
    start_mask_reversed = torch.flip(start_mask, dims=[1])
    last_start_positions_reversed = start_mask_reversed.int().argmax(dim=1)
    start_positions = seq_len - 1 - last_start_positions_reversed
    start_positions = torch.where(
        start_mask.any(dim=1),
        start_positions,
        torch.full((batch_size,), -1, dtype=torch.long, device=device),
    )
    range_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    valid_mask = (range_tensor > start_positions.unsqueeze(1)) & (
        range_tensor < end_positions.unsqueeze(1)
    )
    extracted_tokens = torch.where(valid_mask, output_tokens, torch.zeros_like(output_tokens))
    cumsum_indices = torch.cumsum(valid_mask.int(), dim=1) - 1
    output_mask = valid_mask & (cumsum_indices < tokens_per_future_traj)
    if output_mask.any():
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        output_positions = cumsum_indices[output_mask]
        batch_ids = batch_indices[output_mask]
        token_values = extracted_tokens[output_mask]
        token_values = token_values - future_token_start_idx
        token_values = torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)
        traj_tokens[batch_ids, output_positions] = token_values
    return traj_tokens


def test_source_subtracts_offset_and_clamps_to_vocab():
    src = _TOK.read_text()
    assert "token_values = token_values - future_token_start_idx" in src
    assert "invalid_tokens = (token_values < 0) | (token_values > traj_tokenizer_vocab_size)" in src
    assert "token_values = torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)" in src
    # Warn uses exclusive ``> vocab``; clamp uses inclusive ``vocab-1``.
    assert "token_values > traj_tokenizer_vocab_size" in src


def test_below_offset_and_above_vocab_clamp():
    torch = pytest.importorskip("torch")
    start, end = 9, 10
    # offset=100, vocab=10 → raw 95→-5→0, raw 115→15→9, raw 103→3.
    ids = torch.tensor([[start, 95, 103, 115, end]])
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=3,
        future_token_start_idx=100,
        traj_tokenizer_vocab_size=10,
    )
    torch.testing.assert_close(got, torch.tensor([[0, 3, 9]]))


def test_token_equal_to_vocab_size_is_clamped_silently():
    torch = pytest.importorskip("torch")
    start, end = 1, 2
    # raw 110 - 100 = 10, which is NOT ``> vocab_size`` (10) so no warn,
    # but clamp(max=9) still rewrites it.
    ids = torch.tensor([[start, 110, end]])
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=1,
        future_token_start_idx=100,
        traj_tokenizer_vocab_size=10,
    )
    torch.testing.assert_close(got, torch.tensor([[9]]))
