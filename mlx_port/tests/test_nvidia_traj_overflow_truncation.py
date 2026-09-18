"""``extract_traj_tokens`` keeps the first N in-span bins and drops the tail.

Weight-free. Complementary to open #22 (last start / empty→zeros / clamp)
and #33 (missing-start scoops the prefix). A longer CoC+traj span is not
an error: ``cumsum < tokens_per_future_traj`` silently truncates, so a
first-vs-last swap would decode a different path without failing.
Invalid ids use ``> vocab_size`` (not ``>=``) then clamp to ``vocab-1``.
"""

from __future__ import annotations

from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def _extract_traj_tokens(
    output_tokens,
    special_token_ids,
    tokens_per_future_traj,
    future_token_start_idx,
    traj_tokenizer_vocab_size,
):
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
        token_values = extracted_tokens[output_mask] - future_token_start_idx
        token_values = torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)
        traj_tokens[batch_ids, output_positions] = token_values
    return traj_tokens


def test_source_caps_with_cumsum_and_gt_vocab_check():
    src = _TOKEN_UTILS.read_text()
    assert "output_mask = valid_mask & (cumsum_indices < tokens_per_future_traj)" in src
    assert "token_values = token_values - future_token_start_idx" in src
    assert (
        "invalid_tokens = (token_values < 0) | (token_values > traj_tokenizer_vocab_size)" in src
    )
    assert "torch.clamp(token_values, min=0, max=traj_tokenizer_vocab_size - 1)" in src


def test_overflow_keeps_first_n_not_last_n():
    start, end, offset = 7, 8, 100
    # Five in-span bins; only the first three must survive.
    ids = torch.tensor([[0, start, 100, 101, 102, 103, 104, end]], dtype=torch.long)
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=3,
        future_token_start_idx=offset,
        traj_tokenizer_vocab_size=20,
    )
    assert got.tolist() == [[0, 1, 2]]


def test_exact_span_is_not_truncated():
    start, end, offset = 7, 8, 50
    ids = torch.tensor([[start, 50, 51, 52, end]], dtype=torch.long)
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=3,
        future_token_start_idx=offset,
        traj_tokenizer_vocab_size=10,
    )
    assert got.tolist() == [[0, 1, 2]]


def test_oob_ids_clamp_to_vocab_and_zero():
    start, end, offset, vocab = 7, 8, 100, 10
    # 99-100 = -1 → 0; 120-100 = 20 > vocab → clamp 9. vocab_size itself
    # is not in the ``> vocab`` warning predicate but still clamps to 9.
    ids = torch.tensor([[start, 99, 110, 120, end]], dtype=torch.long)
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=3,
        future_token_start_idx=offset,
        traj_tokenizer_vocab_size=vocab,
    )
    assert got.tolist() == [[0, 9, 9]]
