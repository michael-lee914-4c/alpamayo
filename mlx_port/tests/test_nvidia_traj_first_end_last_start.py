"""``extract_traj_tokens`` uses the first end marker and the last start.

Weight-free. Complementary to open #42 (offset-then-clamp), #35 (count
overflow keeps first N), and #33 (missing-start ``-1`` scoops the prefix).
A spurious early ``<|traj_future_end|>`` collapses the window because
``argmax`` is first-True; a second start after that end yields zeros, not
the later real span. Decode then treats empty bins as origin.
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


def test_source_uses_first_end_and_last_start():
    src = _TOK.read_text()
    assert "end_mask.int().argmax(dim=1)" in src
    assert "start_mask_reversed = torch.flip(start_mask, dims=[1])" in src
    assert "start_positions = seq_len - 1 - last_start_positions_reversed" in src
    assert "range_tensor > start_positions.unsqueeze(1)" in src
    assert "range_tensor < end_positions.unsqueeze(1)" in src


def test_last_start_before_first_end_keeps_inner_span():
    torch = pytest.importorskip("torch")
    start, end = 9, 10
    # Two starts; last start still precedes the (only) end.
    ids = torch.tensor([[start, 101, start, 102, 103, end]])
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=2,
        future_token_start_idx=100,
        traj_tokenizer_vocab_size=10,
    )
    torch.testing.assert_close(got, torch.tensor([[2, 3]]))


def test_early_end_before_last_start_collapses_to_zeros():
    torch = pytest.importorskip("torch")
    start, end = 9, 10
    # Decoy end sits between the two starts, so first-end < last-start.
    ids = torch.tensor([[start, 101, end, start, 102, 103, end]])
    got = _extract_traj_tokens(
        ids,
        {"traj_future_start": start, "traj_future_end": end},
        tokens_per_future_traj=2,
        future_token_start_idx=100,
        traj_tokenizer_vocab_size=10,
    )
    torch.testing.assert_close(got, torch.zeros(1, 2, dtype=ids.dtype))
