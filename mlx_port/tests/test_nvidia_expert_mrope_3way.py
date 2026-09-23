"""Stage-2 expert RoPE is a 3-way Qwen mRoPE index plus VLM delta.

Weight-free. Complementary to open #30 (train ``rope+kv_len`` vs infer
first-start+1) and #37 (stale mRoPE cache). ``einops.repeat(..., "l -> 3 b l")``
is the Qwen 2.5-VL layout. A rank-2 ``[B, L]`` index would broadcast into
the wrong axis and rotate expert tokens as if they were vision patches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"


def _process_position_ids_qwen2_5_vl(rope_deltas, kv_len, batch_size, num_expert_tokens, device="cpu"):
    torch = pytest.importorskip("torch")
    einops = pytest.importorskip("einops")
    position_ids = torch.arange(num_expert_tokens, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()
    delta = rope_deltas + kv_len
    position_ids += delta.to(position_ids.device)
    return position_ids


def test_source_repeats_three_axes_then_adds_delta_plus_kv():
    src = _S2.read_text()
    assert 'position_ids = einops.repeat(position_ids, "l -> 3 b l", b=batch_size).clone()' in src
    assert "delta = vlm_outputs.rope_deltas + vlm_outputs.past_key_values.get_seq_length()" in src
    assert "position_ids += delta.to(position_ids.device)" in src
    assert "expert_embeds = action_embeds" in src


def test_shape_is_three_by_batch_by_expert_len():
    torch = pytest.importorskip("torch")
    rope_deltas = torch.tensor([[10], [20]])
    ids = _process_position_ids_qwen2_5_vl(
        rope_deltas, kv_len=5, batch_size=2, num_expert_tokens=4
    )
    assert ids.shape == (3, 2, 4)
    # Local expert steps stay 0..L-1; the VLM prefix is added as a scalar per row.
    torch.testing.assert_close(ids[0, 0], torch.tensor([15, 16, 17, 18]))
    torch.testing.assert_close(ids[0, 1], torch.tensor([25, 26, 27, 28]))
    # All three mRoPE axes share the same 1D offset (no vision tokens after start).
    torch.testing.assert_close(ids[0], ids[1])
    torch.testing.assert_close(ids[1], ids[2])


def test_rank2_repeat_is_not_the_qwen_layout():
    torch = pytest.importorskip("torch")
    einops = pytest.importorskip("einops")
    local = torch.arange(4)
    rank2 = einops.repeat(local, "l -> b l", b=2)
    rank3 = einops.repeat(local, "l -> 3 b l", b=2)
    assert rank2.shape == (2, 4)
    assert rank3.shape == (3, 2, 4)
    assert rank2.shape != rank3.shape
