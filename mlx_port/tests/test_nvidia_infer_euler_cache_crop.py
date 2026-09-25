"""Infer Euler steps crop the expert KV cache back to the prefill length.

Weight-free. Complementary to open #33 / #44 (train last-start crop vs infer
first-start offset). After each denoising step the expert appends future
tokens; ``prompt_cache.crop(prefill_seq_len)`` rewinds so the next ``t`` sees
the same prefix. Dropping the crop lets KV grow by ``n_diffusion_tokens`` per
step and shifts mRoPE / the ``offset:-n_diff`` attention blank.

The attention tensor is allocated ``float32`` zeros, then ``finfo.min`` is
written on ``[offset[i] : -n_diffusion_tokens]``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ALP = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


def test_source_crops_cache_to_prefill_after_each_expert_step():
    src = _ALP.read_text()
    assert "prefill_seq_len = prompt_cache.get_seq_length()" in src
    assert "prompt_cache.crop(prefill_seq_len)" in src
    assert "attention_mask = torch.zeros(" in src
    assert "dtype=torch.float32," in src
    assert "attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(" in src


def test_blank_slice_is_between_offset_and_future_tokens():
    torch = pytest.importorskip("torch")
    b_star = 2
    n_diff = 4
    prefill = 10
    offset = torch.tensor([3, 8])
    attention_mask = torch.zeros((b_star, 1, n_diff, prefill + n_diff), dtype=torch.float32)
    for i in range(b_star):
        attention_mask[i, :, :, offset[i] : -n_diff] = torch.finfo(attention_mask.dtype).min

    # Future-token columns stay 0 (attendable). Prefix before offset stays 0.
    assert torch.all(attention_mask[..., -n_diff:] == 0)
    assert torch.all(attention_mask[0, :, :, :3] == 0)
    assert torch.all(attention_mask[1, :, :, :8] == 0)
    # Pad between first-start+1 and the future block is -inf.
    assert torch.all(attention_mask[0, :, :, 3:10] == torch.finfo(torch.float32).min)
    assert torch.all(attention_mask[1, :, :, 8:10] == torch.finfo(torch.float32).min)
    # Row 1 has a later start, so columns 3..7 stay attendable.
    assert torch.all(attention_mask[1, :, :, 3:8] == 0)
