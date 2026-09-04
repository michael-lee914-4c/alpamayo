"""Unit tests for the mlx_vlm diffusion expert wrapper and NVIDIA mask helpers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_port.models.expert_mlx import (
    AlpamayoExpert,
    cache_seq_len,
    expert_attention_mask,
    expert_non_causal_train_mask,
    expert_position_ids,
    expert_rope_mask_contract,
    text_config_from_vlm_and_overrides,
    traj_future_start_offsets,
    trim_cache,
    sync_cache_idx,
)
from mlx_vlm.models.qwen3_vl.config import TextConfig


def test_text_config_keeps_vlm_gqa_and_overrides_width():
    base = {
        "model_type": "qwen3_vl",
        "num_hidden_layers": 36,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 151936,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 5_000_000.0,
        "max_position_embeddings": 262144,
        "rope_scaling": {"type": "default", "mrope_section": [24, 20, 20]},
        "tie_word_embeddings": False,
    }
    cfg = text_config_from_vlm_and_overrides(
        base,
        {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "intermediate_size": 8256,
            "head_dim": 128,
            "dtype": "bfloat16",
        },
    )
    assert cfg.hidden_size == 2048
    assert cfg.num_attention_heads == 16
    assert cfg.intermediate_size == 8256
    assert cfg.num_key_value_heads == 8
    assert cfg.num_hidden_layers == 36
    assert cfg.vocab_size == 1
    assert cfg.head_dim == 128


def test_traj_future_start_offsets_and_mask():
    seq = np.array([[1, 2, 99, 7, 8], [1, 99, 3, 4, 5]], dtype=np.int32)
    offsets = traj_future_start_offsets(seq, 99)
    np.testing.assert_array_equal(offsets, [3, 2])

    prefix_len = 5
    n_diff = 4
    mask = np.asarray(expert_attention_mask(2, n_diff, prefix_len, offsets))
    assert mask.shape == (2, 1, n_diff, prefix_len + n_diff)
    assert mask[0, 0, 0, 2] == 0.0
    assert mask[0, 0, 0, 3] < 0.0
    assert mask[0, 0, 0, 4] < 0.0
    assert mask[0, 0, 0, 5] == 0.0
    assert mask[1, 0, 0, 1] == 0.0
    assert mask[1, 0, 0, 2] < 0.0


def test_expert_position_ids_adds_delta_and_offset():
    pos = np.asarray(expert_position_ids(4, 2, np.array([[10], [20]]), np.array([3, 5])))
    assert pos.shape == (3, 2, 4)
    np.testing.assert_array_equal(pos[0, 0], [13, 14, 15, 16])
    np.testing.assert_array_equal(pos[2, 1], [25, 26, 27, 28])


def test_expert_rope_mask_contract_matches_nvidia():
    """pos0 = offset + rope_deltas; hide [offset, prefix); diffusion block is 0."""
    tfs = 155681
    prompt = [1, 2, 3]
    generated = [10, 11, tfs, 99]
    seq = np.array(prompt + generated, dtype=np.int32)
    offset = seq.tolist().index(tfs) + 1
    rope_deltas = np.array([[-31680]])
    prefix_len = offset + 3
    n_diff = 64
    c = expert_rope_mask_contract(seq, tfs, rope_deltas, prefix_len, n_diff)
    assert c["tfs_idx"] == seq.tolist().index(tfs)
    assert c["offset"] == offset
    assert c["pos0"] == offset + int(rope_deltas[0, 0])
    assert c["pos0_matches_offset_plus_delta"]
    assert c["hide_start"] == offset
    assert c["hide_end"] == prefix_len
    assert c["n_hidden"] == prefix_len - offset
    assert c["diffusion_block_max_abs"] == 0.0
    assert c["pos_last"] == c["pos0"] + n_diff - 1

    empty = expert_rope_mask_contract(seq, tfs, rope_deltas, offset, n_diff)
    assert empty["n_hidden"] == 0
    assert empty["hide_start"] is None
    assert empty["diffusion_block_max_abs"] == 0.0


def test_tiny_expert_forward_hidden_shape():
    cfg = TextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=8,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10_000.0,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [6, 5, 5]},
        tie_word_embeddings=True,
    )
    expert = AlpamayoExpert(cfg)
    embeds = mx.random.normal((2, 8, 64))
    pos = expert_position_ids(8, 2, np.zeros((2, 1), dtype=np.int32), np.array([0, 0]))
    hidden = expert(inputs_embeds=embeds, position_ids=pos, cache=None, mask=None)
    assert hidden.shape == (2, 8, 64)
    mx.eval(hidden)
    assert len(expert.layers) == 1


def test_trim_cache_restores_prefix_len():
    from mlx_lm.models.cache import KVCache

    cache = [KVCache()]
    keys = mx.random.normal((1, 2, 6, 4))
    values = mx.random.normal((1, 2, 6, 4))
    cache[0].update_and_fetch(keys, values)
    assert cache_seq_len(cache) == 6
    extra = mx.random.normal((1, 2, 4, 4))
    cache[0].update_and_fetch(extra, extra)
    assert cache_seq_len(cache) == 10
    cache[0]._idx = 9
    sync_cache_idx(cache)
    assert cache[0]._idx == 10
    trim_cache(cache, 4)
    assert cache_seq_len(cache) == 6
    assert cache[0]._idx == 6


def test_cache_seq_len_empty_or_uninitialized_is_zero():
    assert cache_seq_len([]) == 0
    assert cache_seq_len([None]) == 0


def test_expert_attention_mask_skips_hide_when_offset_not_in_prefix():
    no_hide = np.asarray(expert_attention_mask(1, 4, 5, np.array([5])))
    assert no_hide.shape == (1, 1, 4, 9)
    assert not np.any(no_hide[0, 0, 0, :5] < 0)
    past = np.asarray(expert_attention_mask(1, 4, 5, np.array([8])))
    assert not np.any(past[0, 0, 0, :5] < 0)
    negative = np.asarray(expert_attention_mask(1, 4, 5, np.array([-1])))
    assert not np.any(negative[0, 0, 0, :5] < 0)


def test_expert_non_causal_train_mask_is_zeros_and_rejects_empty():
    mask = np.asarray(expert_non_causal_train_mask(2, 4, 6))
    assert mask.shape == (2, 1, 4, 10)
    assert np.all(mask == 0.0)
    try:
        expert_non_causal_train_mask(0, 4, 1)
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("expected ValueError for batch=0")
    try:
        expert_non_causal_train_mask(1, 0, 1)
    except ValueError as exc:
        assert "n_tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError for n_tokens=0")
