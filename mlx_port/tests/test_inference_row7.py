"""Structural tests for Row 7 inference components."""

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_port.inference import _rope_deltas_np
from mlx_port.models.token_utils_mlx import (
    AlpamayoGenerateStop,
    ExpertLogitsProcessor,
    StopAfterEOS,
    hf_eos_token_ids,
    make_vlm_generate_stop,
    replace_padding_after_eos,
)


def test_expert_logits_processor_masks_traj_tokens():
    proc = ExpertLogitsProcessor(traj_token_offset=100, traj_vocab_size=50)
    scores = mx.zeros((2, 200))
    out = proc(mx.array([[1, 2]]), scores)
    # The masked region should be -inf
    assert mx.all(out[0, 100:150] == float("-inf"))
    assert out[0, 99] == 0.0
    assert out[0, 150] == 0.0


def test_stop_after_eos_stops_one_token_after():
    stop = StopAfterEOS(eos_token_id=42)
    ids = mx.array([[1, 2, 42], [3, 4, 5]])
    stop(ids, None)
    # After first call with EOS in row 0, eos_found should be set
    assert stop.eos_found[0] == True
    assert stop.eos_found[1] == False


def test_stop_after_eos_reuse_without_reset_stops_immediately():
    stop = StopAfterEOS(eos_token_id=42)
    assert stop(mx.array([[1, 42]])) is False
    assert stop(mx.array([[1, 42, 7]])) is True
    assert stop(mx.array([[9]])) is True


def test_stop_after_eos_reset_allows_next_sample():
    stop = StopAfterEOS(eos_token_id=42)
    assert stop(mx.array([[1, 42]])) is False
    assert stop(mx.array([[1, 42, 7]])) is True
    stop.reset()
    assert stop(mx.array([[9]])) is False
    assert stop(mx.array([[9, 42]])) is False
    assert stop(mx.array([[9, 42, 3]])) is True


def test_alpamayo_generate_stop_reset_allows_next_sample():
    stop = AlpamayoGenerateStop(delayed_eos_id=42, immediate_eos_ids=[99])
    assert stop(mx.array([[1, 42]])) is False
    assert stop(mx.array([[1, 42, 7]])) is True
    stop.reset()
    assert stop(mx.array([[9]])) is False
    assert stop(mx.array([[9, 42]])) is False
    assert stop(mx.array([[9, 42, 3]])) is True


def test_replace_padding_after_eos():
    tokens = mx.array([[1, 2, 42, 99, 100], [3, 42, 5, 6, 7]])
    out = replace_padding_after_eos(tokens, eos_token_id=42, pad_token_id=0)
    assert out[0, 3] == 0 and out[0, 4] == 0
    assert out[1, 2] == 0 and out[1, 3] == 0 and out[1, 4] == 0


def test_hf_generate_stop_ends_immediately_on_im_end():
    """HF generate finishes a sequence as soon as eos_token_id is emitted."""
    im_end, endoftext, traj_future = 151645, 151643, 155681
    stop = AlpamayoGenerateStop(
        delayed_eos_id=traj_future,
        immediate_eos_ids=[im_end, endoftext],
    )
    assert stop(mx.array([[19434, im_end]])) is True
    assert stop(mx.array([[19434, endoftext]])) is True
    assert stop(mx.array([[19434, 2115]])) is False


def test_hf_generate_stop_keeps_delayed_traj_future_start():
    im_end, traj_future = 151645, 155681
    stop = AlpamayoGenerateStop(
        delayed_eos_id=traj_future,
        immediate_eos_ids=[im_end, traj_future],
    )
    assert traj_future not in stop.immediate_eos_ids
    first = stop(mx.array([[1, 2, traj_future]]))
    assert first is False
    assert stop(mx.array([[1, 2, traj_future, 99]])) is True


def test_qwen_hf_eos_ids_match_generation_config():
    from transformers import AutoTokenizer

    from mlx_port.processor import LOCAL_QWEN_PROCESSOR_PATH

    if not Path(LOCAL_QWEN_PROCESSOR_PATH).exists():
        pytest.skip("local Qwen3-VL processor not present")

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_QWEN_PROCESSOR_PATH, trust_remote_code=True
    )
    ids = hf_eos_token_ids(tokenizer)
    assert ids == [151645, 151643]
    stop = make_vlm_generate_stop(tokenizer, delayed_eos_id=155681)
    assert stop.immediate_eos_ids == {151645, 151643}


def test_rope_deltas_np_broadcasts_scalar_and_truncates():
    model = SimpleNamespace(
        vlm=SimpleNamespace(language_model=SimpleNamespace(_rope_deltas=None))
    )
    zeros = _rope_deltas_np(model, 3)
    assert zeros.shape == (3, 1)
    assert np.all(zeros == 0)
    model.vlm.language_model._rope_deltas = np.array([5])
    broadcast = _rope_deltas_np(model, 2)
    assert broadcast.shape == (2, 1)
    assert np.all(broadcast == 5)
    model.vlm.language_model._rope_deltas = np.array([1, 2, 3, 4])
    truncated = _rope_deltas_np(model, 2)
    assert truncated.reshape(-1).tolist() == [1, 2]