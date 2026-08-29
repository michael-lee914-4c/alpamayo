"""Unit tests for NVIDIA-matching temperature + top_p sampling."""

import mlx.core as mx
import numpy as np

from mlx_port.inference import apply_top_p, sample_next_token


def test_greedy_is_argmax():
    logits = mx.array([[1.0, 10.0, 2.0, -5.0]])
    tok = sample_next_token(logits, temperature=0.0, top_p=0.98)
    assert int(tok.item()) == 1


def test_top_p_keeps_token_that_crosses_threshold():
    # Softmax([10, 1, 1, 0]) is dominated by index 0 (~0.999).
    logits = mx.array([[10.0, 1.0, 1.0, 0.0]])
    masked = np.asarray(apply_top_p(logits, 0.9))
    assert np.isfinite(masked[0, 0])
    assert not np.isfinite(masked[0, 1])
    assert not np.isfinite(masked[0, 2])
    assert not np.isfinite(masked[0, 3])


def test_top_p_one_is_noop():
    logits = mx.array([[1.0, 2.0, 3.0]])
    out = np.asarray(apply_top_p(logits, 1.0))
    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0])


def test_nucleus_does_not_sample_masked_tail():
    mx.random.seed(0)
    logits = mx.array([[20.0, -20.0, -20.0, -20.0]])
    for _ in range(8):
        tok = int(sample_next_token(logits, temperature=0.6, top_p=0.98).item())
        assert tok == 0
