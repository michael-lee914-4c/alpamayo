"""Unit tests for NVIDIA-matching temperature + top_p sampling."""

import mlx.core as mx
import numpy as np

from mlx_port.inference import (
    _history_last_group,
    _image_kwargs_from_tokenized,
    _lookup_token_id,
    _rewind_kv_cache,
    apply_top_p,
    sample_next_token,
)


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


def test_history_last_group_keeps_last_not_first_traj():
    xyz = np.zeros((1, 2, 4, 3), dtype=np.float32)
    xyz[0, 0, :, 0] = 1.0
    xyz[0, 1, :, 0] = 9.0
    rot = np.zeros((1, 2, 4, 3, 3), dtype=np.float32)
    rot[0, 1] = 1.0
    xyz_b, rot_b = _history_last_group(xyz, rot)
    assert tuple(xyz_b.shape) == (1, 4, 3)
    assert float(np.asarray(xyz_b)[0, 0, 0]) == 9.0
    assert float(np.abs(np.asarray(rot_b)).max()) == 1.0
    xyz2, rot2 = _history_last_group(xyz_b, rot_b)
    assert tuple(xyz2.shape) == (1, 4, 3)
    assert tuple(rot2.shape) == (1, 4, 3, 3)


def test_lookup_token_id_drops_unk_and_multi_piece():
    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, text):
            return {"ok": 5, "<unk>": 0}.get(text)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            if text == "two tokens":
                return [1, 2]
            if text == "single":
                return [7]
            return []

    tok = _Tok()
    assert _lookup_token_id(tok, "ok") == 5
    assert _lookup_token_id(tok, "<unk>") is None
    assert _lookup_token_id(tok, "two tokens") is None
    assert _lookup_token_id(tok, "single") == 7


def test_rewind_kv_cache_trims_only_decode_tail():
    class _C:
        def __init__(self):
            self.offset = 10
            self._idx = 10
            self.trimmed = 0

        def trim(self, n):
            self.trimmed = int(n)
            self.offset -= int(n)

    cache = [_C()]
    _rewind_kv_cache(cache, 6)
    assert cache[0].trimmed == 4
    assert cache[0].offset == 6
    assert cache[0]._idx == 6
    _rewind_kv_cache(cache, 6)
    assert cache[0].trimmed == 4
    assert cache[0]._idx == 6


def test_image_kwargs_from_tokenized_transposes_hwc():
    hwc = np.zeros((2, 4, 8, 8, 3), dtype=np.float32)
    out = _image_kwargs_from_tokenized(
        {"pixel_values": hwc, "image_grid_thw": np.ones((2, 3), dtype=np.int32)}
    )
    assert tuple(np.asarray(out["pixel_values"]).shape) == (2, 3, 4, 8, 8)
    assert np.asarray(out["image_grid_thw"]).dtype == np.int32
    flats = np.zeros((16, 1536), dtype=np.float32)
    out2 = _image_kwargs_from_tokenized({"pixel_values": flats})
    assert tuple(np.asarray(out2["pixel_values"]).shape) == (16, 1536)
    assert "image_grid_thw" not in out2
