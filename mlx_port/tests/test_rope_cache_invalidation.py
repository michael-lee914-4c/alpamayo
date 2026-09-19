"""Stale 3-row mRoPE must not leak from clip A onto clip B.

``run_local_traj_sample`` and Stage-1/2 SFT load one model and reuse it
across clips. The get_rope_index idempotency guard is only valid for the
same prompt (prefill + base-class re-entry), not for a later batch.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_port.models.alpamayo_qwen3vl import (
    AlpamayoLanguageModel,
    _same_rope_prompt,
    invalidate_multimodal_rope,
)

IMAGE_ID = 151655
VIDEO_ID = 151656
VISION_START = 151652
VISION_END = 151653
TEXT_ID = 100


def _make_prompt(n_images, grid_hw=(4, 6), prefix=3, suffix=5):
    merge = 2
    h, w = grid_hw
    n_pads = (h // merge) * (w // merge)
    tokens = [TEXT_ID] * prefix
    for i in range(n_images):
        tokens.append(VISION_START)
        tokens.extend([IMAGE_ID] * n_pads)
        tokens.append(VISION_END)
    tokens.extend([TEXT_ID] * suffix)
    grids = np.array([[1, h, w]] * n_images, dtype=np.int64)
    return np.array([tokens], dtype=np.int64), grids


def _lm():
    obj = SimpleNamespace(
        config=SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2),
            image_token_id=IMAGE_ID,
            video_token_id=VIDEO_ID,
        ),
        _pos_valid=False,
        _position_ids=None,
        _rope_deltas=None,
        _attention_mask=None,
        _rope_cached_input_ids=None,
        _rope_cached_image_grid=None,
        _rope_cached_video_grid=None,
    )
    obj.get_rope_index = AlpamayoLanguageModel.get_rope_index.__get__(obj, type(obj))
    return obj


def _mx_ids_and_grid(n_images, prefix=3, suffix=5):
    ids, grids = _make_prompt(n_images=n_images, prefix=prefix, suffix=suffix)
    return mx.array(ids), mx.array(grids, dtype=mx.int32)


def test_invalidate_multimodal_rope_clears_cached_state():
    lm = SimpleNamespace(
        _pos_valid=True,
        _position_ids=object(),
        _rope_deltas=object(),
        _attention_mask=object(),
        _rope_cached_input_ids=object(),
        _rope_cached_image_grid=object(),
        _rope_cached_video_grid=object(),
    )
    invalidate_multimodal_rope(lm)
    assert lm._pos_valid is False
    assert lm._position_ids is None
    assert lm._rope_deltas is None
    assert lm._attention_mask is None
    assert lm._rope_cached_input_ids is None
    assert not _same_rope_prompt(lm, np.array([[1, 2]]), None, None)


def test_get_rope_index_reuses_cache_only_for_the_same_prompt():
    lm = _lm()
    ids_a, grid_a = _mx_ids_and_grid(n_images=2, prefix=3)
    ids_b, grid_b = _mx_ids_and_grid(n_images=1, prefix=7)

    pos_a, delta_a = lm.get_rope_index(ids_a, image_grid_thw=grid_a)
    pos_a_again, delta_a_again = lm.get_rope_index(ids_a, image_grid_thw=grid_a)
    assert pos_a is pos_a_again
    assert delta_a is delta_a_again

    pos_b, _ = lm.get_rope_index(ids_b, image_grid_thw=grid_b)
    assert pos_b is not pos_a
    assert tuple(np.asarray(pos_b).shape) != tuple(np.asarray(pos_a).shape)

    # Same length, different token ids: still recompute (multi-clip eval).
    ids_c_np = np.asarray(ids_a).copy()
    ids_c_np[0, 0] += 1
    pos_c, _ = lm.get_rope_index(mx.array(ids_c_np), image_grid_thw=grid_a)
    assert pos_c is not pos_a


def test_vision_prefill_invalidation_drops_previous_clip_rope():
    lm = _lm()
    ids_a, grid_a = _mx_ids_and_grid(n_images=2, prefix=3)
    pos_a, _ = lm.get_rope_index(ids_a, image_grid_thw=grid_a)
    assert lm._pos_valid is True

    invalidate_multimodal_rope(lm)
    assert lm._pos_valid is False
    assert not _same_rope_prompt(lm, ids_a, grid_a, None)

    pos_a2, _ = lm.get_rope_index(ids_a, image_grid_thw=grid_a)
    np.testing.assert_array_equal(np.asarray(pos_a2), np.asarray(pos_a))
    assert pos_a2 is not pos_a
