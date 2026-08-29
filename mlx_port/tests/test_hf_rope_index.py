"""Parity tests for HF Qwen3-VL get_rope_index / get_vision_position_ids."""

import types

import numpy as np
import torch

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from mlx_port.models.rope_index_mlx import (
    compute_hf_rope_index,
    get_vision_position_ids,
    mm_token_type_ids_from_input_ids,
    split_temporal_grids,
)

IMAGE_ID = 151655
VIDEO_ID = 151656
VISION_START = 151652
VISION_END = 151653
TEXT_ID = 100


def _hf_dummy(spatial_merge_size=2):
    dummy = types.SimpleNamespace()
    dummy.config = types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=spatial_merge_size)
    )
    dummy.get_vision_position_ids = types.MethodType(
        Qwen3VLModel.get_vision_position_ids, dummy
    )
    dummy.get_rope_index = types.MethodType(Qwen3VLModel.get_rope_index, dummy)
    return dummy


def _make_prompt(n_images, grid_hw=(4, 6), prefix=3, between=2, suffix=5):
    """Build a tiny Qwen-style prompt: text + (vision_start + pads + vision_end)*N + suffix."""
    merge = 2
    h, w = grid_hw
    n_pads = (h // merge) * (w // merge)
    tokens = [TEXT_ID] * prefix
    for i in range(n_images):
        tokens.append(VISION_START)
        tokens.extend([IMAGE_ID] * n_pads)
        tokens.append(VISION_END)
        if i + 1 < n_images:
            tokens.extend([TEXT_ID] * between)
    tokens.extend([TEXT_ID] * suffix)
    grids = np.array([[1, h, w]] * n_images, dtype=np.int64)
    return np.array([tokens], dtype=np.int64), grids


def test_get_vision_position_ids_matches_hf():
    start = 11
    grid = [1, 68, 120]
    ours = get_vision_position_ids(start, grid, 1, 2)
    hf = (
        _hf_dummy()
        .get_vision_position_ids(start, torch.tensor(grid), 1, 2)
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(ours, hf)
    # Compact spatial extent: W is start..start+59, not start..start+2039.
    assert ours.shape == (3, 34 * 60)
    assert ours[2].min() == start
    assert ours[2].max() == start + 59
    assert ours[1].max() == start + 33
    assert set(ours[0].tolist()) == {start}


def test_get_rope_index_matches_hf_two_images():
    input_ids, grids = _make_prompt(n_images=2, grid_hw=(4, 6))
    types = mm_token_type_ids_from_input_ids(input_ids, IMAGE_ID, VIDEO_ID)
    pos, deltas = compute_hf_rope_index(
        input_ids,
        image_grid_thw=grids,
        mm_token_type_ids=types,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    hf_pos, hf_deltas = _hf_dummy().get_rope_index(
        torch.from_numpy(input_ids),
        torch.from_numpy(types),
        image_grid_thw=torch.from_numpy(grids),
    )
    np.testing.assert_array_equal(pos, hf_pos.cpu().numpy())
    np.testing.assert_array_equal(deltas, hf_deltas.cpu().numpy())


def test_grouped_thw_matches_per_frame_grids():
    input_ids, per_frame = _make_prompt(n_images=16, grid_hw=(68, 120), suffix=8)
    grouped = np.array([[4, 68, 120]] * 4, dtype=np.int64)
    pos_flat, d_flat = compute_hf_rope_index(
        input_ids,
        image_grid_thw=per_frame,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    pos_grp, d_grp = compute_hf_rope_index(
        input_ids,
        image_grid_thw=grouped,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    np.testing.assert_array_equal(pos_flat, pos_grp)
    np.testing.assert_array_equal(d_flat, d_grp)

    seq_len = input_ids.shape[1]
    max_pos = int(pos_flat.max())
    # HF compact layout: 16 images * ~60 + text, not ~seq_len (~32k).
    assert max_pos < 2000
    assert max_pos < seq_len // 8
    assert int(d_flat[0, 0]) == max_pos + 1 - seq_len

    tail = pos_flat[:, 0, -8:]
    # Suffix is text: T=H=W incrementing by 1.
    np.testing.assert_array_equal(tail[0], tail[1])
    np.testing.assert_array_equal(tail[1], tail[2])
    np.testing.assert_array_equal(np.diff(tail[0]), np.ones(7, dtype=np.int64))


def test_vision_start_is_text_not_vision():
    input_ids, grids = _make_prompt(n_images=1, grid_hw=(4, 6), prefix=1, suffix=1)
    types = mm_token_type_ids_from_input_ids(input_ids, IMAGE_ID, VIDEO_ID)
    # vision_start / vision_end stay type 0.
    assert types[0, 1] == 0
    assert types[0, 2] == 1
    pos, _ = compute_hf_rope_index(
        input_ids,
        image_grid_thw=grids,
        mm_token_type_ids=types,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    # Token 1 is vision_start: 1D text at current_pos=1.
    assert pos[0, 0, 1] == pos[1, 0, 1] == pos[2, 0, 1] == 1
    # First image pad uses get_vision_position_ids(start=2, ...).
    assert pos[0, 0, 2] == 2
    assert pos[1, 0, 2] == 2
    assert pos[2, 0, 2] == 2


def test_video_grid_is_split_like_hf():
    split = split_temporal_grids(np.array([[4, 68, 120]], dtype=np.int64))
    np.testing.assert_array_equal(split, np.array([[1, 68, 120]] * 4))

    # One video [2,4,6] becomes two T=1 runs of 2*3 pads, with text between
    # (HF timestamp separators are type 0).
    merge = 2
    pads = (4 // merge) * (6 // merge)
    tokens = [TEXT_ID] + [VIDEO_ID] * pads + [TEXT_ID] + [VIDEO_ID] * pads + [TEXT_ID]
    input_ids = np.array([tokens], dtype=np.int64)
    types = mm_token_type_ids_from_input_ids(input_ids, IMAGE_ID, VIDEO_ID)
    video_grid = np.array([[2, 4, 6]], dtype=np.int64)
    pos, deltas = compute_hf_rope_index(
        input_ids,
        video_grid_thw=video_grid,
        mm_token_type_ids=types,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    hf_pos, hf_deltas = _hf_dummy().get_rope_index(
        torch.from_numpy(input_ids),
        torch.from_numpy(types),
        video_grid_thw=torch.from_numpy(video_grid),
    )
    np.testing.assert_array_equal(pos, hf_pos.cpu().numpy())
    np.testing.assert_array_equal(deltas, hf_deltas.cpu().numpy())


def test_decode_continues_from_max_plus_one():
    input_ids, grids = _make_prompt(n_images=2, grid_hw=(68, 120), suffix=4)
    pos, deltas = compute_hf_rope_index(
        input_ids,
        image_grid_thw=grids,
        image_token_id=IMAGE_ID,
        video_token_id=VIDEO_ID,
        spatial_merge_size=2,
    )
    seq_len = input_ids.shape[1]
    # Official decode: arange(1) + cache_offset + rope_deltas = max_pos + 1.
    decode_pos = 0 + seq_len + int(deltas[0, 0])
    assert decode_pos == int(pos.max()) + 1
