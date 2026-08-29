"""HF-faithful Qwen3-VL RoPE index (get_vision_position_ids / get_rope_index).

Port of transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLModel
get_vision_position_ids and get_rope_index. The stock mlx_vlm path follows
the older Qwen2-VL layout (advance by token count / max+1). Qwen3-VL instead:

* marks only image/video pad tokens as multimodal (vision_start/end are text)
* places vision T/H/W from get_vision_position_ids(start, grid, merge=2)
* advances current_pos by max(H, W) // spatial_merge_size after each image
  (60 for a 68×120 grid), not by the 2040 pad tokens

Alpamayo prompts have one image-pad run per camera frame (16 runs). When the
port groups image_grid_thw to 4×[4,H,W] for Conv3D, those rows are split to
16×[1,H,W] before RoPE so the HF iterator stays aligned with the token runs.
"""

from __future__ import annotations

import itertools
from typing import Optional

import mlx.core as mx
import numpy as np


def get_vision_position_ids(
    start_position: int,
    grid_thw,
    temp_merge_size: int = 1,
    spatial_merge_size: int = 1,
    time_interval: int = 1,
) -> np.ndarray:
    """3-row (T, H, W) positions for one image/video grid. Shape (3, n_tokens)."""
    t_raw, h_raw, w_raw = (int(x) for x in np.asarray(grid_thw).reshape(-1)[:3])
    llm_grid_t = t_raw // temp_merge_size
    llm_grid_h = h_raw // spatial_merge_size
    llm_grid_w = w_raw // spatial_merge_size

    # Match HF: add start_position after arange; T offset is applied last.
    position_temporal = np.arange(llm_grid_t, dtype=np.int64) * time_interval
    position_width = np.arange(llm_grid_w, dtype=np.int64) + int(start_position)
    position_height = np.arange(llm_grid_h, dtype=np.int64) + int(start_position)

    position_width = np.tile(position_width, llm_grid_h * llm_grid_t)
    position_height = np.repeat(position_height, llm_grid_w)
    position_height = np.tile(position_height, llm_grid_t)
    position_temporal = np.repeat(position_temporal, llm_grid_h * llm_grid_w) + int(
        start_position
    )
    return np.stack([position_temporal, position_height, position_width], axis=0)


def mm_token_type_ids_from_input_ids(
    input_ids: np.ndarray,
    image_token_id: int,
    video_token_id: int,
) -> np.ndarray:
    """HF ProcessorMixin.create_mm_token_type_ids: 0=text, 1=image, 2=video."""
    types = np.zeros_like(input_ids, dtype=np.int64)
    types[input_ids == image_token_id] = 1
    types[input_ids == video_token_id] = 2
    return types


def split_temporal_grids(grid_thw: np.ndarray) -> np.ndarray:
    """HF video split: repeat each row T times and set T=1."""
    grid = np.asarray(grid_thw, dtype=np.int64)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    rows = []
    for row in grid:
        t, h, w = int(row[0]), int(row[1]), int(row[2])
        for _ in range(max(t, 1)):
            rows.append([1, h, w])
    return np.asarray(rows, dtype=np.int64)


def _maybe_split_image_grids(image_grid_thw: np.ndarray, n_image_runs: int) -> np.ndarray:
    """Split T>1 image grids when they were grouped (4×[4,H,W] vs 16 pad runs)."""
    grid = np.asarray(image_grid_thw, dtype=np.int64)
    if grid.ndim == 1:
        grid = grid.reshape(1, -1)
    if grid.shape[0] == n_image_runs:
        return grid
    t_sum = int(grid[:, 0].sum())
    if t_sum == n_image_runs:
        return split_temporal_grids(grid)
    raise ValueError(
        f"image_grid_thw has {grid.shape[0]} rows (T sum={t_sum}) but the "
        f"prompt has {n_image_runs} image-pad runs. Pass one [T,H,W] per "
        f"image token run, or T-grouped rows whose T values sum to the run count."
    )


def _modality_groups(token_types: list[int]) -> list[tuple[int, int, int]]:
    groups = []
    for key, group in itertools.groupby(enumerate(token_types), lambda x: int(x[1])):
        group = list(group)
        start_index = group[0][0]
        end_index = group[-1][0] + 1
        groups.append((int(key), start_index, end_index))
    return groups


def compute_hf_rope_index(
    input_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
    mm_token_type_ids=None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    spatial_merge_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (position_ids [3,B,S], mrope_deltas [B,1]) matching HF Qwen3-VL."""
    ids = np.asarray(input_ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    batch_size, seq_length = ids.shape

    if mm_token_type_ids is None:
        types = mm_token_type_ids_from_input_ids(ids, image_token_id, video_token_id)
    else:
        types = np.asarray(mm_token_type_ids)
        if types.ndim == 1:
            types = types[None, :]

    mask = None if attention_mask is None else np.asarray(attention_mask)

    image_grid = None if image_grid_thw is None else np.asarray(image_grid_thw)
    video_grid = None if video_grid_thw is None else np.asarray(video_grid_thw)
    if video_grid is not None:
        video_grid = split_temporal_grids(video_grid)

    position_ids = np.zeros((3, batch_size, seq_length), dtype=np.int64)
    mrope_position_deltas = []

    image_rows: list[np.ndarray] = []
    if image_grid is not None:
        first_types = types[0] if mask is None else types[0][mask[0].astype(bool)]
        n_image_runs = sum(1 for k, _, _ in _modality_groups(first_types.tolist()) if k == 1)
        image_grid = _maybe_split_image_grids(image_grid, n_image_runs)
        image_rows = [image_grid[i] for i in range(image_grid.shape[0])]
    video_rows: list[np.ndarray] = []
    if video_grid is not None:
        video_rows = [video_grid[i] for i in range(video_grid.shape[0])]

    image_iter = iter(image_rows)
    video_iter = iter(video_rows)

    for batch_idx in range(batch_size):
        current_ids = ids[batch_idx]
        current_types = types[batch_idx]
        if mask is not None:
            keep = mask[batch_idx].astype(bool)
            current_ids = current_ids[keep]
            current_types = current_types[keep]

        current_pos = 0
        llm_pos_ids_list: list[np.ndarray] = []
        for modality_type, start_idx, end_idx in _modality_groups(current_types.tolist()):
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    np.broadcast_to(
                        np.arange(text_len, dtype=np.int64)[None, :], (3, text_len)
                    )
                    + current_pos
                )
                current_pos += text_len
                continue
            if modality_type == 1:
                grid_thw = next(image_iter)
            elif modality_type == 2:
                grid_thw = next(video_iter)
            else:
                raise ValueError(f"unsupported mm token type {modality_type}")
            vision_position_ids = get_vision_position_ids(
                current_pos, grid_thw, 1, spatial_merge_size
            )
            expected = end_idx - start_idx
            if vision_position_ids.shape[1] != expected:
                raise ValueError(
                    f"vision RoPE length {vision_position_ids.shape[1]} != "
                    f"token-run length {expected} for grid {grid_thw.tolist()} "
                    f"merge={spatial_merge_size}"
                )
            llm_pos_ids_list.append(vision_position_ids)
            current_pos += max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        if mask is not None:
            position_ids[:, batch_idx, mask[batch_idx].astype(bool)] = llm_positions
        else:
            position_ids[:, batch_idx] = llm_positions
        mrope_position_deltas.append(int(llm_positions.max()) + 1 - len(current_ids))

    deltas = np.asarray(mrope_position_deltas, dtype=np.int64).reshape(-1, 1)
    return position_ids, deltas


def compute_hf_rope_index_mx(
    input_ids: mx.array,
    image_grid_thw: Optional[mx.array] = None,
    video_grid_thw: Optional[mx.array] = None,
    attention_mask: Optional[mx.array] = None,
    mm_token_type_ids: Optional[mx.array] = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    spatial_merge_size: int = 2,
) -> tuple[mx.array, mx.array]:
    pos, deltas = compute_hf_rope_index(
        input_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    return mx.array(pos), mx.array(deltas)
