"""NVIDIA infer expert offset: first start, else last token.

Weight-free. Complementary to open #30 (infer first-start +1 RoPE).
``sample_trajectories_from_data_with_vlm_rollout`` uses ``argmax`` of
the start mask (first hit), then falls back to ``L-1`` when a row has
no ``<|traj_future_start|>``. ``offset = pos + 1`` also blanks
``attention_mask[..., offset: -n_diffusion]`` so padding after the
marker is not attended.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODEL = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_expert_offset_uses_first_start_not_last(torch):
    src = _MODEL.read_text()
    assert "traj_future_start_positions = traj_future_start_mask.int().argmax(dim=1)" in src
    assert "valid_token_pos_id = torch.where(" in src
    assert "offset = valid_token_pos_id + 1" in src

    start_id = 9
    # Two starts on row 0 (cols 2 and 6). First-start offset is 3, last would be 7.
    sequences = torch.tensor(
        [
            [1, 2, start_id, 3, 4, 5, start_id, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    mask = sequences == start_id
    has_start = mask.any(dim=1)
    first = mask.int().argmax(dim=1)
    last_token = torch.full((sequences.shape[0],), sequences.shape[1] - 1)
    pos = torch.where(has_start, first, last_token)
    offset = pos + 1
    assert has_start.tolist() == [True, False]
    assert first[0].item() == 2
    assert offset.tolist() == [3, 8]


def test_missing_start_falls_back_to_last_token_plus_one(torch):
    src = _MODEL.read_text()
    assert "has_traj_future_start, traj_future_start_positions, last_token_positions" in src
    assert "No <traj_future_start> token found" in src

    start_id = 9
    sequences = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    mask = sequences == start_id
    has_start = mask.any(dim=1)
    last_token = torch.full((1,), sequences.shape[1] - 1)
    pos = torch.where(has_start, mask.int().argmax(dim=1), last_token)
    assert not bool(has_start[0])
    assert int(pos[0]) == 3
    assert int(pos[0] + 1) == 4


def test_attention_mask_blanks_from_offset_to_diffusion_tail(torch):
    src = _MODEL.read_text()
    assert "attention_mask[i, :, :, offset[i] : -n_diffusion_tokens] = torch.finfo(" in src

    n_diff = 2
    kv_len = 6
    offset = torch.tensor([3, 5])
    mask = torch.zeros((2, 1, n_diff, kv_len + n_diff), dtype=torch.float32)
    fill = float(torch.finfo(mask.dtype).min)
    for i in range(2):
        mask[i, :, :, offset[i] : -n_diff] = fill
    # Row 0: columns 3,4 (kv 3,4) blanked; 0..2 and diffusion tail stay 0.
    assert float(mask[0, 0, 0, 3]) == fill
    assert float(mask[0, 0, 0, 4]) == fill
    assert float(mask[0, 0, 0, 2]) == 0.0
    assert float(mask[0, 0, 0, 6]) == 0.0
    # Row 1 offset 5 blanks only kv col 5.
    assert float(mask[1, 0, 0, 5]) == fill
    assert float(mask[1, 0, 0, 4]) == 0.0
