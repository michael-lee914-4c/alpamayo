"""NVIDIA RL packer: do_rescale=True, image-pad collapse, rank-3 unsqueeze.

Weight-free. Complementary to open #27 (camera×frame flatten) and
#30 (SFT Qwen uint8 /255 + ``do_rescale=False``) — RL vLLM prompts set
``do_rescale=True``, collapse consecutive ``image_pad`` tokens, and
promote rank-3 ego to ``(B, 1, T, ...)`` before fuse. Unifying the
rescale flags or dropping the unsqueeze would double-scale images or
fail the 4D hist/future tokenize asserts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_rl_mm_kwargs_rescale_true_sft_rescale_false():
    packer = _PACKER.read_text()
    qwen = _QWEN.read_text()
    assert '"do_rescale": True' in packer
    assert "do_rescale=False" in qwen
    # SFT already divided uint8 by 255; RL leaves that to the HF processor.
    assert "images.float() / 255.0" in qwen
    assert "do_rescale=False" not in packer


def test_consecutive_image_pad_collapses_to_one_per_run():
    src = _PACKER.read_text()
    assert "tid == img_id and i > 0 and token_ids[i - 1] == img_id" in src

    img_id = 99
    token_ids = [1, img_id, img_id, img_id, 7, img_id, img_id, 8]
    collapsed = [
        tid
        for i, tid in enumerate(token_ids)
        if not (tid == img_id and i > 0 and token_ids[i - 1] == img_id)
    ]
    assert collapsed == [1, img_id, 7, img_id, 8]
    # A missing predecessor check would keep the whole run or drop the first pad.
    naive_drop_all_pad = [tid for tid in token_ids if tid != img_id]
    assert naive_drop_all_pad != collapsed


def test_rank3_ego_is_unsqueezed_to_n_traj_one(torch):
    src = _PACKER.read_text()
    assert "if ego_history_xyz.dim() == 3:" in src
    assert "ego_history_xyz = ego_history_xyz.unsqueeze(1)" in src
    assert "if ego_history_rot.dim() == 4:" in src
    assert "ego_history_rot = ego_history_rot.unsqueeze(1)" in src
    assert "if isinstance(ego_future_xyz, torch.Tensor) and ego_future_xyz.dim() == 3:" in src

    xyz = torch.zeros(2, 4, 3)
    rot = torch.eye(3).expand(2, 4, 3, 3).clone()
    if xyz.dim() == 3:
        xyz = xyz.unsqueeze(1)
    if rot.dim() == 4:
        rot = rot.unsqueeze(1)
    assert xyz.shape == (2, 1, 4, 3)
    assert rot.shape == (2, 1, 4, 3, 3)
    # Already-batched (B, n_traj, T, 3) must not grow a second n_traj axis.
    already = torch.zeros(2, 1, 4, 3)
    kept = already if already.dim() != 3 else already.unsqueeze(1)
    assert kept.shape == (2, 1, 4, 3)
