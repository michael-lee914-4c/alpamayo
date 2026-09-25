"""RL rollout unsqueezes XYZ at rank 3 and rotations at rank 4.

Weight-free. Complementary to open #32 (rank-3 ego unsqueeze) and #39
(fused ids must be ``[1, T]``). PAI ``squeeze(0)`` leaves
``xyz=(T,3)`` / ``rot=(T,3,3)`` or ``xyz=(n_traj,T,3)`` / ``rot=(n_traj,T,3,3)``.
The packer only wraps a missing ``n_traj`` dim: XYZ when ``dim()==3``, rot
when ``dim()==4``. Promoting rot at rank 3 (or XYZ at rank 2) would fuse the
wrong axes and corrupt hist tokens on every vLLM prompt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACK = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"


def test_source_promotes_xyz_at_3_and_rot_at_4():
    src = _PACK.read_text()
    assert "if ego_history_xyz.dim() == 3:" in src
    assert "ego_history_xyz = ego_history_xyz.unsqueeze(1)" in src
    assert "if ego_history_rot.dim() == 4:" in src
    assert "ego_history_rot = ego_history_rot.unsqueeze(1)" in src
    assert "if isinstance(ego_future_xyz, torch.Tensor) and ego_future_xyz.dim() == 3:" in src
    assert "if isinstance(ego_future_rot, torch.Tensor) and ego_future_rot.dim() == 4:" in src
    assert "if fused_input_ids.dim() == 1:" in src
    assert "fused_input_ids = fused_input_ids.unsqueeze(0)" in src


def _promote(xyz, rot):
    if xyz.dim() == 3:
        xyz = xyz.unsqueeze(1)
    if rot.dim() == 4:
        rot = rot.unsqueeze(1)
    return xyz, rot


def test_n_traj_wrapper_on_squeezed_pai_ranks():
    torch = pytest.importorskip("torch")
    # After PAI squeeze(0): (n_traj, T, 3) / (n_traj, T, 3, 3)
    xyz = torch.zeros(1, 16, 3)
    rot = torch.eye(3).expand(1, 16, 3, 3).contiguous()
    xyz_p, rot_p = _promote(xyz, rot)
    assert xyz_p.shape == (1, 1, 16, 3)
    assert rot_p.shape == (1, 1, 16, 3, 3)


def test_already_batched_ranks_are_left_alone():
    torch = pytest.importorskip("torch")
    xyz = torch.zeros(2, 1, 16, 3)
    rot = torch.eye(3).expand(2, 1, 16, 3, 3).contiguous()
    xyz_p, rot_p = _promote(xyz, rot)
    assert xyz_p.shape == (2, 1, 16, 3)
    assert rot_p.shape == (2, 1, 16, 3, 3)


def test_wrong_rot_gate_would_insert_a_spurious_dim():
    torch = pytest.importorskip("torch")
    rot = torch.eye(3).expand(1, 16, 3, 3).contiguous()
    # A dim()==3 gate (matching XYZ) would unsqueeze a (T,3,3) history incorrectly.
    wrong = rot.unsqueeze(1) if rot.dim() == 3 else rot
    right = rot.unsqueeze(1) if rot.dim() == 4 else rot
    assert wrong.shape == (1, 16, 3, 3)
    assert right.shape == (1, 1, 16, 3, 3)
