"""Parity tests for DeltaTrajectoryTokenizerMLX vs NVIDIA."""

import numpy as np
import torch

from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer
from mlx_port.models.token_utils_mlx import tokenize_history_trajectory
from mlx_port.models.trajectory_tokenizer_mlx import DeltaTrajectoryTokenizerMLX


def _random_traj(B=2, T=16, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(scale=0.4, size=(B, T, 3)).astype(np.float32)
    xyz[..., 2] *= 0.05
    yaw = rng.normal(scale=0.1, size=(B, T)).astype(np.float32)
    rot = np.zeros((B, T, 3, 3), dtype=np.float32)
    rot[..., 0, 0] = np.cos(yaw)
    rot[..., 0, 1] = -np.sin(yaw)
    rot[..., 1, 0] = np.sin(yaw)
    rot[..., 1, 1] = np.cos(yaw)
    rot[..., 2, 2] = 1.0
    return xyz, rot


def test_encode_matches_nvidia_xyz():
    xyz, rot = _random_traj()
    nv = DeltaTrajectoryTokenizer()
    mlx = DeltaTrajectoryTokenizerMLX()
    nv_ids = nv.encode(
        hist_xyz=torch.zeros(xyz.shape[0], 1, 3),
        hist_rot=torch.eye(3).expand(xyz.shape[0], 1, 3, 3).contiguous(),
        fut_xyz=torch.from_numpy(xyz),
        fut_rot=torch.from_numpy(rot),
    ).numpy()
    mlx_ids = np.array(
        mlx.encode(
            hist_xyz=xyz[:, :1],
            hist_rot=rot[:, :1],
            fut_xyz=xyz,
            fut_rot=rot,
        )
    )
    assert mlx_ids.shape == (2, 48)
    np.testing.assert_array_equal(mlx_ids, nv_ids)


def test_encode_matches_nvidia_with_yaw():
    xyz, rot = _random_traj(seed=1)
    nv = DeltaTrajectoryTokenizer(predict_yaw=True)
    mlx = DeltaTrajectoryTokenizerMLX(predict_yaw=True)
    nv_ids = nv.encode(
        hist_xyz=torch.zeros(xyz.shape[0], 1, 3),
        hist_rot=torch.eye(3).expand(xyz.shape[0], 1, 3, 3).contiguous(),
        fut_xyz=torch.from_numpy(xyz),
        fut_rot=torch.from_numpy(rot),
    ).numpy()
    mlx_ids = np.array(
        mlx.encode(
            hist_xyz=xyz[:, :1],
            hist_rot=rot[:, :1],
            fut_xyz=xyz,
            fut_rot=rot,
        )
    )
    assert mlx_ids.shape == (2, 64)
    np.testing.assert_array_equal(mlx_ids, nv_ids)


def test_tokenize_history_emits_48_offset_ids():
    xyz, rot = _random_traj(B=1)
    tok = DeltaTrajectoryTokenizerMLX()
    ids = np.array(
        tokenize_history_trajectory(
            tok,
            {
                "ego_history_xyz": xyz[None, ...],
                "ego_history_rot": rot[None, ...],
            },
            start_idx=151669,
        )
    )
    assert ids.shape == (1, 48)
    assert ids.min() >= 151669
    assert ids.max() < 151669 + 1000
