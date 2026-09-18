"""History tokenize pins ``hist_xyz[:, :1]`` (first step) and traj-major flatten.

Weight-free. Complementary to open #25 (rank-4 / flatten / ``start_idx``)
and #30 (future tokenize is *not* the hist-as-fut trick). Encode sees
the first history waypoint as ``hist_*`` and the full history as
``fut_*``. ``einops`` then concatenates traj 0's tokens before traj 1.
A last-step pin or an ``(n_traj n)`` swap retargets every fused pad.
"""

from __future__ import annotations

from pathlib import Path

import einops
import torch

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


class _RecordingTok:
    def __init__(self):
        self.calls = []

    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        self.calls.append(
            {
                "hist_xyz": hist_xyz.clone(),
                "hist_rot": hist_rot.clone(),
                "fut_xyz": fut_xyz.clone(),
                "fut_rot": fut_rot.clone(),
            }
        )
        # Distinct per flattened row so rearrange order is visible.
        rows = hist_xyz.shape[0]
        return torch.arange(rows * 2, dtype=torch.long).reshape(rows, 2)


def _tokenize_history_trajectory(tokenizer, traj_data, start_idx=0):
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4
    B = traj_data["ego_history_xyz"].shape[0]
    hist_xyz = traj_data["ego_history_xyz"].flatten(start_dim=0, end_dim=1)
    hist_rot = traj_data["ego_history_rot"].flatten(start_dim=0, end_dim=1)
    hist_idx = (
        tokenizer.encode(
            hist_xyz=hist_xyz[:, :1],
            hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz,
            fut_rot=hist_rot,
        )
        + start_idx
    )
    return einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)


def test_source_uses_first_timestep_and_traj_major_rearrange():
    src = _BASE.read_text()
    assert "hist_xyz=hist_xyz[:, :1]" in src
    assert "hist_rot=hist_rot[:, :1]" in src
    assert "fut_xyz=hist_xyz" in src
    assert 'einops.rearrange(hist_idx, "(b n_traj) n -> b (n_traj n)", b=B)' in src
    assert "flatten(start_dim=0, end_dim=1)" in src


def test_encode_sees_first_not_last_history_step():
    tok = _RecordingTok()
    xyz = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            ]
        ]
    )
    rot = torch.eye(3).expand(1, 2, 3, 3, 3).clone()
    _tokenize_history_trajectory(tok, {"ego_history_xyz": xyz, "ego_history_rot": rot})
    assert len(tok.calls) == 1
    hist = tok.calls[0]["hist_xyz"]
    fut = tok.calls[0]["fut_xyz"]
    assert hist.shape == (2, 1, 3)
    assert torch.equal(hist[0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(hist[1, 0], torch.tensor([10.0, 0.0, 0.0]))
    assert not torch.equal(hist[0, 0], xyz[0, 0, -1])
    assert fut.shape == (2, 3, 3)
    assert torch.equal(fut[0], xyz[0, 0])
    assert torch.equal(fut[1], xyz[0, 1])


def test_n_traj_tokens_are_traj0_then_traj1():
    tok = _RecordingTok()
    xyz = torch.zeros(1, 2, 3, 3)
    rot = torch.eye(3).expand(1, 2, 3, 3, 3).clone()
    got = _tokenize_history_trajectory(
        tok, {"ego_history_xyz": xyz, "ego_history_rot": rot}, start_idx=3000
    )
    # encode returns [[0, 1], [2, 3]] then +3000, rearranged traj-major.
    assert got.tolist() == [[3000, 3001, 3002, 3003]]


def test_rank_3_history_is_rejected():
    tok = _RecordingTok()
    try:
        _tokenize_history_trajectory(
            tok,
            {
                "ego_history_xyz": torch.zeros(2, 3, 3),
                "ego_history_rot": torch.zeros(2, 3, 3, 3),
            },
        )
    except AssertionError:
        return
    raise AssertionError("rank-3 history must fail")
