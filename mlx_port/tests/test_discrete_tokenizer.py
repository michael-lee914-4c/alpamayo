"""Discrete future-traj tokenizer + fuse (NVIDIA Stage 1)."""

import numpy as np

from mlx_port.models.token_utils_mlx import (
    fuse_traj_tokens,
    tokenize_future_trajectory,
)
from mlx_port.models.trajectory_tokenizer_mlx import DiscreteTrajectoryTokenizerMLX


class _StubActionSpace:
    def __init__(self, n_wp=64):
        self.n_wp = n_wp

    def get_action_space_dims(self):
        return (self.n_wp, 2)

    def traj_to_action(self, hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=None):
        del hist_xyz, hist_rot, fut_rot, t0_states
        b = int(np.asarray(fut_xyz).shape[0])
        return np.zeros((b, self.n_wp, 2), dtype=np.float64)


class _StubFutureTok:
    def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
        b = int(np.asarray(fut_xyz).shape[0])
        t = int(np.asarray(fut_xyz).shape[1]) * 2
        return np.arange(b * t, dtype=np.int64).reshape(b, t)


def test_discrete_zero_action_bin_matches_nvidia_formula():
    tok = DiscreteTrajectoryTokenizerMLX(
        action_space=_StubActionSpace(n_wp=4),
        dims_min=[-10.0, -10.0],
        dims_max=[10.0, 10.0],
        num_bins=3000,
    )
    hist_xyz = np.zeros((2, 2, 3), dtype=np.float32)
    hist_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 2, 3, 3)).copy()
    fut_xyz = np.zeros((2, 4, 3), dtype=np.float32)
    fut_rot = np.broadcast_to(np.eye(3, dtype=np.float32), (2, 4, 3, 3)).copy()
    ids = np.array(tok.encode(hist_xyz, hist_rot, fut_xyz, fut_rot))
    assert ids.shape == (2, 8)
    # (0 - (-10)) / 20 * 2999 = 1499.5 → rint 1500 (half-even)
    assert int(ids[0, 0]) == 1500
    assert np.all(ids == 1500)
    assert tok.vocab_size == 3000


def test_tokenize_future_emits_128_offset_ids():
    B, n_traj, t_hist, t_fut = 1, 1, 16, 64
    data = {
        "ego_history_xyz": np.zeros((B, n_traj, t_hist, 3), dtype=np.float32),
        "ego_history_rot": np.broadcast_to(
            np.eye(3, dtype=np.float32), (B, n_traj, t_hist, 3, 3)
        ).copy(),
        "ego_future_xyz": np.zeros((B, n_traj, t_fut, 3), dtype=np.float32),
        "ego_future_rot": np.broadcast_to(
            np.eye(3, dtype=np.float32), (B, n_traj, t_fut, 3, 3)
        ).copy(),
    }
    ids = np.array(tokenize_future_trajectory(_StubFutureTok(), data, start_idx=151669))
    assert ids.shape == (1, 128)
    assert int(ids[0, 0]) == 151669
    assert int(ids[0, -1]) == 151669 + 127


def test_fuse_replaces_history_and_future_pads():
    hist_pad, fut_pad = 100, 200
    ids = np.array(
        [[1, hist_pad, hist_pad, 2, 3, fut_pad, fut_pad, fut_pad, fut_pad, 4]],
        dtype=np.int32,
    )
    B, n_traj, t_hist, t_fut = 1, 1, 16, 2
    data = {
        "ego_history_xyz": np.zeros((B, n_traj, t_hist, 3), dtype=np.float32),
        "ego_history_rot": np.broadcast_to(
            np.eye(3, dtype=np.float32), (B, n_traj, t_hist, 3, 3)
        ).copy(),
        "ego_future_xyz": np.zeros((B, n_traj, t_fut, 3), dtype=np.float32),
        "ego_future_rot": np.broadcast_to(
            np.eye(3, dtype=np.float32), (B, n_traj, t_fut, 3, 3)
        ).copy(),
    }

    class _Hist:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            del hist_xyz, hist_rot, fut_rot
            b = int(np.asarray(fut_xyz).shape[0])
            n = int(np.asarray(fut_xyz).shape[1]) * 3
            return np.full((b, n), 7, dtype=np.int64)

    class _Fut:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            del hist_xyz, hist_rot, fut_rot
            b = int(np.asarray(fut_xyz).shape[0])
            n = int(np.asarray(fut_xyz).shape[1]) * 2
            return np.arange(n, dtype=np.int64)[None, :].repeat(b, axis=0)

    # hist encode uses history-as-future: 16*3=48 tokens, but we only have 2 pads.
    # Use a 1-step history so hist tokens = 3... still 3 != 2.
    # Build pads to match: 48 hist pads is too long for this unit test.
    # Tokenize hist with T=1 (3 xyz tokens) and 3 hist pads; future T=2 → 4 pads.
    data["ego_history_xyz"] = np.zeros((1, 1, 1, 3), dtype=np.float32)
    data["ego_history_rot"] = np.broadcast_to(
        np.eye(3, dtype=np.float32), (1, 1, 1, 3, 3)
    ).copy()
    ids = np.array(
        [[1, hist_pad, hist_pad, hist_pad, 2, fut_pad, fut_pad, fut_pad, fut_pad, 4]],
        dtype=np.int32,
    )
    out = np.array(
        fuse_traj_tokens(
            ids,
            data,
            hist_traj_tokenizer=_Hist(),
            hist_token_start_idx=10,
            traj_token_ids={"history": hist_pad, "future": fut_pad},
            traj_tokenizer=_Fut(),
            future_token_start_idx=1000,
        )
    )
    assert list(out[0]) == [1, 17, 17, 17, 2, 1000, 1001, 1002, 1003, 4]


def test_tokenize_future_rejects_3d():
    tok = _StubFutureTok()
    try:
        tokenize_future_trajectory(
            tok,
            {
                "ego_history_xyz": np.zeros((1, 16, 3)),
                "ego_history_rot": np.zeros((1, 16, 3, 3)),
                "ego_future_xyz": np.zeros((1, 64, 3)),
                "ego_future_rot": np.zeros((1, 64, 3, 3)),
            },
        )
    except ValueError as exc:
        assert "4D" in str(exc)
    else:
        raise AssertionError("expected ValueError for 3D future xyz")


def _rank4_traj(b=1, n=1, t_hist=2, t_fut=2):
    eye = np.eye(3, dtype=np.float32)
    return {
        "ego_history_xyz": np.zeros((b, n, t_hist, 3), dtype=np.float32),
        "ego_history_rot": np.broadcast_to(eye, (b, n, t_hist, 3, 3)).copy(),
        "ego_future_xyz": np.zeros((b, n, t_fut, 3), dtype=np.float32),
        "ego_future_rot": np.broadcast_to(eye, (b, n, t_fut, 3, 3)).copy(),
    }


def test_tokenize_future_requires_history_and_future_rot():
    tok = _StubFutureTok()
    data = _rank4_traj()
    try:
        tokenize_future_trajectory(tok, {**data, "ego_future_rot": None})
    except ValueError as exc:
        assert "ego_future_rot" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing future rot")
    try:
        tokenize_future_trajectory(tok, {k: data[k] for k in ("ego_future_xyz", "ego_future_rot")})
    except ValueError as exc:
        assert "history" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing history")


def test_tokenize_future_rejects_history_future_batch_mismatch():
    data = _rank4_traj(b=2, t_fut=2)
    data["ego_history_xyz"] = np.zeros((1, 1, 2, 3), dtype=np.float32)
    data["ego_history_rot"] = np.broadcast_to(
        np.eye(3, dtype=np.float32), (1, 1, 2, 3, 3)
    ).copy()
    try:
        tokenize_future_trajectory(_StubFutureTok(), data)
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("expected ValueError for history/future batch mismatch")


def test_fuse_history_only_leaves_future_pads():
    hist_pad, fut_pad = 100, 200
    ids = np.array([[1, hist_pad, hist_pad, 2, fut_pad, 3]], dtype=np.int32)
    data = {
        "ego_history_xyz": np.zeros((1, 1, 1, 3), dtype=np.float32),
        "ego_history_rot": np.broadcast_to(np.eye(3, dtype=np.float32), (1, 1, 1, 3, 3)).copy(),
    }

    class _Hist:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            del hist_xyz, hist_rot, fut_rot
            b = int(np.asarray(fut_xyz).shape[0])
            return np.full((b, 2), 7, dtype=np.int64)

    out = np.array(
        fuse_traj_tokens(
            ids,
            data,
            hist_traj_tokenizer=_Hist(),
            hist_token_start_idx=10,
            traj_token_ids={"history": hist_pad, "future": fut_pad},
        )
    )
    assert list(out[0]) == [1, 17, 17, 2, fut_pad, 3]


def test_fuse_future_requires_tokenizer_and_pad_id():
    data = _rank4_traj(t_hist=1, t_fut=1)
    ids = np.array([[1, 100, 200]], dtype=np.int32)

    class _Hist:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            del hist_xyz, hist_rot, fut_rot
            return np.zeros((1, 1), dtype=np.int64)

    try:
        fuse_traj_tokens(
            ids,
            data,
            hist_traj_tokenizer=_Hist(),
            hist_token_start_idx=0,
            traj_token_ids={"history": 100, "future": 200},
        )
    except AttributeError as exc:
        assert "traj_tokenizer" in str(exc)
    else:
        raise AssertionError("expected AttributeError when future xyz is set without tokenizer")

    try:
        fuse_traj_tokens(
            ids,
            data,
            hist_traj_tokenizer=_Hist(),
            hist_token_start_idx=0,
            traj_token_ids={"history": 100},
            traj_tokenizer=_StubFutureTok(),
            future_token_start_idx=0,
        )
    except ValueError as exc:
        assert "future" in str(exc)
    else:
        raise AssertionError("expected ValueError when future pad id is missing")


def test_fuse_without_history_returns_ids_unchanged():
    ids = np.array([[1, 2, 3]], dtype=np.int32)
    out = fuse_traj_tokens(
        ids,
        None,
        hist_traj_tokenizer=object(),
        hist_token_start_idx=0,
        traj_token_ids={"history": 100},
    )
    assert np.array_equal(np.asarray(out), ids)
