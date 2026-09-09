"""NVIDIA hist-token fuse and discrete-action contracts.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #17–#24,
which covered MLX fuse / last-n_traj squeeze and NVIDIA labels — not the
CUDA replace_pad / hist flatten / n_waypoints raise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_fns(rel: str, names: tuple[str, ...], extra: dict | None = None):
    path = _SRC / rel
    tree = ast.parse(path.read_text())
    keep = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and getattr(node, "name", None) in names
    ]
    missing = set(names) - {node.name for node in keep}
    if missing:
        raise AssertionError(f"{path} is missing {sorted(missing)}")
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"__name__": "extracted", **(extra or {})}
    exec(compile(mod, str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_fuse_fns(torch):
    einops = pytest.importorskip("einops")
    _ensure_src_on_path()
    try:
        return _load_fns(
            "alpamayo_r1/models/base_model.py",
            ("replace_pad_token", "tokenize_history_trajectory"),
            extra={"torch": torch, "einops": einops, "Any": object},
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA fuse helpers unavailable: {exc}")


def _load_discrete_cls(torch):
    _ensure_src_on_path()
    hyu = SimpleNamespace(instantiate=lambda cfg: None)
    try:
        return _load_fns(
            "alpamayo_r1/action_space/discrete_action_space.py",
            ("DiscreteTrajectoryTokenizer",),
            extra={"hyu": hyu, "Any": object, "ActionSpace": object, "torch": torch},
        )["DiscreteTrajectoryTokenizer"]
    except Exception as exc:
        pytest.skip(f"NVIDIA DiscreteTrajectoryTokenizer unavailable: {exc}")


def _load_unicycle_cls(torch):
    _ensure_src_on_path()
    logging = SimpleNamespace(RankedLogger=lambda *a, **k: SimpleNamespace(setLevel=lambda *_a, **_k: None))
    # traj_to_action only needs the n_waypoints check for this test; stub the rest.
    extras = {
        "torch": torch,
        "ActionSpace": torch.nn.Module,
        "dxy_theta_to_v": None,
        "dxy_theta_to_v_without_v0": None,
        "solve_xs_eq_y": None,
        "theta_smooth": None,
        "unwrap_angle": None,
        "rot_2d_to_3d": None,
        "rotation_matrix_torch": None,
        "so3_to_yaw_torch": None,
        "logging": logging,
        "logger": logging.RankedLogger(),
    }
    try:
        return _load_fns(
            "alpamayo_r1/action_space/unicycle_accel_curvature.py",
            ("UnicycleAccelCurvatureActionSpace",),
            extra=extras,
        )["UnicycleAccelCurvatureActionSpace"]
    except Exception as exc:
        pytest.skip(f"NVIDIA unicycle action space unavailable: {exc}")


def test_replace_pad_token_scatters_in_order(torch):
    fns = _load_fuse_fns(torch)
    ids = torch.tensor([[1, 9, 9, 2, 9]], dtype=torch.long)
    out = fns["replace_pad_token"](ids, torch.tensor([7, 8, 6], dtype=torch.long), 9)
    assert out.tolist() == [[1, 7, 8, 2, 6]]


def test_tokenize_history_flattens_n_traj_and_adds_offset(torch):
    fns = _load_fuse_fns(torch)

    class Tok:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            del hist_rot, fut_xyz, fut_rot
            # encode() sees flattened [B*n_traj, T, 3]; return 2 tokens per group.
            b = hist_xyz.shape[0]
            base = torch.arange(b, dtype=torch.long) * 10
            return torch.stack([base, base + 1], dim=-1)

    data = {
        "ego_history_xyz": torch.zeros((2, 3, 4, 3)),
        "ego_history_rot": torch.eye(3).expand(2, 3, 4, 3, 3).clone(),
    }
    try:
        fns["tokenize_history_trajectory"](Tok(), {"ego_history_xyz": torch.zeros((2, 4, 3))})
    except AssertionError as exc:
        assert "4D" in str(exc)
    else:
        raise AssertionError("expected AssertionError when history is not [B, n_traj, T, 3]")

    ids = fns["tokenize_history_trajectory"](Tok(), data, start_idx=100)
    # Flattened encode order is (b0n0, b0n1, b0n2, b1n0, ...); rearrange groups by batch.
    assert ids.shape == (2, 6)
    assert ids[0].tolist() == [100, 101, 110, 111, 120, 121]
    assert ids[1].tolist() == [130, 131, 140, 141, 150, 151]


def test_traj_to_action_raises_when_future_len_ne_n_waypoints(torch):
    cls = _load_unicycle_cls(torch)
    space = cls(n_waypoints=64)
    hist_xyz = torch.zeros((1, 3, 3))
    hist_rot = torch.eye(3).expand(1, 3, 3, 3).clone()
    fut_xyz = torch.zeros((1, 8, 3))
    fut_rot = torch.eye(3).expand(1, 8, 3, 3).clone()
    try:
        space.traj_to_action(hist_xyz, hist_rot, fut_xyz, fut_rot)
    except ValueError as exc:
        assert "64" in str(exc) and "8" in str(exc)
    else:
        raise AssertionError("expected ValueError when future T != n_waypoints")


def test_discrete_tokenizer_clamps_bins_and_denormalizes(torch):
    cls = _load_discrete_cls(torch)

    class StubSpace:
        def get_action_space_dims(self):
            return (2, 2)

        def traj_to_action(self, hist_xyz, hist_rot, fut_xyz, fut_rot, **kwargs):
            del hist_xyz, hist_rot, fut_xyz, fut_rot, kwargs
            return torch.tensor([[[-100.0, 100.0], [0.0, 0.0]]], dtype=torch.float32)

        def action_to_traj(self, action, hist_xyz, hist_rot):
            del hist_xyz, hist_rot
            return action, action

    tok = object.__new__(cls)
    tok.action_space = StubSpace()
    tok.dims_min = [-1.0, -1.0]
    tok.dims_max = [1.0, 1.0]
    tok.num_bins = 5
    hist_xyz = torch.zeros((1, 1, 3))
    hist_rot = torch.eye(3).expand(1, 1, 3, 3).clone()
    fut_xyz = torch.zeros((1, 2, 3))
    fut_rot = torch.eye(3).expand(1, 2, 3, 3).clone()
    ids = tok.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
    assert ids.shape == (1, 4)
    # -100 / 100 clamp to {0, 4}; 0 maps to the center bin 2.
    assert ids.tolist() == [[0, 4, 2, 2]]

    decoded, _, _ = tok.decode(hist_xyz, hist_rot, ids)
    # Bin 0 → -1, bin 4 → +1, bin 2 → 0.
    assert torch.allclose(
        decoded.reshape(1, 2, 2),
        torch.tensor([[[-1.0, 1.0], [0.0, 0.0]]]),
        atol=1e-6,
    )
