"""NVIDIA ``tokenize_future_trajectory``: fut≠hist, flatten, offset.

Weight-free. Complementary to open #25 (hist tokenize passes hist as
``fut_xyz`` and flattens ``n_traj``) — Stage-1/2 SFT future fuse must
encode the *future* waypoints, not replay that hist-as-fut trick.
A silent copy-paste would CE-train history tokens in the future slots.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    src = str(_REPO / "src")
    for path in (str(_REPO), src):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_future_fn(torch):
    einops = pytest.importorskip("einops")
    _ensure_src_on_path()
    # File lives under finetune/, not src/.
    path = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
    tree = ast.parse(path.read_text())
    keep = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "tokenize_future_trajectory"
    ]
    if not keep:
        pytest.skip("tokenize_future_trajectory missing from sft_base_model.py")
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"__name__": "extracted", "torch": torch, "einops": einops, "Any": object, "Mapping": dict}
    try:
        exec(compile(mod, str(path), "exec"), ns)
    except Exception as exc:
        pytest.skip(f"NVIDIA tokenize_future_trajectory unavailable: {exc}")
    return ns["tokenize_future_trajectory"]


def test_future_tokenize_encodes_fut_not_hist_and_adds_offset(torch):
    tokenize_future_trajectory = _load_future_fn(torch)
    seen = {}

    class Tok:
        def encode(self, hist_xyz, hist_rot, fut_xyz, fut_rot):
            seen["hist_xyz"] = hist_xyz.detach().clone()
            seen["fut_xyz"] = fut_xyz.detach().clone()
            del hist_rot, fut_rot
            # Distinct marker per flattened row so rearrange is testable.
            b = fut_xyz.shape[0]
            base = torch.arange(b, dtype=torch.long) * 10
            return torch.stack([base, base + 1], dim=-1)

    hist = torch.zeros((2, 3, 4, 3))
    hist[..., 0] = -1.0
    fut = torch.zeros((2, 3, 5, 3))
    fut[..., 0] = 7.0
    data = {
        "ego_history_xyz": hist,
        "ego_history_rot": torch.eye(3).expand(2, 3, 4, 3, 3).clone(),
        "ego_future_xyz": fut,
        "ego_future_rot": torch.eye(3).expand(2, 3, 5, 3, 3).clone(),
    }
    ids = tokenize_future_trajectory(Tok(), data, start_idx=200)
    assert seen["fut_xyz"].shape == (6, 5, 3)
    assert seen["hist_xyz"].shape == (6, 4, 3)
    assert float(seen["fut_xyz"][..., 0].unique().item()) == 7.0
    assert float(seen["hist_xyz"][..., 0].unique().item()) == -1.0
    # Flatten order is (b0n0, b0n1, b0n2, b1n0, ...); +200 offset.
    assert ids.shape == (2, 6)
    assert ids[0].tolist() == [200, 201, 210, 211, 220, 221]
    assert ids[1].tolist() == [230, 231, 240, 241, 250, 251]


def test_future_tokenize_rejects_rank3_future(torch):
    tokenize_future_trajectory = _load_future_fn(torch)

    class Tok:
        def encode(self, **kwargs):
            raise AssertionError("encode must not run on a rank-3 future")

    data = {
        "ego_history_xyz": torch.zeros((2, 4, 3)),
        "ego_history_rot": torch.eye(3).expand(2, 4, 3, 3).clone(),
        "ego_future_xyz": torch.zeros((2, 5, 3)),
        "ego_future_rot": torch.eye(3).expand(2, 5, 3, 3).clone(),
    }
    try:
        tokenize_future_trajectory(Tok(), data)
    except AssertionError as exc:
        assert "4D" in str(exc)
    else:
        raise AssertionError("expected AssertionError when future is not [B, n_traj, T, 3]")
