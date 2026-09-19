"""Corner order is bottom-then-top; PAI camera ids must match constants.

Weight-free. Complementary to open #24 (``xyzrot_to_corners`` is
identity-scale then translate) and #28 (``CAMERA_NAMES_TO_INDICES`` is
0–6). The first four corners are z = −h/2 (bottom) and the next four
are +h/2 (top). ``load_physical_aiavdataset`` keeps a second name→index
map; a drift sends an unknown camera to 0 (cross-left) instead of the
canonical index.
``from_alpamayo_checkpoint`` defaults ``add_special_tokens`` to True
while ``ReasoningVLAConfig`` defaults to False — complementary to #34.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from alpamayo_r1.common.constants import CAMERA_NAMES_TO_INDICES  # noqa: E402
from alpamayo_r1.geometry.coordinates import xyzrot_to_corners  # noqa: E402

_LOAD = _SRC / "alpamayo_r1" / "load_physical_aiavdataset.py"
_BASE = _SRC / "alpamayo_r1" / "models" / "base_model.py"
_SFT = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"


def _camera_name_to_index_from_source() -> dict[str, int]:
    tree = ast.parse(_LOAD.read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "load_physical_aiavdataset":
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if not stmt.targets or not isinstance(stmt.targets[0], ast.Name):
                continue
            if stmt.targets[0].id != "camera_name_to_index":
                continue
            return ast.literal_eval(stmt.value)
    raise AssertionError("camera_name_to_index not found in load_physical_aiavdataset")


def test_source_documents_bottom_then_top():
    src = (_SRC / "alpamayo_r1" / "geometry" / "coordinates.py").read_text()
    assert "The first 4 points are the bottom" in src
    assert "the next 4 are the top corners" in src
    assert "corns = dims.unsqueeze(-2) * corns" in src
    assert "corns = corns + xyz.unsqueeze(-2)" in src


def test_identity_box_bottom_then_top():
    xyz = torch.zeros(3)
    rot = torch.eye(3)
    dims = torch.tensor([2.0, 4.0, 6.0])
    corners = xyzrot_to_corners(xyz, rot, dims)
    assert tuple(corners.shape) == (8, 3)
    # Bottom four: z = -3; top four: z = +3.
    assert torch.allclose(corners[:4, 2], torch.full((4,), -3.0))
    assert torch.allclose(corners[4:, 2], torch.full((4,), 3.0))
    # Length along +X, width along +Y.
    xs, ys = corners[:, 0], corners[:, 1]
    assert set(xs.round().tolist()) == {-1.0, 1.0}
    assert set(ys.round().tolist()) == {-2.0, 2.0}


def test_translated_box_adds_center_after_scale():
    xyz = torch.tensor([10.0, 20.0, 30.0])
    rot = torch.eye(3)
    dims = torch.tensor([2.0, 2.0, 2.0])
    corners = xyzrot_to_corners(xyz, rot, dims)
    assert torch.allclose(corners.mean(0), xyz)
    assert torch.allclose(corners[0], torch.tensor([9.0, 19.0, 29.0]))


def test_pai_loader_camera_map_matches_constants():
    got = _camera_name_to_index_from_source()
    assert got == CAMERA_NAMES_TO_INDICES
    assert got["camera_cross_left_120fov"] == 0
    assert got["camera_front_tele_30fov"] == 6
    unknown = got.get("not_a_camera")
    assert unknown is None  # loader then defaults to 0 via .get(name, 0)


def test_from_alpamayo_checkpoint_defaults_add_special_tokens_true():
    sft = _SFT.read_text()
    assert '"add_special_tokens": checkpoint_config.get("add_special_tokens", True)' in sft
    base = _BASE.read_text()
    assert "add_special_tokens: bool = False" in base
    assert "if self.add_special_tokens:" in base
    assert "tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)" in base
