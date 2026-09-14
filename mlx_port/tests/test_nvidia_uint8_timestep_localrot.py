"""NVIDIA uint8 /255, action-in last timestep, local-frame rotation.

Weight-free. Complementary to open #25 (camera sort) / #26 (2D↔3D) /
#28 (local-frame *origin* + t0 stamps) / #29 (Fourier log-space) —
those never locked (1) uint8 images being divided by 255 with
``do_rescale=False``, (2) ``PerWaypointActionInProjV2`` reading only
``timesteps[..., -1]``, or (3) ``R_t0^{-1}`` conjugating world yaw.
A 255× vision scale or t0-yaw skip would look like a model regression.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_qwen_uint8_divides_by_255_float_passthrough(torch):
    src = (_SRC / "alpamayo_r1" / "processor" / "qwen_processor.py").read_text()
    assert "images.float() / 255.0" in src
    assert "do_rescale=False" in src
    assert "images.flatten(0, 1)" in src

    raw = torch.tensor([0, 128, 255], dtype=torch.uint8)
    scaled = raw.float() / 255.0
    assert float(scaled[0].item()) == 0.0
    assert float(scaled[-1].item()) == pytest.approx(1.0, abs=1e-6)
    assert 0.49 < float(scaled[1].item()) < 0.51
    already = torch.tensor([0.25, 1.0], dtype=torch.float32)
    assert torch.allclose(already.float(), already)
    # A missing uint8 branch would leave 0..255 as the VLM input.
    assert float(raw.float().max().item()) == 255.0


def test_action_in_proj_encodes_last_timestep_dim_only(torch):
    _ensure_src_on_path()
    try:
        from alpamayo_r1.models.action_in_proj import PerWaypointActionInProjV2
    except Exception as exc:
        pytest.skip(f"NVIDIA PerWaypointActionInProjV2 unavailable: {exc}")

    torch.manual_seed(0)
    proj = PerWaypointActionInProjV2(
        in_dims=[2],
        out_dim=8,
        num_enc_layers=1,
        hidden_size=16,
        num_fourier_feats=4,
        max_freq=10.0,
    )
    proj.eval()
    x = torch.zeros(2, 5, 2)
    t_last = torch.zeros(2, 3, 4)
    t_last[..., -1] = 0.75
    t_first = torch.zeros(2, 3, 4)
    t_first[..., 0] = 0.75
    t_only_last = torch.zeros(2, 3, 4)
    t_only_last[..., -1] = 0.75
    with torch.no_grad():
        out_last = proj(x, t_last)
        out_first = proj(x, t_first)
        out_only = proj(x, t_only_last)
    assert out_last.shape == (2, 5, 8)
    assert torch.allclose(out_last, out_only, atol=1e-5)
    # Leading timestep dims must be ignored; a [..., 0] swap would match first.
    assert not torch.allclose(out_last, out_first, atol=1e-4)


def test_local_frame_conjugates_t0_yaw(torch):
    spt = pytest.importorskip("scipy.spatial.transform")
    src = (_SRC / "alpamayo_r1" / "load_physical_aiavdataset.py").read_text()
    assert "t0_rot_inv.apply(ego_history_xyz - t0_xyz)" in src
    assert "t0_rot_inv.apply(ego_future_xyz - t0_xyz)" in src
    assert "t0_rot_inv * spt.Rotation.from_quat" in src
    assert "ego_history_xyz[-1]" in src

    # t0 at origin with yaw = +π/2 (world +X → world +Y). Inverse maps +Y → +X.
    t0_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    t0_rot = spt.Rotation.from_euler("z", math.pi / 2)
    t0_rot_inv = t0_rot.inv()
    world = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float64)
    local = t0_rot_inv.apply(world - t0_xyz)
    np.testing.assert_allclose(local[0], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(local[1], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(local[2], [2.0, 0.0, 0.0], atol=1e-7)
    # t0 orientation in the local frame is identity (conjugated with itself).
    local_rot = (t0_rot_inv * t0_rot).as_matrix()
    np.testing.assert_allclose(local_rot, np.eye(3), atol=1e-7)
    # Translation-only (no rotation) would leave +Y as +Y.
    raw = world - t0_xyz
    assert not np.allclose(raw[1], [1.0, 0.0, 0.0], atol=1e-5)
