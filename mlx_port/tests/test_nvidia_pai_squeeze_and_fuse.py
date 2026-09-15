"""NVIDIA PAI squeeze(0) and hist-only vs future fuse mixin.

Weight-free. Complementary to open #28 (local-frame + ``(1, 1, T, 3)``
load layout) and #30 (future tokenize) — ``PAIDataset`` ``squeeze(0)``
keeps the ``n_traj`` dim, ``include_extr_intr`` writes LWH plus
rear-axle/length, the *base* fuse mixin is hist-only, and
``DistanceMetrics`` uses dummy vehicle ``(4, 3, 2)``. A full
``squeeze()`` would drop ``n_traj`` and fail the 4D tokenize asserts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai.py"
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_METRIC = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_api.py"
_CONST = _REPO / "src" / "alpamayo_r1" / "common" / "constants.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_pai_squeeze0_keeps_n_traj_dim(torch):
    src = _PAI.read_text()
    assert "if key.startswith(\"ego_\"):" in src
    assert "sample_data[key] = sample_data[key].squeeze(0)" in src
    assert ".squeeze()" not in src

    # load_physical_aiavdataset writes (1, 1, T, 3); PAI squeezes batch only.
    loaded = {
        "ego_history_xyz": torch.zeros(1, 1, 16, 3),
        "ego_future_xyz": torch.zeros(1, 1, 64, 3),
        "image_frames": torch.zeros(4, 4, 3, 8, 8),
    }
    for key in list(loaded):
        if key.startswith("ego_"):
            loaded[key] = loaded[key].squeeze(0)
    assert loaded["ego_history_xyz"].shape == (1, 16, 3)
    assert loaded["ego_future_xyz"].shape == (1, 64, 3)
    # A bare squeeze() would drop n_traj and fail ndim==4 tokenize.
    dropped = torch.zeros(1, 1, 16, 3).squeeze()
    assert dropped.shape == (16, 3)
    assert dropped.shape != loaded["ego_history_xyz"].shape


def test_include_extr_intr_lwh_and_rear_axle_offset(torch):
    src = _PAI.read_text()
    assert "[vehicle_dimensions.length, vehicle_dimensions.width, vehicle_dimensions.height]" in src
    assert "vehicle_dimensions.rear_axle_to_bbox_center / vehicle_dimensions.length" in src

    vehicle = SimpleNamespace(length=4.8, width=1.9, height=1.6, rear_axle_to_bbox_center=1.2)
    ego_lwh = torch.tensor([vehicle.length, vehicle.width, vehicle.height])
    offset = torch.tensor(vehicle.rear_axle_to_bbox_center / vehicle.length)
    assert ego_lwh.tolist() == pytest.approx([4.8, 1.9, 1.6])
    assert float(offset.item()) == pytest.approx(1.2 / 4.8)
    # Using width as the denominator would silently retune every corner metric.
    wrong = vehicle.rear_axle_to_bbox_center / vehicle.width
    assert wrong != float(offset.item())


def test_base_fuse_is_hist_only_future_mixin_fuses_both():
    base = _BASE.read_text()
    future = _SFT_BASE.read_text()
    # Base mixin tokenizes history and returns; it never calls future tokenize.
    hist_fuse = base.split("class TrajectoryFusionMixin:")[1].split("class ReasoningVLAConfig")[0]
    assert "tokenize_history_trajectory" in hist_fuse
    assert "tokenize_future_trajectory" not in hist_fuse
    assert "return input_ids" in hist_fuse

    mixin = future.split("class TrajectoryFusionWithFutureMixin")[1].split(
        "class ReasoningVLAOutput"
    )[0]
    assert "tokenize_history_trajectory" in mixin
    assert "tokenize_future_trajectory" in mixin
    assert "traj_token_ids[\"future\"]" in mixin
    assert "if has_future:" in mixin


def test_distance_metrics_dummy_vehicle_is_4_3_2():
    metric = _METRIC.read_text()
    const = _CONST.read_text()
    assert "EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)" in metric
    assert "EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)" in const
    assert "torch.tensor(EGO_VEHICLE_LWH" in metric
    # ReasoningSampler never requests CoC extras (prefix-only pred_xyz/rot).
    assert "return_extra=False" in metric
