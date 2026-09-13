"""f-theta visibility culling and PAI relative stamps from the global tmin.

Weight-free. Complementary to open #28 (history/future include t0, local
origin, image stamps end at t0) — that never locked behind-camera / OOB
drops or the cross-camera ``relative = (abs - min) * 1e-6`` that VLM
temporal embeddings read.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _load_ftheta():
    path = _SRC / "alpamayo_r1" / "visualization" / "viz.py"
    if "matplotlib" not in sys.modules:
        mpl = ModuleType("matplotlib")
        mpl.pyplot = ModuleType("matplotlib.pyplot")
        sys.modules["matplotlib"] = mpl
        sys.modules["matplotlib.pyplot"] = mpl.pyplot
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = ModuleType("cv2")
    spec = importlib.util.spec_from_file_location("nv_viz_ftheta", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA viz.project_waypoints_ftheta unavailable: {exc}")
    return module.project_waypoints_ftheta


def test_ftheta_keeps_on_axis_and_drops_behind_or_oob():
    project = _load_ftheta()
    cam_rot = np.eye(3)
    cam_t = np.zeros(3)
    # pinhole-like: radius = f * theta, f=100, 64x64, principal at center.
    intr = [64.0, 64.0, 32.0, 32.0, 0.0, 100.0, 0.0, 0.0, 0.0]
    wp = np.array(
        [
            [0.0, 0.0, 2.0],  # on-axis → (cx, cy)
            [0.0, 0.0, -1.0],  # behind camera
            [50.0, 0.0, 1.0],  # far off-axis → OOB
        ]
    )
    pix = project(wp, cam_rot, cam_t, intr)
    assert pix.shape == (1, 2)
    np.testing.assert_allclose(pix[0], [32.0, 32.0], atol=1e-5)


def _stub_physical_ai_av() -> None:
    stub = sys.modules.get("physical_ai_av")
    if stub is None:
        stub = ModuleType("physical_ai_av")
        sys.modules["physical_ai_av"] = stub
    stub.PhysicalAIAVDatasetInterface = getattr(stub, "PhysicalAIAVDatasetInterface", object)


def _load_loader():
    _ensure_src_on_path()
    _stub_physical_ai_av()
    path = _SRC / "alpamayo_r1" / "load_physical_aiavdataset.py"
    spec = importlib.util.spec_from_file_location("nv_pai_loader_rel_ts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA PAI loader unavailable: {exc}")
    return module


class _Ego:
    def __call__(self, timestamps):
        ts = np.asarray(timestamps, dtype=np.int64)
        n = len(ts)
        xyz = np.zeros((n, 3))
        quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        pose = SimpleNamespace(
            translation=xyz,
            rotation=SimpleNamespace(as_quat=lambda q=quat: q),
        )
        return SimpleNamespace(pose=pose)


class _Cam:
    def __init__(self, offset_us: int) -> None:
        self.offset_us = offset_us

    def decode_images_from_timestamps(self, image_timestamps):
        ts = np.asarray(image_timestamps, dtype=np.int64) + self.offset_us
        frames = np.zeros((len(ts), 4, 4, 3), dtype=np.uint8)
        return frames, ts


def test_relative_timestamps_use_global_camera_tmin(torch):
    mod = _load_loader()

    class _Avdi:
        features = SimpleNamespace(LABELS=SimpleNamespace(EGOMOTION="egomotion"))

        def get_clip_feature(self, clip_id, feature, maybe_stream=True):
            if feature == "egomotion":
                return _Ego()
            if feature == "camera_front_wide_120fov":
                return _Cam(offset_us=0)
            return _Cam(offset_us=200_000)

    out = mod.load_physical_aiavdataset(
        "clip0",
        t0_us=5_100_000,
        avdi=_Avdi(),
        num_history_steps=2,
        num_future_steps=2,
        time_step=0.1,
        camera_features=["camera_front_wide_120fov", "camera_front_tele_30fov"],
        num_frames=2,
    )
    rel = out["relative_timestamps"].numpy()
    # Sorted by camera index: wide=1 then tele=6. Tele stamps are +0.2 s.
    assert rel.shape == (2, 2)
    np.testing.assert_allclose(rel[0], [0.0, 0.1], atol=1e-6)
    np.testing.assert_allclose(rel[1], [0.2, 0.3], atol=1e-6)


def test_non_string_camera_feature_raises():
    mod = _load_loader()

    class _Avdi:
        features = SimpleNamespace(LABELS=SimpleNamespace(EGOMOTION="egomotion"))

        def get_clip_feature(self, clip_id, feature, maybe_stream=True):
            if feature == "egomotion":
                return _Ego()
            return _Cam(offset_us=0)

    with pytest.raises(ValueError, match="Unexpected camera feature type"):
        mod.load_physical_aiavdataset(
            "clip0",
            t0_us=5_100_000,
            avdi=_Avdi(),
            num_history_steps=2,
            num_future_steps=2,
            camera_features=[object()],
            num_frames=1,
        )
