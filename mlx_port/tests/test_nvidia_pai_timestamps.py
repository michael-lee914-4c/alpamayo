"""PAI load: history includes t0, future starts at t0+dt, local-frame origin.

Weight-free. Complementary to open #26 (event-t0 window) and #27 (chunk
parse) — those never locked the history/future/image stamp algebra or the
t0-relative ego transform that every Stage-1/2 GT trajectory is built on.
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


def _stub_physical_ai_av() -> None:
    stub = sys.modules.get("physical_ai_av")
    if stub is None:
        stub = ModuleType("physical_ai_av")
        sys.modules["physical_ai_av"] = stub
    stub.PhysicalAIAVDatasetInterface = getattr(stub, "PhysicalAIAVDatasetInterface", object)
    if not hasattr(stub, "egomotion"):
        stub.egomotion = SimpleNamespace()
    if not hasattr(stub, "video"):
        stub.video = SimpleNamespace()
    if "physical_ai_av.dataset" not in sys.modules:
        dataset = ModuleType("physical_ai_av.dataset")

        class Features:
            def __init__(self, df):
                self.features_df = df

        dataset.Features = Features
        stub.dataset = dataset
        sys.modules["physical_ai_av.dataset"] = dataset


def _load_loader():
    _ensure_src_on_path()
    _stub_physical_ai_av()
    path = _SRC / "alpamayo_r1" / "load_physical_aiavdataset.py"
    spec = importlib.util.spec_from_file_location("nv_pai_loader_ts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA PAI loader unavailable: {exc}")
    return module


class _Ego:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, timestamps):
        ts = np.asarray(timestamps, dtype=np.int64)
        self.calls.append(ts.copy())
        n = len(ts)
        # 1 m/s along +X from time 0, identity orientation (scipy xyzw).
        xyz = np.stack([ts.astype(np.float64) * 1e-6, np.zeros(n), np.zeros(n)], axis=-1)
        quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        pose = SimpleNamespace(
            translation=xyz,
            rotation=SimpleNamespace(as_quat=lambda q=quat: q),
        )
        return SimpleNamespace(pose=pose)


class _Cam:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def decode_images_from_timestamps(self, image_timestamps):
        ts = np.asarray(image_timestamps, dtype=np.int64)
        self.calls.append(ts.copy())
        frames = np.zeros((len(ts), 8, 8, 3), dtype=np.uint8)
        return frames, ts


def _avdi(ego: _Ego, cams: dict[str, _Cam] | None = None):
    cams = cams or {}

    class _Avdi:
        features = SimpleNamespace(LABELS=SimpleNamespace(EGOMOTION="egomotion"))

        def get_clip_feature(self, clip_id, feature, maybe_stream=True):
            if feature == "egomotion":
                return ego
            if feature in cams:
                return cams[feature]
            return next(iter(cams.values())) if cams else _Cam()

    return _Avdi()


def test_history_includes_t0_and_future_starts_one_step_later():
    mod = _load_loader()
    ego = _Ego()
    cam = _Cam()
    t0_us = 5_100_000
    dt_us = 100_000
    n_hist, n_fut, n_frames = 16, 64, 4
    out = mod.load_physical_aiavdataset(
        "clip0",
        t0_us=t0_us,
        avdi=_avdi(ego, {"camera_front_wide_120fov": cam}),
        num_history_steps=n_hist,
        num_future_steps=n_fut,
        time_step=0.1,
        camera_features=["camera_front_wide_120fov"],
        num_frames=n_frames,
    )

    hist_ts, fut_ts = ego.calls
    assert hist_ts.tolist() == [t0_us + i * dt_us for i in range(-(n_hist - 1), 1)]
    assert hist_ts[-1] == t0_us
    assert fut_ts.tolist() == [t0_us + i * dt_us for i in range(1, n_fut + 1)]
    assert fut_ts[0] == t0_us + dt_us
    assert cam.calls[0].tolist() == [t0_us - (n_frames - 1 - i) * dt_us for i in range(n_frames)]
    assert cam.calls[0][-1] == t0_us
    assert out["t0_us"] == t0_us


def test_t0_must_exceed_history_span():
    mod = _load_loader()
    with pytest.raises(ValueError, match="history time range"):
        mod.load_physical_aiavdataset(
            "clip0",
            t0_us=1_600_000,
            avdi=_avdi(_Ego(), {"cam": _Cam()}),
            num_history_steps=16,
            num_future_steps=4,
            time_step=0.1,
            camera_features=["cam"],
            num_frames=1,
        )


def test_local_frame_puts_t0_at_origin_and_sorts_camera_ids(torch):
    mod = _load_loader()
    ego = _Ego()
    cams = {
        "camera_front_tele_30fov": _Cam(),
        "camera_cross_left_120fov": _Cam(),
    }
    out = mod.load_physical_aiavdataset(
        "clip0",
        t0_us=5_100_000,
        avdi=_avdi(ego, cams),
        num_history_steps=4,
        num_future_steps=3,
        time_step=0.1,
        camera_features=["camera_front_tele_30fov", "camera_cross_left_120fov"],
        num_frames=2,
    )

    hist = out["ego_history_xyz"].numpy()[0, 0]
    fut = out["ego_future_xyz"].numpy()[0, 0]
    np.testing.assert_allclose(hist[-1], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(fut[0], [0.1, 0.0, 0.0], atol=1e-6)
    # Unknown names default to 0; known names match CAMERA_NAMES_TO_INDICES.
    assert out["camera_indices"].tolist() == [0, 6]
    assert tuple(out["image_frames"].shape) == (2, 2, 3, 8, 8)


def test_unknown_camera_name_defaults_to_index_zero(torch):
    mod = _load_loader()
    out = mod.load_physical_aiavdataset(
        "clip0",
        t0_us=5_100_000,
        avdi=_avdi(_Ego(), {"not_a_camera": _Cam()}),
        num_history_steps=2,
        num_future_steps=2,
        camera_features=["not_a_camera"],
        num_frames=1,
    )
    assert out["camera_indices"].tolist() == [0]


def test_camera_index_map_matches_loader_and_display_names():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.common.constants import (
            CAMERA_INDICES_TO_DISPLAY_NAMES,
            CAMERA_NAMES_TO_INDICES,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA camera constants unavailable: {exc}")

    assert CAMERA_NAMES_TO_INDICES == {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }
    assert CAMERA_INDICES_TO_DISPLAY_NAMES[1] == "Front camera"
    assert CAMERA_INDICES_TO_DISPLAY_NAMES[6] == "Front telephoto camera"
    assert set(CAMERA_NAMES_TO_INDICES.values()) == set(range(7))
