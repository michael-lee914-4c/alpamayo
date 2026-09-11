"""PAI chunk-id parse (end-exclusive) and RL camera×frame flatten.

Weight-free. Complementary to open #26 (event-t0 window) and #25 (camera
sort before flatten) — this locks ``"0-3" → [0,1,2]``, clip-index filtering
by chunk, and the RL reshape that vLLM packing depends on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
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
    if "physical_ai_av" in sys.modules:
        return
    stub = ModuleType("physical_ai_av")
    stub.egomotion = SimpleNamespace()
    stub.video = SimpleNamespace()
    dataset = ModuleType("physical_ai_av.dataset")

    class Features:
        def __init__(self, df):
            self.features_df = df

    dataset.Features = Features
    stub.dataset = dataset
    sys.modules["physical_ai_av"] = stub
    sys.modules["physical_ai_av.dataset"] = dataset


def _stub_nv_logging() -> None:
    if "alpamayo_r1.common.logging" in sys.modules:
        return
    logging_mod = ModuleType("alpamayo_r1.common.logging")

    class RankedLogger:
        def __init__(self, *args, **kwargs):
            pass

        def setLevel(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    logging_mod.RankedLogger = RankedLogger
    sys.modules["alpamayo_r1.common.logging"] = logging_mod


def _load_pai_utils():
    _ensure_src_on_path()
    _stub_physical_ai_av()
    _stub_nv_logging()
    path = _SRC / "alpamayo_r1" / "data" / "pai_utils.py"
    spec = importlib.util.spec_from_file_location("nv_pai_utils_chunk", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA pai_utils unavailable: {exc}")
    return module


def _write_pai_meta(tmp_path: Path) -> None:
    features = pd.DataFrame(
        {"clip_files_in_zip": [json.dumps([])]},
        index=pd.Index(["egomotion"], name="feature"),
    )
    features.to_csv(tmp_path / "features.csv")
    clip_index = pd.DataFrame(
        {
            "chunk": [0, 1, 2, 1],
            "event_t0s": [
                np.array([2_000_000], dtype=np.int64),
                np.array([2_000_000], dtype=np.int64),
                np.array([2_000_000], dtype=np.int64),
                np.array([2_000_000], dtype=np.int64),
            ],
            "end_timestamp": [20_000_000] * 4,
        },
        index=["a", "b", "c", "d"],
    )
    clip_index.to_parquet(tmp_path / "clip_index.parquet")
    (tmp_path / "metadata").mkdir()
    presence = pd.DataFrame({"cam": [True] * 4}, index=["a", "b", "c", "d"])
    presence.to_parquet(tmp_path / "metadata" / "feature_presence.parquet")


def test_chunk_ids_range_string_is_end_exclusive(tmp_path):
    mod = _load_pai_utils()
    _write_pai_meta(tmp_path)
    avdi = mod.PhysicalAIAVDatasetLocalInterface(local_dir=tmp_path, chunk_ids="0-3")
    assert avdi.chunk_ids == [0, 1, 2]
    # End-exclusive: chunk 3 would be dropped; clips a/b/c/d are chunks 0/1/2/1.
    assert set(avdi.get_all_clip_ids()) == {"a", "b", "c", "d"}


def test_chunk_ids_int_and_list_and_get_all_clip_ids(tmp_path):
    mod = _load_pai_utils()
    _write_pai_meta(tmp_path)
    one = mod.PhysicalAIAVDatasetLocalInterface(local_dir=tmp_path, chunk_ids=1)
    assert one.chunk_ids == [1]
    assert set(one.get_all_clip_ids()) == {"b", "d"}

    listed = mod.PhysicalAIAVDatasetLocalInterface(local_dir=tmp_path, chunk_ids=[0, 2])
    assert listed.chunk_ids == [0, 2]
    assert set(listed.get_all_clip_ids()) == {"a", "c"}

    all_chunks = mod.PhysicalAIAVDatasetLocalInterface(local_dir=tmp_path, chunk_ids=None)
    assert all_chunks.chunk_ids is None
    assert set(all_chunks.get_all_clip_ids()) == {"a", "b", "c", "d"}


def test_get_all_clip_ids_filters_index_without_init():
    mod = _load_pai_utils()
    obj = object.__new__(mod.PhysicalAIAVDatasetLocalInterface)
    obj.chunk_ids = [1]
    obj.clip_index = pd.DataFrame({"chunk": [1, 2, 1]}, index=["a", "b", "c"])
    assert obj.get_all_clip_ids() == ["a", "c"]
    obj.chunk_ids = None
    assert obj.get_all_clip_ids() == ["a", "b", "c"]


def _load_pai_dataset():
    _ensure_src_on_path()
    _stub_physical_ai_av()
    _stub_nv_logging()
    hydra = ModuleType("hydra")
    hydra_utils = ModuleType("hydra.utils")
    hydra_utils.instantiate = lambda *a, **k: None
    omegaconf = ModuleType("omegaconf")
    omegaconf.OmegaConf = SimpleNamespace(create=lambda x: x)
    sys.modules.setdefault("hydra", hydra)
    sys.modules.setdefault("hydra.utils", hydra_utils)
    sys.modules.setdefault("omegaconf", omegaconf)
    # Keep pai_utils from importing the real physical_ai_av package.
    if "alpamayo_r1.data.pai_utils" not in sys.modules:
        sys.modules["alpamayo_r1.data.pai_utils"] = _load_pai_utils()
    load_mod = ModuleType("alpamayo_r1.load_physical_aiavdataset")
    load_mod.load_physical_aiavdataset = lambda *a, **k: {}
    sys.modules["alpamayo_r1.load_physical_aiavdataset"] = load_mod
    path = _SRC / "alpamayo_r1" / "data" / "pai.py"
    spec = importlib.util.spec_from_file_location("nv_pai_dataset", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA PAIDataset unavailable: {exc}")
    return module


def test_rl_reshape_flattens_camera_then_frame_and_leaves_ego(torch):
    mod = _load_pai_dataset()
    n_cam, n_frame, c, h, w = 2, 3, 3, 4, 4
    frames = torch.arange(n_cam * n_frame * c * h * w, dtype=torch.float32).reshape(
        n_cam, n_frame, c, h, w
    )
    sample = {
        "image_frames": frames.clone(),
        "camera_indices": torch.tensor([7, 9]),
        "absolute_timestamps": torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64),
        "relative_timestamps": torch.tensor([[0.0, 0.1, 0.2], [0.0, 0.1, 0.2]]),
        "ego_history_xyz": torch.zeros(1, 16, 3),
        "ego_future_xyz": torch.zeros(1, 64, 3),
    }
    ds = object.__new__(mod.PAIDataset)
    ds.clip_ids = ["c0"]
    ds.use_default_keyframe = True
    ds.DEFAULT_T0_US = 5_100_000
    ds.reshape_tensors_for_rl = True
    ds.include_extr_intr = False
    ds.vla_preprocess_func = None
    ds.num_history_steps = 16
    ds.num_future_steps = 64
    ds.time_step = 0.1
    ds.avdi = SimpleNamespace()

    with patch.object(mod, "load_physical_aiavdataset", return_value=dict(sample)):
        out = ds[0]

    assert tuple(out["image_frames"].shape) == (6, 1, c, h, w)
    assert out["camera_indices"].tolist() == [7, 7, 7, 9, 9, 9]
    assert out["absolute_timestamps"].tolist() == [10, 11, 12, 20, 21, 22]
    assert out["relative_timestamps"].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.0, 0.1, 0.2])
    # Ego keys are squeezed (leading 1) but not camera-flattened.
    assert tuple(out["ego_history_xyz"].shape) == (16, 3)
    assert tuple(out["ego_future_xyz"].shape) == (64, 3)
    np.testing.assert_array_equal(
        out["image_frames"].squeeze(1).numpy(),
        frames.reshape(6, c, h, w).numpy(),
    )
