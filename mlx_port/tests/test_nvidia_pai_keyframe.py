"""PAI keyframe t0: event_t0s[i] vs DEFAULT_T0_US=5.1s.

Weight-free. Complementary to open #26 (event-t0 window) and #27 (chunk
parse) — those never locked ``get_clip_key_frame`` indexing or the
``use_default_keyframe`` switch that Stage-1/2 smokes rely on (t0=5.1s).
"""

from __future__ import annotations

import importlib.util
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


def _stub_physical_ai_av() -> None:
    stub = sys.modules.get("physical_ai_av")
    if stub is None:
        stub = ModuleType("physical_ai_av")
        sys.modules["physical_ai_av"] = stub
    if not hasattr(stub, "egomotion"):
        stub.egomotion = SimpleNamespace()
    if not hasattr(stub, "video"):
        stub.video = SimpleNamespace()
    if getattr(stub, "PhysicalAIAVDatasetInterface", None) is None:
        stub.PhysicalAIAVDatasetInterface = object
    if "physical_ai_av.dataset" not in sys.modules:
        dataset = ModuleType("physical_ai_av.dataset")

        class Features:
            def __init__(self, df):
                self.features_df = df

        dataset.Features = Features
        stub.dataset = dataset
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
    spec = importlib.util.spec_from_file_location("nv_pai_utils_keyframe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA pai_utils unavailable: {exc}")
    return module


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
    if "alpamayo_r1.data.pai_utils" not in sys.modules:
        sys.modules["alpamayo_r1.data.pai_utils"] = _load_pai_utils()
    load_mod = ModuleType("alpamayo_r1.load_physical_aiavdataset")
    load_mod.load_physical_aiavdataset = lambda *a, **k: {}
    sys.modules["alpamayo_r1.load_physical_aiavdataset"] = load_mod
    path = _SRC / "alpamayo_r1" / "data" / "pai.py"
    spec = importlib.util.spec_from_file_location("nv_pai_dataset_keyframe", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA PAIDataset unavailable: {exc}")
    return module


def test_get_clip_key_frame_indexes_event_t0s():
    mod = _load_pai_utils()
    obj = object.__new__(mod.PhysicalAIAVDatasetLocalInterface)
    obj.clip_index = pd.DataFrame(
        {"event_t0s": [np.array([1_600_000, 2_000_000, 3_000_000], dtype=np.int64)]},
        index=["c0"],
    )
    assert int(obj.get_clip_key_frame("c0")) == 1_600_000
    assert int(obj.get_clip_key_frame("c0", 1)) == 2_000_000
    assert int(obj.get_clip_key_frame("c0", 2)) == 3_000_000


def test_default_t0_us_is_5_1_seconds():
    mod = _load_pai_dataset()
    assert mod.PAIDataset.DEFAULT_T0_US == 5_100_000


def test_use_default_keyframe_overrides_clip_event():
    torch = pytest.importorskip("torch")
    mod = _load_pai_dataset()

    def _make_ds(*, use_default: bool):
        ds = object.__new__(mod.PAIDataset)
        ds.clip_ids = ["c0"]
        ds.use_default_keyframe = use_default
        ds.DEFAULT_T0_US = mod.PAIDataset.DEFAULT_T0_US
        ds.avdi = SimpleNamespace(get_clip_key_frame=lambda clip_id: 2_000_000)
        ds.reshape_tensors_for_rl = False
        ds.include_extr_intr = False
        ds.vla_preprocess_func = None
        ds.num_history_steps = 16
        ds.num_future_steps = 64
        ds.time_step = 0.1
        return ds

    captured: list[int] = []

    def _fake_load(clip_id, t0_us, **kwargs):
        captured.append(int(t0_us))
        return {
            "ego_history_xyz": torch.zeros(1, 16, 3),
            "image_frames": torch.zeros(1, 1, 3, 4, 4),
        }

    with patch.object(mod, "load_physical_aiavdataset", side_effect=_fake_load):
        _make_ds(use_default=True)[0]
        _make_ds(use_default=False)[0]

    assert captured == [5_100_000, 2_000_000]
