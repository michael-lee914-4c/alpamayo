"""PAI clip event-t0 safe-margin filter and train/eval frac clamps.

Weight-free. Complementary to open #19 clip-pool listing — this locks the
1.6s / 6.4s event window and the split helper that refuses empty sides.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_pai_utils():
    _ensure_src_on_path()
    if "physical_ai_av" not in sys.modules:
        stub = ModuleType("physical_ai_av")
        stub.egomotion = SimpleNamespace()
        stub.video = SimpleNamespace()
        dataset = ModuleType("physical_ai_av.dataset")
        dataset.Features = object
        stub.dataset = dataset
        sys.modules["physical_ai_av"] = stub
        sys.modules["physical_ai_av.dataset"] = dataset
    path = _SRC / "alpamayo_r1" / "data" / "pai_utils.py"
    spec = importlib.util.spec_from_file_location("nv_pai_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA pai_utils unavailable: {exc}")
    return module


def test_event_t0_filter_keeps_margin_window_and_drops_empty():
    mod = _load_pai_utils()
    obj = object.__new__(mod.PhysicalAIAVDatasetLocalInterface)
    obj.start_safe_margin_seconds = 1.6
    obj.end_safe_margin_seconds = 6.4
    # 1.5s is before the start margin; 1.6s is kept; 2.0s needs 8.4s of clip.
    obj.clip_index = pd.DataFrame(
        {
            "event_t0s": [
                np.array([1_500_000], dtype=np.int64),
                np.array([1_600_000], dtype=np.int64),
                np.array([2_000_000], dtype=np.int64),
                np.array([2_000_000], dtype=np.int64),
                np.array([], dtype=np.int64),
            ],
            "end_timestamp": [
                20_000_000,
                20_000_000,
                8_300_000,  # 2.0s + 6.4s = 8.4s > 8.3s → drop
                8_400_000,  # exactly fits
                20_000_000,
            ],
        },
        index=["early", "on_start", "short_tail", "fits", "empty"],
    )
    obj.filter_clips_by_event_t0s()
    assert list(obj.clip_index.index) == ["on_start", "fits"]
    np.testing.assert_array_equal(obj.clip_index.loc["on_start", "event_t0s"], [1_600_000])
    np.testing.assert_array_equal(obj.clip_index.loc["fits", "event_t0s"], [2_000_000])


def test_event_t0_filter_without_end_ts_uses_start_margin_only():
    mod = _load_pai_utils()
    obj = object.__new__(mod.PhysicalAIAVDatasetLocalInterface)
    obj.start_safe_margin_seconds = 1.6
    obj.end_safe_margin_seconds = 6.4
    obj.clip_index = pd.DataFrame(
        {
            "event_t0s": [
                np.array([1_599_999, 1_600_000], dtype=np.int64),
            ]
        },
        index=["mixed"],
    )
    obj.filter_clips_by_event_t0s()
    np.testing.assert_array_equal(obj.clip_index.loc["mixed", "event_t0s"], [1_600_000])


def test_event_t0_filter_is_noop_without_column():
    mod = _load_pai_utils()
    obj = object.__new__(mod.PhysicalAIAVDatasetLocalInterface)
    obj.start_safe_margin_seconds = 1.6
    obj.end_safe_margin_seconds = 6.4
    obj.clip_index = pd.DataFrame({"chunk": [0]}, index=["x"])
    obj.filter_clips_by_event_t0s()
    assert list(obj.clip_index.index) == ["x"]


def test_split_train_eval_clamps_tiny_frac_and_rejects_edges():
    from mlx_port.gt_eval import split_train_eval

    ids = [f"c{i}" for i in range(10)]
    train, ev = split_train_eval(ids, seed=0, train_frac=0.01)
    assert len(train) == 1
    assert len(ev) == 9
    assert not (set(train) & set(ev))
    train_hi, ev_hi = split_train_eval(ids, seed=0, train_frac=0.99)
    assert len(train_hi) == 9
    assert len(ev_hi) == 1
    try:
        split_train_eval(ids, seed=0, train_frac=0.0)
    except ValueError as exc:
        assert "train_frac" in str(exc)
    else:
        raise AssertionError("expected ValueError for train_frac=0")
    try:
        split_train_eval(ids, seed=0, train_frac=1.0)
    except ValueError as exc:
        assert "train_frac" in str(exc)
    else:
        raise AssertionError("expected ValueError for train_frac=1")
