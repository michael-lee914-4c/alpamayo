"""NVIDIA PAI chunk fallback, eval numel, and comfort K-then-N means.

Weight-free. Complementary to open #27 (end-exclusive ``\"a-b\"``),
#28 (double-call preprocess is *not* locked), #32 (``squeeze(0)`` /
``include_extr_intr``), and comfort body-frame / exclusive-bound tests.
Invalid ``chunk_ids`` log and leave ``None`` (full index). Comfort
averages ``_within_bound`` over K then N. ``evaluate_hf`` only folds
``metric/`` keys by ``sum / numel``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PAI_UTILS = _REPO / "src" / "alpamayo_r1" / "data" / "pai_utils.py"
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai.py"
_EVAL = _REPO / "finetune" / "sft" / "evaluate_hf.py"
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"
_VIZ = _REPO / "src" / "alpamayo_r1" / "visualization" / "viz.py"
_HELPER = _REPO / "src" / "alpamayo_r1" / "helper.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _parse_chunk_ids(chunk_ids):
    # Mirrors PhysicalAIAVDatasetLocalInterface.__init__ exactly.
    parsed = None
    if chunk_ids is not None:
        if isinstance(chunk_ids, str) and "-" in chunk_ids:
            chunk_start = int(chunk_ids.split("-")[0])
            chunk_end = int(chunk_ids.split("-")[1])
            parsed = list(range(chunk_start, chunk_end))
        elif isinstance(chunk_ids, (list, tuple, Iterable)):
            parsed = list(chunk_ids)
        elif isinstance(chunk_ids, int):
            parsed = [chunk_ids]
        else:
            parsed = None
    return parsed


def test_invalid_chunk_ids_leave_filter_none_or_char_split():
    src = _PAI_UTILS.read_text()
    assert 'isinstance(chunk_ids, str) and "-" in chunk_ids' in src
    assert 'elif isinstance(chunk_ids, (list, tuple, Iterable)):' in src
    assert "logger.error(f\"Invalid chunk_ids: {chunk_ids} {type(chunk_ids)}\")" in src
    assert "self.chunk_ids = None" in src
    # Range strings stay end-exclusive.
    assert _parse_chunk_ids("0-3") == [0, 1, 2]
    assert _parse_chunk_ids(4) == [4]
    # A comma list is a str *and* an Iterable, so it becomes characters.
    assert _parse_chunk_ids("0,1") == ["0", ",", "1"]
    # float is the true "invalid" path: log + leave None → load every chunk.
    assert _parse_chunk_ids(1.5) is None


def test_missing_feature_returns_none_not_raise():
    src = _PAI_UTILS.read_text()
    assert "if feature not in self.features.features_df.index:" in src
    assert "return None" in src
    pai = _PAI.read_text()
    # include_extr_intr dereferences .length with no None guard.
    assert 'sample_data["ego_lwh"] = torch.tensor(' in pai
    assert "vehicle_dimensions.length" in pai
    assert "vehicle_dimensions.width" in pai
    assert "vehicle_dimensions.height" in pai


def test_preprocess_runs_again_after_rl_reshape():
    src = _PAI.read_text()
    first = src.index("if self.vla_preprocess_func is not None:")
    reshape = src.index("if self.reshape_tensors_for_rl:")
    second = src.index("if self.vla_preprocess_func is not None:", first + 1)
    assert first < reshape < second
    assert "image_frames.reshape(n_cam * n_frame" in src
    assert "camera_indices.repeat_interleave(n_frame)" in src


def test_evaluate_hf_only_metric_keys_by_numel(torch):
    src = _EVAL.read_text()
    assert 'if not k.startswith("metric/"):' in src
    assert "metric_sums[k] += gathered_metric.float().sum().item()" in src
    assert "metric_counts[k] += gathered_metric.numel()" in src
    assert 'final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]' in src

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    batches = [
        {"metric/ade": torch.tensor([1.0, 3.0]), "loss": torch.tensor([9.0])},
        {"metric/ade": torch.tensor([5.0]), "other": torch.tensor([0.0])},
    ]
    for output_batch in batches:
        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            sums[k] = sums.get(k, 0.0) + float(v.float().sum().item())
            counts[k] = counts.get(k, 0) + int(v.numel())
    assert set(sums) == {"metric/ade"}
    assert sums["metric/ade"] / counts["metric/ade"] == pytest.approx((1 + 3 + 5) / 3)


def test_comfort_means_over_k_then_n(torch):
    src = _COMFORT.read_text()
    assert "comfort_metric_dict[name] = _within_bound(ego_dynamics[dyn_key], lo, hi).mean(2)" in src
    assert "comfort_metric_dict[name] = comfort_metric_dict[name].mean(dim=-1)" in src

    # [B=1, N=2, K=2] within-bound flags. Mean over K then N is 0.5, not 0 or 1.
    flags = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    over_k = flags.mean(2)
    over_n = over_k.mean(dim=-1)
    assert over_k.tolist() == [[0.5, 0.5]]
    assert over_n.tolist() == [0.5]
    any_k = flags.max(dim=2).values.mean(dim=-1)
    assert any_k.tolist() == [1.0]
    assert any_k.tolist() != over_n.tolist()


def test_viz_front_wide_t0_is_flat_index_7():
    src = _VIZ.read_text()
    assert "t0_front_wide = frames[7]  # 7 = t0 frame of front wide" in src
    assert "frames = image_frames.flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()" in src
    helper = _HELPER.read_text()
    assert "if frames.ndim != 4:" in helper
    assert 'raise ValueError(f"{frames.ndim=}, expected 4 (N, C, H, W)")' in helper
