"""``evaluate_hf`` averages only ``metric/`` keys and rewrites them to ``val/``.

Weight-free. Complementary to open #28 (MetricRunner ``metric/`` prefix)
and #33 (``sum`` / ``numel``). ``sample_ade`` and other working tensors
live on ``output_batch`` without that prefix; folding them into the
dashboard would average ADE with image counts or logprobs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_EVAL = _REPO / "finetune" / "sft" / "evaluate_hf.py"
_RUN = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_runner.py"


def _aggregate_eval(output_batches: list[dict]) -> dict[str, float]:
    metric_sums: dict[str, float] = defaultdict(float)
    metric_counts: dict[str, int] = defaultdict(int)
    for output_batch in output_batches:
        for k, v in output_batch.items():
            if not k.startswith("metric/"):
                continue
            metric_sums[k] += float(sum(v))
            metric_counts[k] += len(v)
    final = {}
    for key in metric_sums:
        if metric_counts[key] > 0:
            final["val/" + key] = metric_sums[key] / metric_counts[key]
    return final


def test_source_filters_metric_prefix_and_rewrites_to_val():
    eval_src = _EVAL.read_text()
    run_src = _RUN.read_text()
    assert 'if not k.startswith("metric/"):' in eval_src
    assert 'final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]' in eval_src
    assert "metric_counts[k] += gathered_metric.numel()" in eval_src
    assert 'output_batch.update({"metric/" + k: v for k, v in per_sample_metrics.items()})' in run_src


def test_sample_ade_is_not_averaged_into_val():
    batches = [
        {
            "metric/min_ade": [1.0, 3.0],
            "sample_ade": [99.0, 99.0],
            "pred_xyz": [0.0],
        },
        {
            "metric/min_ade": [5.0],
            "sample_ade": [0.0],
        },
    ]
    got = _aggregate_eval(batches)
    assert got == {"val/metric/min_ade": (1.0 + 3.0 + 5.0) / 3.0}
    assert "val/sample_ade" not in got
    assert "val/metric/sample_ade" not in got


def test_empty_metric_keys_are_skipped():
    assert _aggregate_eval([{"pred_xyz": [1.0]}]) == {}
    assert _aggregate_eval([{"metric/ade": []}]) == {}
