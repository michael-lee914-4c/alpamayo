"""DistanceMetrics converts ``[0.5, 1, 3, 5]`` seconds via ``int(t/dt)``.

Weight-free. Complementary to open #27 (3 s ADE omitted when ``T < 3s``)
and #34 (``compute_minade`` skips horizons ``> T``). The wrapper — not
the primitive — owns the seconds list and the ``int(t / time_step)``
conversion that names ``min_ade/by_t=*`` keys.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_METRIC = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_api.py"
_DISTANCE = _REPO / "src" / "alpamayo_r1" / "metrics" / "distance_metrics.py"
_RUNNER = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_runner.py"


def test_source_horizon_seconds_and_metric_prefix():
    api = _METRIC.read_text()
    assert "timestep_horizons_in_seconds = [0.5, 1, 3, 5]" in api
    assert "timestep_horizons = [int(t / self.time_step) for t in timestep_horizons_in_seconds]" in api
    assert "timestep_horizon = int(3.0 / self.time_step)" in api
    # Dummy vehicle size used by the same evaluate() path.
    assert "EGO_VEHICLE_LWH = (4.0, 3.0, 2.0)" in api

    dist = _DISTANCE.read_text()
    assert "timestep_horizons: Iterable[int] = [5, 10, 30, 50]" in dist
    assert 'out[f"min_ade/by_t={t * time_step:.1f}"] = min_ade_t' in dist

    runner = _RUNNER.read_text()
    assert 'output_batch.update({"metric/" + k: v for k, v in per_sample_metrics.items()})' in runner


def test_seconds_to_steps_at_default_and_shifted_dt():
    seconds = [0.5, 1, 3, 5]
    assert [int(t / 0.1) for t in seconds] == [5, 10, 30, 50]
    # 20 Hz would silently retune every by_t key if someone changes time_step
    # without updating the seconds list.
    assert [int(t / 0.05) for t in seconds] == [10, 20, 60, 100]
    # Floor, not round: 0.5 / 0.3 = 1.666 → 1, not 2.
    assert [int(t / 0.3) for t in seconds] == [1, 3, 10, 16]


def test_minade_key_uses_one_decimal_times_step():
    time_step = 0.1
    keys = [f"min_ade/by_t={t * time_step:.1f}" for t in [5, 10, 30, 50]]
    assert keys == [
        "min_ade/by_t=0.5",
        "min_ade/by_t=1.0",
        "min_ade/by_t=3.0",
        "min_ade/by_t=5.0",
    ]
    # A :.0f rewrite would collapse 0.5 s into ``by_t=0``.
    assert f"{5 * time_step:.0f}" == "0"


def test_metric_runner_prefixes_then_evaluate_rewrites_val():
    # MetricRunner writes metric/<name>; evaluate_hf then writes val/metric/<name>.
    per_sample = {"min_ade": 1.0, "ade/by_t=3.0": 2.0}
    prefixed = {"metric/" + k: v for k, v in per_sample.items()}
    assert set(prefixed) == {"metric/min_ade", "metric/ade/by_t=3.0"}
    final = {"val/" + k: v for k, v in prefixed.items()}
    assert set(final) == {"val/metric/min_ade", "val/metric/ade/by_t=3.0"}
    eval_src = (_REPO / "finetune" / "sft" / "evaluate_hf.py").read_text()
    assert 'if not k.startswith("metric/"):' in eval_src
    assert 'final_metrics_dict["val/" + key] = metric_sums[key] / metric_counts[key]' in eval_src
