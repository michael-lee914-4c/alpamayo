"""``summarize_metric`` emits ``name`` + ``name_std`` and never ``name_sq``.

Weight-free. Complementary to open #41 (N=1 / ``disable_summary`` skip
``_std``) and #38 (rank≠2 raise). The docstring still mentions
``name_sq``, but the body only writes ``mean`` and ``sqrt(mean_sq - mean²)``.
Dashboards that expect squared keys would silently go empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_UTILS = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_utils.py"


def _summarize(metric: dict[str, np.ndarray], disable_summary: bool = False) -> dict[str, np.ndarray]:
    result = {}
    for name, val in metric.items():
        if val.ndim != 2:
            raise ValueError(name)
        mean = val.mean(axis=1)
        result[name] = mean
        if val.shape[1] > 1 and not disable_summary:
            mean_sq = np.square(val).mean(axis=1)
            result[name + "_std"] = np.sqrt(mean_sq - mean**2)
    return result


def test_source_writes_std_not_sq():
    src = _UTILS.read_text()
    assert 'result[metric_name + "_std"] = std' in src
    assert "mean_sq = val.pow(2).mean(1)" in src
    assert 'result[metric_name + "_sq"]' not in src


def test_multi_group_keys_are_mean_and_std_only():
    val = np.array([[1.0, 3.0, 5.0]], dtype=np.float64)
    out = _summarize({"min_ade": val})
    assert set(out) == {"min_ade", "min_ade_std"}
    np.testing.assert_allclose(out["min_ade"], [3.0])
    np.testing.assert_allclose(out["min_ade_std"], [np.sqrt(8.0 / 3.0)])


def test_single_group_omits_std_and_sq():
    out = _summarize({"min_ade": np.array([[2.0]])})
    assert set(out) == {"min_ade"}
