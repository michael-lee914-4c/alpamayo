"""``summarize_metric`` std is population ``sqrt(E[x^2]-μ^2)``, not Bessel.

Weight-free. Complementary to open #46 (never emits ``_sq``) and #41
(N=1 / ``disable_summary``). The implementation uses
``(val.pow(2).mean(1) - mean**2).sqrt()`` — divide by N, not N-1.
``tensor.std(dim=1)`` (PyTorch default ``unbiased=True``) would report a
larger spread and change every ``min_ade_*_std`` / comfort-std dashboard
number without touching the mean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_UTILS = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_utils.py"


def _population_std(val: np.ndarray) -> np.ndarray:
    mean = val.mean(axis=1)
    mean_sq = np.square(val).mean(axis=1)
    return np.sqrt(mean_sq - mean**2)


def test_source_uses_second_moment_not_unbiased_std():
    src = _UTILS.read_text()
    body = src.split("def summarize_metric")[1].split("def apply_prefix")[0]
    assert "mean_sq = val.pow(2).mean(1)" in body
    assert "std = (mean_sq - mean**2).sqrt()" in body
    assert "val.std(" not in body
    assert "unbiased" not in body


def test_population_std_is_smaller_than_bessel_for_n_gt_1():
    val = np.array([[0.0, 2.0, 4.0]], dtype=np.float64)
    pop = _population_std(val)
    bessel = val.std(axis=1, ddof=1)
    assert pop.shape == (1,)
    np.testing.assert_allclose(pop, [np.sqrt(8.0 / 3.0)])
    assert float(pop[0]) < float(bessel[0])


def test_n1_skips_std_even_with_population_formula():
    # source gate is val.shape[1] > 1; N=1 would also be 0 under either std
    val = np.array([[3.0]], dtype=np.float64)
    assert val.shape[1] == 1
    assert _population_std(val)[0] == 0.0
