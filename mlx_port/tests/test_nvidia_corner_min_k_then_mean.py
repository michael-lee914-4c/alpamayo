"""Corner distance is min-over-K, then mean over time and 8 corners.

Weight-free. Complementary to open #25 (grouped corner best-of-K) and
#36 (``xyzrot_to_corners`` bottom-then-top). After
``norm(dim=-1) → [B, N, K, T, 8]``, the reduction is
``min(dim=2)[0].mean(dim=(2, 3))``. Averaging K or minning time first
would change every ``corner_distance`` dashboard number.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DIST = _REPO / "src" / "alpamayo_r1" / "metrics" / "distance_metrics.py"


def test_source_mins_k_then_means_time_and_corners():
    src = _DIST.read_text()
    body = src.split("def compute_grouped_corner_distance")[1]
    assert "distance.min(dim=2)[0].mean(dim=(2, 3))" in body
    assert "distance.mean(dim=2)" not in body
    assert "distance.min(dim=3)" not in body


def test_min_k_then_mean_differs_from_mean_k():
    # [K=2, T=2, 8] after the B,N axes are dropped for the replica.
    dist = np.array(
        [
            np.full((2, 8), 0.0),
            np.full((2, 8), 10.0),
        ],
        dtype=np.float64,
    )
    min_then_mean = dist.min(axis=0).mean()
    mean_k = dist.mean(axis=0).mean()
    assert min_then_mean == 0.0
    assert mean_k == 5.0
