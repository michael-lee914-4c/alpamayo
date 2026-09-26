"""``compute_minade`` picks K from the full-horizon mean, then slices.

Weight-free. Complementary to open #34 (horizon>T skip) and #41 (ADE XY).
``idx = l2.mean(dim=-1).argmin(dim=2)`` chooses one sample using every
timestep, and ``min_ade/by_t=…`` then averages *that* sample's prefix.
Per-horizon argmin would report a different winner at 0.3 s vs 1.0 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DIST = _REPO / "src" / "alpamayo_r1" / "metrics" / "distance_metrics.py"


def _minade_from_l2(l2: np.ndarray, horizons: tuple[int, ...], time_step: float = 0.1):
    """l2: [B, N, K, T] → same reduction as NVIDIA ``compute_minade``."""
    idx = l2.mean(axis=-1).argmin(axis=2)  # [B, N]
    b, n, _, t = l2.shape
    gather = np.take_along_axis(l2, idx[:, :, None, None], axis=2).squeeze(2)  # [B, N, T]
    out = {"min_ade": gather.mean(axis=2)}
    for h in horizons:
        if h <= t:
            out[f"min_ade/by_t={h * time_step:.1f}"] = gather[:, :, :h].mean(axis=2)
    return out, idx


def test_source_argmin_is_full_horizon_mean():
    src = _DIST.read_text()
    assert "idx = l2.mean(dim=-1).argmin(dim=2)" in src
    assert "min_diff[:, :, :t].mean(dim=2)" in src
    assert "l2[..., :t].mean" not in src.split("idx = l2.mean(dim=-1).argmin(dim=2)")[1]


def test_short_horizon_follows_full_horizon_winner_not_local_argmin():
    # K0: error 1.0 at every step. K1: 0.1 for the first 3 steps, then 10.0.
    t = 10
    k0 = np.full(t, 1.0)
    k1 = np.concatenate([np.full(3, 0.1), np.full(t - 3, 10.0)])
    l2 = np.stack([k0, k1], axis=0).reshape(1, 1, 2, t)

    out, idx = _minade_from_l2(l2, horizons=(3, 10))
    assert idx.item() == 0  # full-horizon mean picks K0 (1.0 < 7.03)
    np.testing.assert_allclose(out["min_ade/by_t=0.3"], [[1.0]])
    np.testing.assert_allclose(out["min_ade"], [[1.0]])

    local_winner = l2[..., :3].mean(axis=-1).argmin(axis=2)
    assert local_winner.item() == 1
    assert local_winner.item() != idx.item()
