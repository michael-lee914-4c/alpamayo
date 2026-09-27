"""``compute_minade`` default horizons ``[5,10,30,50]`` label as 0.5/1.0/3.0/5.0 s.

Weight-free. Complementary to open #39 (DistanceMetrics
``[0.5,1,3,5]`` s → ``int(t/dt)``) and #46 (full-horizon K pick). The
function stores *timestep counts* and formats keys as
``min_ade/by_t={t * time_step:.1f}`` with default ``time_step=0.1``.
A port that treats the list as seconds would emit ``by_t=5.0`` for the
0.5 s bucket and fork every eval table.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DIST = _REPO / "src" / "alpamayo_r1" / "metrics" / "distance_metrics.py"


def _horizon_key(t: int, time_step: float = 0.1) -> str:
    return f"min_ade/by_t={t * time_step:.1f}"


def test_source_default_horizons_are_timestep_counts():
    src = _DIST.read_text()
    body = src.split("def compute_minade")[1].split("def compute_grouped_corner_distance")[0]
    assert "timestep_horizons: Iterable[int] = [5, 10, 30, 50]" in body
    assert "time_step: float = 0.1" in body
    assert 'out[f"min_ade/by_t={t * time_step:.1f}"] = min_ade_t' in body


def test_default_labels_are_half_one_three_five_seconds():
    keys = [_horizon_key(t) for t in (5, 10, 30, 50)]
    assert keys == [
        "min_ade/by_t=0.5",
        "min_ade/by_t=1.0",
        "min_ade/by_t=3.0",
        "min_ade/by_t=5.0",
    ]
    # treating the list as seconds would mis-label the 0.5 s bucket
    assert _horizon_key(5, time_step=1.0) == "min_ade/by_t=5.0"
    assert _horizon_key(5, time_step=1.0) != keys[0]
