"""``compute_grouped_corner_distance`` skips ``summarize_metric`` when N==1.

Weight-free. Complementary to open #41 (``disable_summary``) and #47
(population std). Even with ``disable_summary=False``, a single traj
set returns only ``corner_distance`` (mean over the N=1 axis) and never
emits ``corner_distance_std``. Smoke evals with ``num_traj_sets==1``
would then miss std keys that multi-set runs report.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DIST = _REPO / "src" / "alpamayo_r1" / "metrics" / "distance_metrics.py"


def test_source_bypasses_summarize_when_n_is_one():
    src = _DIST.read_text()
    body = src.split("def compute_grouped_corner_distance")[1]
    assert "if disable_summary or (N == 1):" in body
    assert 'return {"corner_distance": distance.mean(dim=1)}' in body
    assert 'return summarize_metric({"corner_distance": distance})' in body
