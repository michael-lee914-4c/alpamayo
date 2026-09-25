"""PAI future stamps use ``(T + 0.5) * dt`` as the exclusive ``arange`` stop.

Weight-free. Complementary to open #28 (history includes t0 / future starts at
``t0+dt``). ``np.arange`` is end-exclusive, so a stop of ``T * dt`` would drop
the last waypoint and silently shorten every ADE / reward horizon by one step.
The extra half-step keeps ``len == T`` and the last offset at ``T * dt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_LOAD = _REPO / "src" / "alpamayo_r1" / "load_physical_aiavdataset.py"


def _future_offsets_us(num_future_steps: int, time_step: float) -> np.ndarray:
    return np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)


def test_source_uses_half_step_exclusive_stop():
    src = _LOAD.read_text()
    assert "(num_future_steps + 0.5) * time_step * 1_000_000" in src
    assert "future_offsets_us = np.arange(" in src
    assert "time_step * 1_000_000," in src


def test_default_horizon_keeps_64_steps_ending_at_6_4s():
    offsets = _future_offsets_us(64, 0.1)
    assert offsets.shape == (64,)
    assert offsets[0] == 100_000
    assert offsets[-1] == 6_400_000


def test_exclusive_stop_without_half_step_drops_last_waypoint():
    time_step = 0.1
    num_future_steps = 64
    dropped = np.arange(
        time_step * 1_000_000,
        num_future_steps * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    kept = _future_offsets_us(num_future_steps, time_step)
    assert dropped.shape == (63,)
    assert kept.shape == (64,)
    np.testing.assert_array_equal(dropped, kept[:-1])
