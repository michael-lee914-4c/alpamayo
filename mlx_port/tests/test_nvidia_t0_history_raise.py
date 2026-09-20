"""PAI loader rejects ``t0`` that cannot fit ``num_history_steps * dt``.

Weight-free. Complementary to open #26 (event-t0 1.6s/6.4s clip filter)
and #28 (history includes t0 / future starts at t0+dt). The loader uses
``n * dt`` for the raise even though the earliest stamp is
``t0 - (n-1)*dt``. Tightening that inequality to ``(n-1)*dt`` would
admit clips the event filter already dropped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_LOADER = _REPO / "src" / "alpamayo_r1" / "load_physical_aiavdataset.py"


def _history_future_stamps(
    t0_us: int,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Replica of NVIDIA history/future stamp construction + raise."""
    history_time_range_us = num_history_steps * time_step * 1_000_000
    if t0_us <= history_time_range_us:
        raise ValueError(
            f"{t0_us=} must be greater than the history time range "
            f"({history_time_range_us=} us)"
        )
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    return t0_us + history_offsets_us, t0_us + future_offsets_us


def test_source_raises_on_n_times_dt_not_n_minus_one():
    src = _LOADER.read_text()
    assert "history_time_range_us = num_history_steps * time_step * 1_000_000" in src
    assert "if t0_us <= history_time_range_us:" in src
    assert "-(num_history_steps - 1) * time_step * 1_000_000" in src
    assert "time_step * 1_000_000," in src


def test_default_t0_is_just_above_the_1_6s_raise():
    hist, fut = _history_future_stamps(5_100_000)
    assert hist[-1] == 5_100_000
    assert hist[0] == 5_100_000 - 15 * 100_000
    assert fut[0] == 5_100_000 + 100_000
    assert len(hist) == 16
    assert len(fut) == 64


def test_t0_equal_to_n_dt_raises_even_though_earliest_stamp_fits():
    # n=16, dt=0.1s → raise uses 1.6s, earliest hist stamp would be t0-1.5s.
    with pytest.raises(ValueError, match="must be greater than the history time range"):
        _history_future_stamps(1_600_000)
    with pytest.raises(ValueError, match="must be greater than the history time range"):
        _history_future_stamps(0)
    hist, _fut = _history_future_stamps(1_600_001)
    assert hist[0] == 1_600_001 - 1_500_000
    assert hist[0] > 0
