"""PAI relative timestamps convert microseconds to seconds with ``* 1e-6``.

Weight-free. Complementary to open #29 (relative stamps from the global
camera tmin) and #41 (image stamps ``[t0-(n-1)dt … t0]``).
``(all_timestamps - camera_tmin).float() * 1e-6`` is seconds. Leaving
the values in microseconds (or dividing by 1e3) desynchronizes image
time embeddings from the 0.1 s traj grid.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_LOADER = _REPO / "src" / "alpamayo_r1" / "load_physical_aiavdataset.py"


def test_source_scales_delta_us_by_1e_minus_6():
    src = _LOADER.read_text()
    assert "relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6" in src
    assert "* 1e-3" not in src
    assert "/ 1_000_000" not in src


def test_100ms_gap_is_0_1_seconds():
    tmin = 5_100_000
    stamps = np.array([tmin, tmin + 100_000], dtype=np.int64)
    rel = (stamps - tmin).astype(np.float64) * 1e-6
    assert rel[0] == 0.0
    assert np.isclose(rel[1], 0.1)
    millis = (stamps - tmin).astype(np.float64) * 1e-3
    assert millis.tolist() == [0.0, 100.0]
