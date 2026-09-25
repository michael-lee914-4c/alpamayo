"""``traj_to_action`` prepends the last history waypoint, assumed to be origin.

Weight-free. Complementary to open #28 (``theta_smooth`` pins heading 0) and
#30 (unconstrained t0 speed). The encoder concatenates
``hist[..., -1:, :2]`` with future XY before differencing. Comments require
that last history point to be the local-frame origin. A non-zero last hist
shifts every ``dxy`` and therefore every accel / kappa token.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def test_source_cats_last_hist_and_assumes_origin():
    src = _UNI.read_text()
    assert "full_xy = torch.cat([traj_history_xyz[..., -1:, :], traj_future_xyz], dim=-2)" in src
    assert "[..., :2]" in src
    assert "dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]" in src
    assert "traj_history_xyz[..., -1, :] is the current position" in src


def test_nonzero_last_hist_shifts_first_dxy():
    torch = pytest.importorskip("torch")
    hist_origin = torch.zeros(1, 3, 3)
    hist_shifted = hist_origin.clone()
    hist_shifted[0, -1, :2] = torch.tensor([1.5, -0.5])
    fut = torch.zeros(1, 4, 3)
    fut[0, :, 0] = torch.arange(1, 5, dtype=torch.float32)

    def first_dxy(hist):
        full_xy = torch.cat([hist[..., -1:, :], fut], dim=-2)[..., :2]
        return full_xy[..., 1:, :] - full_xy[..., :-1, :]

    d_origin = first_dxy(hist_origin)
    d_shift = first_dxy(hist_shifted)
    # First step from origin is (1, 0); from a shifted last-hist it is not.
    torch.testing.assert_close(d_origin[0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(d_shift[0, 0], torch.tensor([-0.5, 0.5]))
    # Later steps (future-to-future) stay the same.
    torch.testing.assert_close(d_origin[0, 1:], d_shift[0, 1:])
