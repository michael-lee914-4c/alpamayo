"""``traj_to_action(..., output_all_states=True)`` slices with ``[:, :-1]``.

Weight-free. Complementary to open #13 (``_v_to_a`` / ``_theta_v_a_to_kappa``)
and #42 (denorm-then-bounds). Extra returns stack ``v[:, :-1]`` and
``theta[:, :-1]`` instead of ``[..., :-1]``. A traj-group leading dim
then drops the wrong axis and the logged (v, accel, theta) states no
longer line up with the (N, 2) action.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def _stack_all_states(v, accel, theta):
    torch = pytest.importorskip("torch")
    return torch.stack([v[:, :-1], accel, theta[:, :-1]], dim=-1)


def test_source_uses_leading_batch_slice_not_ellipsis():
    src = _UNI.read_text()
    assert "if not output_all_states:" in src
    assert "[v[:, :-1], accel, theta[:, :-1]]" in src
    assert "v[..., :-1]" not in src
    assert "theta[..., :-1]" not in src


def test_rank2_batch_matches_ellipsis_slice():
    torch = pytest.importorskip("torch")
    v = torch.arange(8, dtype=torch.float32).view(2, 4)
    accel = torch.ones(2, 3)
    theta = v * 0.1
    got = _stack_all_states(v, accel, theta)
    expect = torch.stack([v[..., :-1], accel, theta[..., :-1]], dim=-1)
    torch.testing.assert_close(got, expect)
    assert got.shape == (2, 3, 3)


def test_extra_leading_dim_diverges_from_ellipsis():
    torch = pytest.importorskip("torch")
    v = torch.arange(16, dtype=torch.float32).view(2, 2, 4)
    accel = torch.ones(2, 2, 3)
    theta = v * 0.1
    # Production ``[:, :-1]`` drops dim 1 (the group), not the waypoint axis.
    with pytest.raises(RuntimeError):
        _stack_all_states(v, accel, theta)
    ellipsis = torch.stack([v[..., :-1], accel, theta[..., :-1]], dim=-1)
    assert ellipsis.shape == (2, 2, 3, 3)
