"""Unicycle encode pins future heading and first velocity.

Weight-free. Complementary to open #26 (unwrap / supplied t0) and #27
(solve pin on ``x[0]``) — this locks ``theta_smooth`` re-zeroing the
first future heading, ``dxy_theta_to_v`` keeping ``v0``, and
``estimate_t0_states`` reading the last history speed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _stub_nv_logging() -> None:
    if "alpamayo_r1.common.logging" in sys.modules:
        return
    logging_mod = ModuleType("alpamayo_r1.common.logging")

    class RankedLogger:
        def __init__(self, *args, **kwargs):
            pass

        def setLevel(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    logging_mod.RankedLogger = RankedLogger
    sys.modules["alpamayo_r1.common.logging"] = logging_mod


def _load_utils():
    _ensure_src_on_path()
    _stub_nv_logging()
    try:
        from alpamayo_r1.action_space.utils import dxy_theta_to_v, theta_smooth
    except Exception as exc:
        pytest.skip(f"NVIDIA action-space utils unavailable: {exc}")
    return dxy_theta_to_v, theta_smooth


def _load_unicycle():
    _ensure_src_on_path()
    _stub_nv_logging()
    try:
        from alpamayo_r1.action_space.unicycle_accel_curvature import (
            UnicycleAccelCurvatureActionSpace,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA unicycle action space unavailable: {exc}")
    return UnicycleAccelCurvatureActionSpace


def _yaw_R(torch, yaw: float, n: int):
    c = float(torch.cos(torch.tensor(yaw)))
    s = float(torch.sin(torch.tensor(yaw)))
    r = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    return r.expand(n, 3, 3).clone()


def test_theta_smooth_pins_first_heading_to_zero(torch):
    _dxy_theta_to_v, theta_smooth = _load_utils()
    del _dxy_theta_to_v
    rot = _yaw_R(torch, float(torch.pi / 2), n=8)
    yaw = theta_smooth(rot, dt=0.1)
    assert yaw.shape == (9,)
    assert float(yaw[0].item()) == pytest.approx(0.0, abs=1e-6)
    # Remaining samples are pulled toward the constant π/2 target, not left at 0.
    assert float(yaw[-1].item()) == pytest.approx(float(torch.pi / 2), abs=0.35)
    assert float(yaw[-1].item()) > 0.8


def test_dxy_theta_to_v_keeps_supplied_v0(torch):
    dxy_theta_to_v, _theta_smooth = _load_utils()
    del _theta_smooth
    # 0.2 m / 0.1 s → 2 m/s along +X, heading 0.
    dxy = torch.tensor([[0.2, 0.0]] * 6, dtype=torch.float32)
    theta = torch.zeros(7)
    v0 = torch.tensor(0.5)
    v = dxy_theta_to_v(dxy, theta, v0, dt=0.1)
    assert v.shape == (7,)
    assert float(v[0].item()) == pytest.approx(0.5, abs=1e-5)
    # Smoothing pulls toward the 2 m/s dxy implies; must not stay pinned at v0.
    assert float(v[-1].item()) > 1.2
    assert float(v[-1].item()) < 2.1


def test_estimate_t0_states_is_last_history_speed(torch):
    UnicycleAccelCurvatureActionSpace = _load_unicycle()
    spc = UnicycleAccelCurvatureActionSpace(dt=0.1, n_waypoints=8)
    # 5 hist steps at 1 m/s +X, ending at the origin.
    t = torch.linspace(-0.4, 0.0, 5)
    hist_xyz = torch.stack([t, torch.zeros(5), torch.zeros(5)], dim=-1)
    hist_rot = _yaw_R(torch, 0.0, n=5)
    states = spc.estimate_t0_states(hist_xyz, hist_rot)
    assert float(states["v"].item()) == pytest.approx(1.0, abs=0.15)
