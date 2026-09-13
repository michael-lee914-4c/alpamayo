"""NVIDIA unicycle inverse kinematics: ``_v_to_a`` and ``_theta_v_a_to_kappa``.

Weight-free. Complementary to open #26 (bounds / action_to_traj origin)
and #28 (theta/v0 pin) — those never locked the CUDA SFT label generators
that go through ``solve_xs_eq_y`` instead of a raw ``dtheta/s``.
"""

from __future__ import annotations

import importlib.util
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


def _ensure_action_space_pkg() -> None:
    _ensure_src_on_path()
    if "alpamayo_r1.action_space" in sys.modules:
        return
    pkg = ModuleType("alpamayo_r1.action_space")
    pkg.__path__ = [str(_SRC / "alpamayo_r1" / "action_space")]
    sys.modules["alpamayo_r1.action_space"] = pkg


def _load_unicycle():
    _ensure_src_on_path()
    _stub_nv_logging()
    _ensure_action_space_pkg()
    path = _SRC / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"
    spec = importlib.util.spec_from_file_location("nv_unicycle_inverse", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA unicycle action space unavailable: {exc}")
    return module.UnicycleAccelCurvatureActionSpace


def test_v_to_a_recovers_constant_accel(torch):
    Space = _load_unicycle()
    space = Space(dt=0.1, n_waypoints=4, a_lambda=1e-10, a_ridge=1e-10)
    # v = 0, 1, 2, 3, 4  →  Δv/dt = 10 m/s²
    v = torch.arange(5, dtype=torch.float32)
    accel = space._v_to_a(v)
    assert accel.shape == (4,)
    assert torch.allclose(accel, torch.full((4,), 10.0), atol=1e-3)


def test_theta_v_a_to_kappa_is_dtheta_over_arc(torch):
    Space = _load_unicycle()
    space = Space(dt=0.1, n_waypoints=4, kappa_lambda=1e-10, kappa_ridge=1e-10)
    # Constant v=10, a=0, Δθ=0.1 → s = dt*v = 1.0 → κ = 0.1
    theta = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4])
    v = torch.full((5,), 10.0)
    a = torch.zeros(4)
    kappa = space._theta_v_a_to_kappa(theta, v, a)
    assert kappa.shape == (4,)
    assert torch.allclose(kappa, torch.full((4,), 0.1), atol=5e-3)
    src = (_SRC / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py").read_text()
    assert "return solve_xs_eq_y(" in src
    assert "s = dt * v[..., :-1] + (dt**2) / 2.0 * a" in src
