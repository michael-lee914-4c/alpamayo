"""NVIDIA ``dxy_theta_to_v_without_v0``: unconstrained trapezoidal speed.

Weight-free. Complementary to open #28 (``dxy_theta_to_v`` keeps a supplied
``v0``; ``estimate_t0_states`` is last-hist speed) — those never locked the
unconstrained least-squares that *produces* t0 speed. A silent
``|Δp|/dt`` or v0-constrained swap here would retune every Stage-2 encode.
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


def _load_v_without_v0():
    _ensure_src_on_path()
    _stub_nv_logging()
    pkg = sys.modules.get("alpamayo_r1.action_space")
    if pkg is None or not getattr(pkg, "__path__", None):
        pkg = ModuleType("alpamayo_r1.action_space")
        pkg.__path__ = [str(_SRC / "alpamayo_r1" / "action_space")]
        sys.modules["alpamayo_r1.action_space"] = pkg
    path = _SRC / "alpamayo_r1" / "action_space" / "utils.py"
    spec = importlib.util.spec_from_file_location("nv_action_space_utils_v0free", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA dxy_theta_to_v_without_v0 unavailable: {exc}")
    return module.dxy_theta_to_v_without_v0


def test_v_without_v0_recovers_constant_plus_x_speed(torch):
    dxy_theta_to_v_without_v0 = _load_v_without_v0()
    # 0.2 m / 0.1 s along +X, heading 0 → 2 m/s at every knot (no v0 pin).
    dxy = torch.tensor([[0.2, 0.0]] * 8, dtype=torch.float32)
    theta = torch.zeros(9)
    v = dxy_theta_to_v_without_v0(dxy, theta, dt=0.1)
    assert v.shape == (9,)
    assert torch.allclose(v, torch.full((9,), 2.0), atol=0.05)


def test_v_without_v0_is_heading_aligned_not_speed_norm(torch):
    dxy_theta_to_v_without_v0 = _load_v_without_v0()
    # Motion is +Y (0.2 m / step) but heading stays 0 (body +X). Trapezoidal
    # LS projects onto u= (cos θ, sin θ) = (1, 0), so recovered v is ~0,
    # not |Δp|/dt = 2.
    dxy = torch.tensor([[0.0, 0.2]] * 6, dtype=torch.float32)
    theta = torch.zeros(7)
    v = dxy_theta_to_v_without_v0(dxy, theta, dt=0.1)
    assert float(v.abs().max().item()) < 0.35
    naive = float(torch.linalg.norm(dxy[0]).item() / 0.1)
    assert naive == pytest.approx(2.0, abs=1e-5)
    assert float(v.mean().item()) < 0.5 * naive
