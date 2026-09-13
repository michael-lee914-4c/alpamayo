"""NVIDIA ``solve_xs_eq_y``: recover y/s and reject a mismatched weight.

Weight-free. Complementary to open #27 (DTD scale / pinned-x0) — those
never locked the unconstrained ``xs = y`` least-squares used by
``_v_to_a`` and ``_theta_v_a_to_kappa``. A silent ``dtheta/s`` swap
here is what previously blew Stage-2 CFM (~11k).
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


def _load_solve():
    _ensure_src_on_path()
    _stub_nv_logging()
    pkg = sys.modules.get("alpamayo_r1.action_space")
    if pkg is None or not getattr(pkg, "__path__", None):
        pkg = ModuleType("alpamayo_r1.action_space")
        pkg.__path__ = [str(_SRC / "alpamayo_r1" / "action_space")]
        sys.modules["alpamayo_r1.action_space"] = pkg
    path = _SRC / "alpamayo_r1" / "action_space" / "utils.py"
    spec = importlib.util.spec_from_file_location("nv_action_space_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA solve_xs_eq_y unavailable: {exc}")
    return module.solve_xs_eq_y


def test_solve_xs_eq_y_recovers_y_over_s_when_unsmoothed(torch):
    solve_xs_eq_y = _load_solve()
    s = torch.full((5,), 2.0)
    y = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    x = solve_xs_eq_y(s, y, lam=0.0, ridge=1e-8)
    assert x.shape == (5,)
    assert torch.allclose(x, y / s, atol=1e-4)


def test_solve_xs_eq_y_identity_slope_recovers_y(torch):
    solve_xs_eq_y = _load_solve()
    y = torch.linspace(-1.0, 3.0, 6)
    x = solve_xs_eq_y(torch.ones_like(y), y, lam=0.0, ridge=1e-8)
    assert torch.allclose(x, y, atol=1e-4)


def test_solve_xs_eq_y_rejects_w_data_shape_mismatch(torch):
    solve_xs_eq_y = _load_solve()
    y = torch.ones(4)
    try:
        solve_xs_eq_y(torch.ones_like(y), y, w_data=torch.ones(3))
    except ValueError as exc:
        assert "w_data" in str(exc)
    else:
        raise AssertionError("expected ValueError when w_data shape != y")
