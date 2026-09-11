"""NVIDIA construct_DTD dt scaling and pinned-initial solve.

Weight-free. Complementary to open #26 D-band shapes — this locks
``lam/dt**{2,4,6}`` on the normal-equation matrix and the
``solve_single_constraint`` pin that every unicycle heading/velocity
smoother goes through.
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
        from alpamayo_r1.action_space.utils import construct_DTD, solve_single_constraint
    except Exception as exc:
        pytest.skip(f"NVIDIA action-space utils unavailable: {exc}")
    return construct_DTD, solve_single_constraint


def test_construct_dtd_second_order_scales_as_dt_to_the_minus_4(torch):
    construct_DTD, _solve = _load_utils()
    kwargs = dict(N=4, lead=(), w_smooth2=1.0, lam=1.0)
    dtd_fast = construct_DTD(dt=0.1, **kwargs)
    dtd_slow = construct_DTD(dt=0.2, **kwargs)
    # lam/dt**4 → (0.2/0.1)**4 = 16.
    assert dtd_fast.shape == (4, 4)
    assert not torch.allclose(dtd_fast, torch.zeros_like(dtd_fast))
    assert torch.allclose(dtd_fast, dtd_slow * 16.0, atol=1e-5)


def test_construct_dtd_third_order_empty_when_n_too_short(torch):
    construct_DTD, _solve = _load_utils()
    dtd_ok = construct_DTD(N=4, lead=(), w_smooth3=1.0, lam=1.0, dt=0.1)
    dtd_short = construct_DTD(N=3, lead=(), w_smooth3=1.0, lam=1.0, dt=0.1)
    assert dtd_ok.shape == (4, 4)
    assert not torch.equal(dtd_ok, torch.zeros_like(dtd_ok))
    assert torch.equal(dtd_short, torch.zeros(3, 3))


def test_construct_dtd_first_order_scales_as_dt_to_the_minus_2(torch):
    construct_DTD, _solve = _load_utils()
    dtd_fast = construct_DTD(N=4, lead=(), w_smooth1=1.0, lam=1.0, dt=0.1)
    dtd_slow = construct_DTD(N=4, lead=(), w_smooth1=1.0, lam=1.0, dt=0.2)
    assert torch.allclose(dtd_fast, dtd_slow * 4.0, atol=1e-5)


def test_solve_single_constraint_pins_x0(torch):
    _dtd, solve_single_constraint = _load_utils()
    target = torch.linspace(1.0, 4.0, 4)
    out = solve_single_constraint(
        x_init=torch.tensor(0.0),
        x_target=target,
        w_smooth3=1.0,
        lam=1e-3,
        dt=0.1,
    )
    assert out.shape == (5,)
    assert float(out[0].item()) == pytest.approx(0.0, abs=1e-6)
    # Remaining points stay near the ramp, not the pinned 0.
    assert float(out[-1].item()) == pytest.approx(4.0, abs=0.25)
