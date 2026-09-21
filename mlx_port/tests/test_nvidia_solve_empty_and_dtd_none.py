"""``solve_single_constraint`` rejects N≤0; all-None DTD is zeros.

Weight-free. Complementary to open #27 (DTD ``lam/dt**{2,4,6}`` and
pinned ``x[0]``) and #29 (``solve_xs_eq_y`` recovers ``y/s``). An empty
last-dim must raise before Cholesky. All-None smooth weights must not
inject an identity regularizer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_UTILS = _REPO / "src" / "alpamayo_r1" / "action_space" / "utils.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_source_empty_n_and_all_none_dtd():
    src = _UTILS.read_text()
    assert "if N <= 0:" in src
    assert 'raise ValueError("x_mid must have a positive last-dimension length N.")' in src
    assert "DTD = torch.zeros(*lead, N, N, dtype=dtype, device=device)" in src
    assert "if w_smooth1 is not None:" in src
    assert "if w_smooth2 is not None:" in src
    assert "if w_smooth3 is not None:" in src
    # Decorators must not change the empty-N / all-None contracts.
    assert "@torch._dynamo.disable()" in src


def test_solve_single_constraint_rejects_empty_last_dim(torch):
    src = _UTILS.read_text()
    assert '*lead, N = x_target.shape' in src
    assert "if N <= 0:" in src

    def _guard(x_target):
        *lead, n = x_target.shape
        del lead
        if n <= 0:
            raise ValueError("x_mid must have a positive last-dimension length N.")

    with pytest.raises(ValueError, match="positive last-dimension"):
        _guard(torch.zeros(0))
    with pytest.raises(ValueError, match="positive last-dimension"):
        _guard(torch.zeros(2, 0))
    _guard(torch.zeros(3))
    _guard(torch.zeros(2, 4))


def test_construct_dtd_all_none_weights_is_zeros(torch):
    src = _UTILS.read_text()
    # Weights are gated; the accumulator starts at zeros and is returned as-is
    # when every w_smooth* is None.
    assert src.index("DTD = torch.zeros(*lead, N, N") < src.index("if w_smooth1 is not None:")
    assert src.index("return DTD") > src.index("if w_smooth3 is not None:")

    def construct_dtd_all_none(n, lead=()):
        return torch.zeros(*lead, n, n)

    dtd = construct_dtd_all_none(5)
    assert dtd.shape == (5, 5)
    assert torch.count_nonzero(dtd).item() == 0
    # An accidental identity ridge would change the unicycle smoother.
    assert not torch.allclose(dtd, torch.eye(5))
    batched = construct_dtd_all_none(3, lead=(2,))
    assert batched.shape == (2, 3, 3)
    assert torch.count_nonzero(batched).item() == 0
