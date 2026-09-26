"""``solve_xs_eq_y`` multiplies ridge by 10 until Cholesky succeeds.

Weight-free. Complementary to open #29 (recovers y/s + ``w_data`` shape).
A singular heading/velocity system (``s=0``, no terminal constraint)
retries with ``ridge *= 10`` in an unbounded ``while L is None`` loop.
Dropping the retry (or using a fixed ridge) changes curvature/speed
recovered from the same waypoints.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_UTILS = _REPO / "src" / "alpamayo_r1" / "action_space" / "utils.py"


def test_source_retries_cholesky_by_scaling_ridge():
    src = _UTILS.read_text()
    body = src.split("def solve_xs_eq_y")[1].split("def dxy_theta_to_v_without_v0")[0]
    assert "while L is None:" in body
    assert "ridge *= 10" in body
    assert "torch.linalg.cholesky(lhs)" in body
    assert "Resolving singularity using ridge" in body


def test_retry_loop_is_unbounded_in_source():
    tree = ast.parse(_UTILS.read_text())
    fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "solve_xs_eq_y"
    )
    whiles = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert len(whiles) == 1
    test = whiles[0].test
    assert isinstance(test, ast.Compare)
    assert ast.unparse(test) == "L is None"
    # No max-iter counter — only ridge growth.
    assigns = [
        ast.unparse(n)
        for n in ast.walk(fn)
        if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == "ridge"
    ]
    assert assigns == ["ridge *= 10"]


def test_zero_slope_needs_ridge_to_be_invertible():
    # ATA is all zeros when s=0; without ridge the normal matrix is singular.
    n = 4
    ata = np.zeros((n, n))
    assert abs(np.linalg.det(ata)) < 1e-12
    ridge = 0.0
    for _ in range(6):
        ridge = 1e-4 if ridge == 0.0 else ridge * 10
        lhs = ata + ridge * np.eye(n)
        if abs(np.linalg.det(lhs)) > 1e-12:
            break
    else:
        raise AssertionError("ridge retry never regularized the singular system")
    assert ridge >= 1e-4
