"""Velocity LS uses third-order jerk smoothness only; accel uses second-order.

Weight-free. Complementary to open #29 (``_v_to_a`` / ``_theta_v_a_to_kappa``)
and #30 (unconstrained t0 speed). Both ``dxy_theta_to_v`` variants pass
``w_smooth3=1.0`` with 1st/2nd-order weights ``None``. Swapping the order
changes the action-space (a, κ) recovered from the same (Δp, θ) and
silently forks SFT / RL tokens.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_UTILS = _REPO / "src" / "alpamayo_r1" / "action_space" / "utils.py"
_UNI = _REPO / "src" / "alpamayo_r1" / "action_space" / "unicycle_accel_curvature.py"


def _dtd_kwargs(path: Path, func_name: str) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and (
                    (isinstance(child.func, ast.Name) and child.func.id == "construct_DTD")
                    or (
                        isinstance(child.func, ast.Attribute)
                        and child.func.attr == "construct_DTD"
                    )
                ):
                    return {kw.arg: kw.value for kw in child.keywords if kw.arg}
    raise AssertionError(f"construct_DTD not found in {func_name}")


def _const(node: ast.expr):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id == "None":
        return None
    raise AssertionError(ast.dump(node))


def test_dxy_theta_to_v_is_third_order_only():
    for name in ("dxy_theta_to_v", "dxy_theta_to_v_without_v0"):
        kwargs = _dtd_kwargs(_UTILS, name)
        assert _const(kwargs["w_smooth1"]) is None
        assert _const(kwargs["w_smooth2"]) is None
        assert _const(kwargs["w_smooth3"]) == 1.0


def test_v_to_a_is_second_order_only():
    src = _UNI.read_text()
    # Inverse accel solve is jerk-smooth (2nd order), not snap-smooth (3rd).
    assert "w_smooth1=None" in src.split("def _v_to_a")[1].split("def _theta_v_a_to_kappa")[0]
    assert "w_smooth2=1.0" in src.split("def _v_to_a")[1].split("def _theta_v_a_to_kappa")[0]
    assert "w_smooth3=None" in src.split("def _v_to_a")[1].split("def _theta_v_a_to_kappa")[0]
