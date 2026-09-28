"""Dummy ego LWH ``(4, 3, 2)`` is duplicated in constants and metric_api.

Weight-free. Complementary to open #15 (``EGO_VEHICLE_LWH=(4,3,2)``)
and #36 (corners + camera map). ``DistanceMetrics`` boxes use
``metric_api.EGO_VEHICLE_LWH`` while other helpers read
``common.constants.EGO_VEHICLE_LWH``. Drifting one copy changes
corner-distance eval without touching the other path.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CONSTANTS = _REPO / "src" / "alpamayo_r1" / "common" / "constants.py"
_METRIC_API = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_api.py"


def _assigned_tuple(src: str, name: str) -> tuple:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name and node.value is not None:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def test_both_copies_are_four_three_two_meters():
    constants_lwh = _assigned_tuple(_CONSTANTS.read_text(), "EGO_VEHICLE_LWH")
    metrics_lwh = _assigned_tuple(_METRIC_API.read_text(), "EGO_VEHICLE_LWH")
    assert constants_lwh == (4.0, 3.0, 2.0)
    assert metrics_lwh == (4.0, 3.0, 2.0)
    assert constants_lwh == metrics_lwh
