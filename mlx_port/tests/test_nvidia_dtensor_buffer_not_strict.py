"""FSDP2 ``copy_state_into_dtensor_shards`` is strict only for parameters.

Weight-free. Complementary to open #42 (``pad_to_match`` last-dim pad).
Missing parameter keys raise ``KeyError`` when ``strict=True``; missing
*buffers* always ``continue``, even under strict. A buffer-only expert
head can stay at init while the VLM loads, with no load error.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_WL = _REPO / "finetune" / "rl" / "utils" / "weight_loading.py"


def _copy_fn() -> ast.FunctionDef:
    tree = ast.parse(_WL.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "copy_state_into_dtensor_shards":
            return node
    raise AssertionError("copy_state_into_dtensor_shards not found")


def _loop_for_call(fn: ast.FunctionDef, method: str) -> ast.For:
    for node in ast.walk(fn):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            func = node.iter.func
            if isinstance(func, ast.Attribute) and func.attr == method:
                return node
    raise AssertionError(method)


def test_source_strict_keyerror_is_only_on_parameters():
    src = _WL.read_text()
    assert 'raise KeyError(f"Checkpoint missing parameter: {full_key}")' in src
    assert "strict: If True, raise KeyError when a parameter is missing" in src


def test_parameter_loop_honors_strict_flag():
    param_loop = _loop_for_call(_copy_fn(), "named_parameters")
    src = ast.unparse(param_loop)
    assert "if strict:" in src
    assert "KeyError" in src


def test_buffer_loop_never_checks_strict():
    buf_loop = _loop_for_call(_copy_fn(), "named_buffers")
    src = ast.unparse(buf_loop)
    assert "strict" not in src
    assert "KeyError" not in src
    assert "continue" in src
