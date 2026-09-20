"""``construct_cot`` teacher-forces ``data['cot']`` or emits start only.

Weight-free. Complementary to open #25 (``construct_meta_action`` missing
key / ask-start) and #36 (content-vs-padding XOR). CoC uses a different
key and the same ask-for contract: a missing ``cot`` on the train path
raises; gen-mode last-component ask must not append ``<|cot_end|>``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SPECIAL_TOKEN_KEYS = [
    "traj_history_start",
    "traj_history_end",
    "traj_history",
    "traj_future_start",
    "traj_future_end",
    "traj_future",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
]


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_base_model_specials() -> None:
    _ensure_src_on_path()
    if "alpamayo_r1.models.base_model" in sys.modules:
        return
    try:
        import alpamayo_r1.models.base_model  # noqa: F401

        return
    except Exception:
        sys.modules.pop("alpamayo_r1.models.base_model", None)
    import alpamayo_r1  # noqa: F401

    models = ModuleType("alpamayo_r1.models")
    models.__path__ = [str(_SRC / "alpamayo_r1" / "models")]
    sys.modules["alpamayo_r1.models"] = models
    sys.modules["alpamayo_r1.models.base_model"] = SimpleNamespace(
        SPECIAL_TOKENS={k: f"<|{k}|>" for k in _SPECIAL_TOKEN_KEYS},
        IGNORE_INDEX=-100,
    )


def _load_conversation():
    _ensure_src_on_path()
    _ensure_base_model_specials()
    path = _SRC / "alpamayo_r1" / "chat_template" / "conversation.py"
    spec = importlib.util.spec_from_file_location("nv_conversation_cot", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nv_conversation_cot"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA conversation unavailable: {exc}")
    return module


def test_source_teacher_forces_cot_key_and_ask_skips_end():
    src = (_SRC / "alpamayo_r1" / "chat_template" / "conversation.py").read_text()
    assert 'assert "cot" in data' in src
    assert "cot not found in data but `cot` in `components_order`" in src
    assert "content_str=cot" in src
    assert "ask_for_component=ask_for_component" in src


def test_construct_cot_teacher_force_wraps_content_and_end():
    conv = _load_conversation()
    filled = conv.construct_cot({"cot": "slow for pedestrians"}, ask_for_component=False)
    assert filled[0]["text"] == (
        "<|cot_start|>slow for pedestrians<|cot_end|>"
    )


def test_construct_cot_missing_key_raises_when_not_asking():
    conv = _load_conversation()
    with pytest.raises(AssertionError, match="cot not found"):
        conv.construct_cot({}, ask_for_component=False)


def test_construct_cot_ask_emits_start_only():
    conv = _load_conversation()
    ask = conv.construct_cot({}, ask_for_component=True)
    assert ask[0]["text"] == "<|cot_start|>"
    # Even if CoC text is present, ask-for must not close the span.
    ask_filled = conv.construct_cot(
        {"cot": "slow for pedestrians"}, ask_for_component=True
    )
    assert ask_filled[0]["text"] == "<|cot_start|>"
    assert "<|cot_end|>" not in ask_filled[0]["text"]
