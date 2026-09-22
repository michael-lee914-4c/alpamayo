"""Train prompts walk ``components_order``; gen walks ``components_prompt``.

Weight-free. Complementary to open #34 (unknown / user-side names →
``"."``). If the two lists differ, swapping the source would teacher-force
a gen instruction during SFT or drop ``traj_future`` from the user ask.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_CONV = _SRC / "alpamayo_r1" / "chat_template" / "conversation.py"
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
    spec = importlib.util.spec_from_file_location("nv_conversation_prompt_src", _CONV)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_CONV}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nv_conversation_prompt_src"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA conversation unavailable: {exc}")
    return module


def test_source_switches_on_generation_mode():
    src = _CONV.read_text()
    assert "components = components_prompt if generation_mode else components_order" in src
    assert 'template = {' in src
    assert '"traj_future": "output the future trajectory"' in src
    assert 'prompt = ", then ".join(prompt_components) + "."' in src


def test_train_uses_order_gen_uses_prompt():
    conv = _load_conversation()
    order = ["cot", "traj_future"]
    prompt = ["traj_future"]
    train = conv.construct_user_prompt(order, prompt, generation_mode=False)
    gen = conv.construct_user_prompt(order, prompt, generation_mode=True)
    assert train[0]["text"] == (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
    assert gen[0]["text"] == "output the future trajectory."
    assert train[0]["text"] != gen[0]["text"]
