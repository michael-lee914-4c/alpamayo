"""``build_conversation`` silently skips unknown ``components_order`` names.

Weight-free. Complementary to open #34 (``construct_user_prompt`` drops
unknown names to ``\".\"``). The ``match`` has no default arm, so a typo
or a user-side last component never becomes assistant content and never
raises. Generation ``ask_for_component`` is ``component == last`` even
when ``last`` is ``prompt`` / ``image``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CONV = _REPO / "src" / "alpamayo_r1" / "chat_template" / "conversation.py"
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _get_component_str(start_str, end_str, content_str=None, padding_str=None, ask_for_component=False):
    component_str = [start_str]
    if not ask_for_component:
        assert (content_str is None) != (padding_str is None)
        if content_str is not None:
            component_str.append(content_str)
        elif padding_str is not None:
            component_str.append(padding_str)
        component_str.append(end_str)
    return "".join(component_str)


def _build_conversation(data, components_order, generation_mode=False, future_pads=2):
    user, assistant = [], []
    last_component = components_order[-1]
    for component in components_order:
        ask_for_component = generation_mode and component == last_component
        match component:
            case "prompt":
                user.append("PROMPT")
            case "traj_history":
                user.append("HIST")
            case "cot":
                if not ask_for_component:
                    assert "cot" in data
                assistant.append(
                    _get_component_str(
                        "<|cot_start|>",
                        "<|cot_end|>",
                        content_str=None if ask_for_component else data.get("cot"),
                        ask_for_component=ask_for_component,
                    )
                )
            case "meta_action":
                if not ask_for_component:
                    assert "meta_action_strings" in data
                assistant.append(
                    _get_component_str(
                        "<|meta_action_start|>",
                        "<|meta_action_end|>",
                        content_str=None if ask_for_component else data.get("meta_action_strings"),
                        ask_for_component=ask_for_component,
                    )
                )
            case "traj_future":
                assistant.append(
                    _get_component_str(
                        "<|traj_future_start|>",
                        "<|traj_future_end|>",
                        padding_str="<|traj_future|>" * future_pads,
                        ask_for_component=ask_for_component,
                    )
                )
    return user, assistant


def test_source_match_has_no_default_and_last_component_ask():
    src = _CONV.read_text()
    assert "last_component = components_order[-1]" in src
    assert "ask_for_component = generation_mode and component == last_component" in src
    assert "match component:" in src
    assert "case _:" not in src
    assert 'case "prompt":' in src
    assert 'case "traj_future":' in src
    proc = _PROC.read_text()
    assert "continue_final_message=generation_mode" in proc
    assert "add_generation_prompt=False" in proc
    assert 'padding_side: str = "left"' in proc


def test_unknown_component_adds_neither_user_nor_assistant():
    user, assistant = _build_conversation(
        {"cot": "go"},
        ["prompt", "traj_futre", "not_a_component", "cot"],
    )
    assert user == ["PROMPT"]
    assert assistant == ["<|cot_start|>go<|cot_end|>"]


def test_generation_last_user_component_still_teacher_forces_assistant():
    # Last name is prompt, so no assistant span is asked-for.
    user, assistant = _build_conversation(
        {"cot": "yield"},
        ["cot", "prompt"],
        generation_mode=True,
    )
    assert user == ["PROMPT"]
    assert assistant == ["<|cot_start|>yield<|cot_end|>"]


def test_generation_last_assistant_is_start_only():
    _, assistant = _build_conversation(
        {"cot": "unused"},
        ["cot", "traj_future"],
        generation_mode=True,
        future_pads=3,
    )
    assert assistant[0] == "<|cot_start|>unused<|cot_end|>"
    assert assistant[1] == "<|traj_future_start|>"
    assert "<|traj_future_end|>" not in assistant[1]
