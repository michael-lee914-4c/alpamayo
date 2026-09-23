"""Generation ask for future traj emits only ``<|traj_future_start|>``.

Weight-free. Complementary to open #38 (CoT missing-key / ask-start-only)
and #36 (component XOR). The last assistant component in gen mode must not
teacher-force pads or ``<|traj_future_end|>`` — that would stop decode
before any waypoint tokens.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CONV = _REPO / "src" / "alpamayo_r1" / "chat_template" / "conversation.py"
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


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


def _construct_traj_future(num_tokens_per_future_traj: int, ask_for_component: bool = False) -> str:
    return _get_component_str(
        start_str="<|traj_future_start|>",
        end_str="<|traj_future_end|>",
        padding_str="<|traj_future|>" * num_tokens_per_future_traj,
        ask_for_component=ask_for_component,
    )


def test_source_ask_skips_end_and_only_last_component():
    conv = _CONV.read_text()
    base = _BASE.read_text()
    assert "if not ask_for_component:" in conv
    assert "ask_for_component = generation_mode and component == last_component" in conv
    assert 'last_component = components_order[-1]' in conv
    assert '"future_start": "<|traj_future_start|>"' in base
    assert '"future_end": "<|traj_future_end|>"' in base


def test_ask_emits_only_start_even_when_padding_is_passed():
    asked = _construct_traj_future(48, ask_for_component=True)
    assert asked == "<|traj_future_start|>"
    assert "<|traj_future|>" not in asked
    assert "<|traj_future_end|>" not in asked


def test_teacher_force_keeps_pads_and_end():
    forced = _construct_traj_future(4, ask_for_component=False)
    assert forced == (
        "<|traj_future_start|>"
        + "<|traj_future|>" * 4
        + "<|traj_future_end|>"
    )
    assert forced.count("<|traj_future|>") == 4
