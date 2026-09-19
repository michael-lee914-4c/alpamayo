"""``get_component_str`` requires exactly one of content vs padding.

Weight-free. Complementary to open #24 (ask-for emits start only) and
#35 (unknown ``components_order`` is a silent skip). Both content and
padding, or neither, raise. A ``None``/empty swap would teacher-force
an empty CoC span or drop the future pads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CONV = _REPO / "src" / "alpamayo_r1" / "chat_template" / "conversation.py"


def get_component_str(
    start_str: str,
    end_str: str,
    content_str: str | None = None,
    padding_str: str | None = None,
    ask_for_component: bool = False,
) -> str:
    """Replica of NVIDIA ``get_component_str`` (avoids importing ``base_model``)."""
    component_str = [start_str]
    if not ask_for_component:
        assert (content_str is None) != (padding_str is None), (
            "Exactly one of content_str or padding_str must be provided"
        )
        if content_str is not None:
            component_str.append(content_str)
        elif padding_str is not None:
            component_str.append(padding_str)
        component_str.append(end_str)
    return "".join(component_str)


def test_source_uses_xor_not_or():
    src = _CONV.read_text()
    assert (
        "assert (content_str is None) != (padding_str is None)" in src
    )
    assert "Exactly one of content_str or padding_str must be provided" in src


def test_content_only_wraps_start_content_end():
    assert get_component_str("<s>", "<e>", content_str="hello") == "<s>hello<e>"


def test_padding_only_wraps_start_pad_end():
    assert get_component_str("<s>", "<e>", padding_str="pp") == "<s>pp<e>"


def test_both_content_and_padding_raise():
    with pytest.raises(AssertionError, match="Exactly one"):
        get_component_str("<s>", "<e>", content_str="hello", padding_str="pp")


def test_neither_content_nor_padding_raise():
    with pytest.raises(AssertionError, match="Exactly one"):
        get_component_str("<s>", "<e>")


def test_ask_for_emits_start_only_even_with_both():
    assert (
        get_component_str(
            "<s>", "<e>", content_str="hello", padding_str="pp", ask_for_component=True
        )
        == "<s>"
    )
    assert get_component_str("<s>", "<e>", ask_for_component=True) == "<s>"
