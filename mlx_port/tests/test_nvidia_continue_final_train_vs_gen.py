"""Qwen preprocess sets ``continue_final_message`` from ``generation_mode``.

Weight-free. Complementary to open #26 (``tokenize_messages`` defaults
``continue_final=True`` / ``add_generation_prompt=False``). SFT preprocess
uses ``continue_final_message=generation_mode``, so train mode is False
and generation is True. Flipping that appends or drops the assistant
suffix and silently retargets every CE label.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"
_HELPER = _REPO / "src" / "alpamayo_r1" / "helper.py"


def test_preprocess_wires_continue_final_to_generation_mode():
    src = _QWEN.read_text()
    assert "tokenize=False" in src
    assert "add_generation_prompt=False" in src
    assert "continue_final_message=generation_mode" in src
    assert "continue_final_message=True" not in src
    assert "continue_final_message=False" not in src


def test_helper_tokenize_defaults_stay_continue_final_true():
    src = _HELPER.read_text()
    assert '"add_generation_prompt": False' in src
    assert '"continue_final_message": True' in src


def test_train_vs_gen_kwargs_are_opposites():
    """Replica of the apply_chat_template kwargs in ``_preprocess_data``."""

    def preprocess_kwargs(generation_mode: bool) -> dict:
        return {
            "tokenize": False,
            "add_generation_prompt": False,
            "continue_final_message": generation_mode,
        }

    train = preprocess_kwargs(False)
    gen = preprocess_kwargs(True)
    assert train["continue_final_message"] is False
    assert gen["continue_final_message"] is True
    assert train["add_generation_prompt"] is False
    assert gen["add_generation_prompt"] is False
    assert train["tokenize"] is False
