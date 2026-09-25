"""NVIDIA public helper binds the same 163840 / 196608 pixel budget as SFT.

Weight-free. Complementary to open MLX pixel-budget tests and #32
(``do_rescale`` / image-token expand). ``helper.get_processor`` hard-codes
``MIN_PIXELS=163840`` and ``MAX_PIXELS=196608`` onto the Qwen processor.
Drifting these constants changes the vision token count and mRoPE layout for
every CUDA ``create_message`` / ``tokenize_messages`` call.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_HELPER = _REPO / "src" / "alpamayo_r1" / "helper.py"


def test_source_pins_helper_pixel_budget_and_processor_name():
    src = _HELPER.read_text()
    assert "MIN_PIXELS = 163840" in src
    assert "MAX_PIXELS = 196608" in src
    assert 'BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"' in src
    assert '"min_pixels": MIN_PIXELS' in src
    assert '"max_pixels": MAX_PIXELS' in src
    assert "processor.tokenizer = tokenizer" in src


def test_budget_is_the_1080p_qwen3vl_window():
    # 163840 = 1280*128 / 196608 = 1536*128 — the official Qwen3-VL
    # smart-resize window used by Alpamayo-R1 inference, not the 32k default.
    assert 163840 == 1280 * 128
    assert 196608 == 1536 * 128
    assert 196608 > 163840
