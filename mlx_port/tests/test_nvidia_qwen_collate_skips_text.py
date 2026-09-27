"""Qwen collate ``torch.cat``s every tokenized key except ``text``.

Weight-free. Complementary to open #46 (``update(processed_inputs)``
overwrites fused ``input_ids``) and #42 (left-pad + ``torch.cat``).
``text`` is collected separately for the tokenizer pad pass; if it were
``torch.cat``'d first the collate would crash on strings, and if another
ragged key were added without the skip list it would crash the same way.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def test_source_skips_text_before_torch_cat():
    src = _QWEN.read_text()
    body = src.split("def collate_fn")[1].split("def _preprocess_data")[0]
    assert 'if k not in ["text"]:' in body
    assert 'tokenized_data[k] = torch.cat([row[k] for row in batched_data["tokenized_data"]])' in body
    assert 'batch_text = [instance["text"] for instance in batched_data["tokenized_data"]]' in body


def test_text_skip_is_only_the_string_key():
    keys = ["input_ids", "attention_mask", "pixel_values", "text", "image_grid_thw"]
    stacked = [k for k in keys if k not in ["text"]]
    assert "text" not in stacked
    assert stacked == ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
