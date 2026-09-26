"""Qwen collate re-tokenizes ``text`` and overwrites already-cat keys.

Weight-free. Complementary to open #42 (left-pad + ``torch.cat`` tokenized
keys). After stacking per-row tensors, ``tokenized_data.update(processed_inputs)``
lets the tokenizer win. Fused traj IDs that live only in ``input_ids``
(not in ``text``) are discarded, so CE / policy masks would supervise the
placeholder band instead of the fused tokens.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def test_source_updates_from_tokenizer_after_cat():
    src = _PROC.read_text()
    body = src.split("def collate_fn")[1].split("def _preprocess_data")[0]
    cat_idx = body.index("torch.cat([row[k] for row in batched_data")
    tok_idx = body.index("self.processor.tokenizer(")
    upd_idx = body.index("tokenized_data.update(processed_inputs)")
    assert cat_idx < tok_idx < upd_idx
    assert "padding_side=padding_side" in body
    assert "padding=True" in body


def test_update_overwrites_fused_input_ids():
    tokenized_data = {"input_ids": [10, 11, 12], "pixel_values": [1]}
    processed_inputs = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    tokenized_data.update(processed_inputs)
    assert tokenized_data["input_ids"] == [1, 2, 3]
    assert tokenized_data["pixel_values"] == [1]
    assert tokenized_data["attention_mask"] == [1, 1, 1]
