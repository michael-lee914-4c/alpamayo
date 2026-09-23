"""Qwen collate defaults to ``padding_side="left"``.

Weight-free. Complementary to open #38 (mixed ``generation_mode`` /
``label_components`` raise) and #41 (unstackable ``image_frames``). Right
padding would put pad tokens after the prompt and shift every generated
id, so Stage-1 CE and vLLM rollouts would score the pad column.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _left_pad(rows: list[list[int]], pad_id: int = 0) -> list[list[int]]:
    width = max(len(row) for row in rows)
    return [[pad_id] * (width - len(row)) + row for row in rows]


def test_source_collate_defaults_to_left_padding():
    src = _PROC.read_text()
    assert 'def collate_fn(self, data: list[dict[str, Any]], padding_side: str = "left")' in src
    assert 'padding_side: str = "left"' in src
    assert "padding_side=padding_side, padding=True" in src
    assert "padding_side: str = \"left\"" in src
    # Text is re-tokenized with padding; other tokenized keys are cat, not stacked.
    assert 'if k not in ["text"]:' in src
    assert 'tokenized_data[k] = torch.cat([row[k] for row in batched_data["tokenized_data"]])' in src


def test_left_pad_keeps_prompt_ids_on_the_right():
    rows = [[1, 2, 3], [7]]
    padded = _left_pad(rows, pad_id=0)
    assert padded == [[1, 2, 3], [0, 0, 7]]
    # Right pad would put 7 at column 0 and break generation alignment.
    right = [row + [0] * (3 - len(row)) for row in rows]
    assert right == [[1, 2, 3], [7, 0, 0]]
    assert padded[1][-1] == 7
    assert right[1][-1] == 0


def test_tokenized_non_text_keys_are_concatenated():
    torch = pytest.importorskip("torch")
    rows = [
        {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.zeros(1, 2, 2)},
        {"input_ids": torch.tensor([[3, 4]]), "pixel_values": torch.ones(1, 2, 2)},
    ]
    cat_ids = torch.cat([row["input_ids"] for row in rows])
    cat_pix = torch.cat([row["pixel_values"] for row in rows])
    assert cat_ids.shape == (2, 2)
    assert cat_pix.shape == (2, 2, 2)
    stacked_ids = torch.stack([row["input_ids"] for row in rows])
    assert stacked_ids.shape == (2, 1, 2)
    assert cat_ids.shape != stacked_ids.shape
