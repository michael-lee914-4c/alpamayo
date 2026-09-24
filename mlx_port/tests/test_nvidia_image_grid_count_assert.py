"""Qwen preprocess asserts image-token count equals ``image_grid_thw`` rows.

Weight-free. Complementary to open #32 (``grid.prod() // merge_size**2``
expand). The while-loop replaces one ``image_token`` per grid row, then
``assert index == len(image_grid_thw)``. Extra placeholders or a missing
grid row raise instead of silently leaving unexpanded tokens that blow
the prefix length.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _expand_image_tokens(text, image_token, image_grid_thw, merge_size):
    index = 0
    merge_length = merge_size**2
    while image_token in text:
        text = text.replace(
            image_token,
            "<|placeholder|>" * (int(image_grid_thw[index].prod()) // merge_length),
            1,
        )
        index += 1
    text = text.replace("<|placeholder|>", image_token)
    assert index == len(image_grid_thw)
    return text, index


def test_source_asserts_index_equals_grid_rows():
    src = _QWEN.read_text()
    assert "while processor.image_token in text:" in src
    assert 'image_inputs["image_grid_thw"][index].prod() // merge_length' in src
    assert "assert index == len(image_inputs[\"image_grid_thw\"])" in src
    assert "merge_length = processor.image_processor.merge_size**2" in src


def test_matching_count_expands_one_row_per_token():
    torch = pytest.importorskip("torch")
    token = "<|image_pad|>"
    text = f"pre {token} mid {token} post"
    grid = torch.tensor([[1, 2, 2], [1, 4, 2]])  # prod 4 and 8; merge=2 → 1 and 2 tokens
    out, index = _expand_image_tokens(text, token, grid, merge_size=2)
    assert index == 2
    assert out.count(token) == 1 + 2
    assert "<|placeholder|>" not in out


def test_extra_placeholder_raises_index_error_before_assert():
    torch = pytest.importorskip("torch")
    token = "<|image_pad|>"
    text = f"{token}{token}{token}"
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    with pytest.raises(IndexError):
        _expand_image_tokens(text, token, grid, merge_size=2)


def test_missing_placeholder_fails_count_assert():
    torch = pytest.importorskip("torch")
    token = "<|image_pad|>"
    text = f"only {token}"
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    with pytest.raises(AssertionError):
        _expand_image_tokens(text, token, grid, merge_size=2)
