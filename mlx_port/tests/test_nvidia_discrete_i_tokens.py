"""Qwen traj vocab is ``<i0>`` … ``<i{V-1}>``, not 1-indexed and not ``<iV>``.

Weight-free. Complementary to open #34 (``add_special_tokens`` default
False vs Qwen always SPECIAL_TOKENS). ``traj_token_start_idx`` is ``<i0>``
and ``traj_token_end_idx`` is the last bin. A 1-based range or an extra
``<iV>`` would shift every fused waypoint id by one bin.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"
_VLM = _REPO / "mlx_port" / "vlm_loader.py"


def _discrete_tokens(traj_vocab_size: int) -> list[str]:
    return [f"<i{v}>" for v in range(traj_vocab_size)]


def test_source_adds_i0_through_i_last_then_looks_them_up():
    src = _PROC.read_text()
    assert 'discrete_tokens = [f"<i{v}>" for v in range(self.traj_vocab_size)]' in src
    assert 'tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")' in src
    assert 'f"<i{self.traj_vocab_size - 1}>"' in src
    assert "assert len(discrete_tokens) == num_new_tokens" in src
    # Special tokens are added after the discrete bins, as specials.
    assert "tokenizer.add_tokens(special_tokens, special_tokens=True)" in src


def test_vocab_is_zero_indexed_and_excludes_i_vocab():
    tokens = _discrete_tokens(8)
    assert tokens[0] == "<i0>"
    assert tokens[-1] == "<i7>"
    assert "<i8>" not in tokens
    assert "<i1>" == tokens[1]
    assert len(tokens) == 8
    assert len(set(tokens)) == 8


def test_mlx_loader_uses_the_same_last_bin_name():
    src = _VLM.read_text()
    assert 'f"<i{traj_vocab_size - 1}>"' in src
