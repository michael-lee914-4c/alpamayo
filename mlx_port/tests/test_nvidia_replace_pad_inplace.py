"""``replace_padding_after_eos`` writes padding into the caller's tensor.

Weight-free. Complementary to open #47 (no-EOS is a no-op) and #24
(first-EOS pad). The function assigns ``token_ids[mask_after] = pad``
and returns the same object. A later decode of the pre-pad IDs would
see pads if the caller kept a reference. Cloning before pad would hide
that mutation.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"


def test_source_assigns_pad_on_the_input_tensor():
    src = _TOKEN_UTILS.read_text()
    body = src.split("def replace_padding_after_eos")[1]
    assert "token_ids[mask_after] = pad_token_id" in body
    assert "return token_ids" in body
    assert "token_ids.clone()" not in body
    assert "token_ids = token_ids.clone()" not in body
