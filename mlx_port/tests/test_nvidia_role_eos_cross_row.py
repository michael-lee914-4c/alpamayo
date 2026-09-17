"""``get_role_eos_mask`` pairs BOS/EOS in global flatten order.

Weight-free. Complementary to open #19 (pairing / trailing-BOS) and #24
(infer-style extra-BOS). ``torch.where`` is row-major across the whole
``[B, L]`` tensor. Uneven per-row BOS/EOS counts pair a BOS on row 0
with an EOS on row 1, so an assistant EOS can land on the wrong sample.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MASK = _REPO / "src" / "alpamayo_r1" / "utils" / "get_label_mask.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _role_eos_mask(input_ids, bos_id: int, eos_id: int, role_id: int):
    """NVIDIA pairing: global where() then zip-by-count, not per row."""
    eos_mask = input_ids.new_zeros(input_ids.shape, dtype=bool)
    bos_positions = (input_ids == bos_id).nonzero(as_tuple=True)
    # torch.where returns the same order as nonzero for a bool mask.
    eos_positions = (input_ids == eos_id).nonzero(as_tuple=True)
    num_bos = bos_positions[0].shape[0]
    num_eos = eos_positions[0].shape[0]
    if num_eos < num_bos:
        bos_positions = (bos_positions[0][:num_eos], bos_positions[1][:num_eos])
    role_tokens = input_ids[bos_positions[0], bos_positions[1] + 1]
    assistant_batch_idx = eos_positions[0][role_tokens == role_id]
    assistant_eos_idx = eos_positions[1][role_tokens == role_id]
    eos_mask[assistant_batch_idx, assistant_eos_idx] = True
    return eos_mask


def test_source_pairs_global_where_not_per_row():
    src = _MASK.read_text()
    assert "bos_positions = torch.where(input_ids == bos_token_id)" in src
    assert "eos_positions = torch.where(input_ids == eos_token_id)" in src
    assert "if num_eos < num_bos:" in src
    assert "bos_positions = (bos_positions[0][:num_eos], bos_positions[1][:num_eos])" in src
    assert "role_tokens = input_ids[bos_positions[0], bos_positions[1] + 1]" in src
    assert "assistant_batch_idx = eos_positions[0][role_tokens == role_id]" in src


def test_uneven_rows_pair_bos_on_row0_with_eos_on_row1(torch):
    bos, eos, role, user, pad = 1, 2, 3, 4, 0
    # Row 0: BOS assistant ... (no EOS). Row 1: BOS user ... EOS.
    # Global pairing: the only BOS (row0 assistant) is zipped with the
    # only EOS (row1), so the assistant-EOS bit is set on row 1.
    ids = torch.tensor(
        [
            [bos, role, 10, 11, pad],
            [bos, user, 20, eos, pad],
        ],
        dtype=torch.long,
    )
    got = _role_eos_mask(ids, bos, eos, role)
    assert got.tolist() == [
        [False, False, False, False, False],
        [False, False, False, True, False],
    ]
    # A per-row implementation would leave both rows False: row 0 has no
    # EOS, row 1's BOS is the user role.
    per_row = torch.zeros_like(got)
    assert not per_row.any()
    assert got.any() and int(got.nonzero()[0, 0]) == 1
