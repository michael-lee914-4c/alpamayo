"""Assistant-span mask asserts paired BOS/EOS; role-EOS truncates extras.

Weight-free. Complementary to open #17 (role-EOS pairing / trailing-BOS)
and #38 (assistant-mask BOS/EOS pair + START_OFFSET=3).
``get_assistant_mask`` raises when ``len(start) != len(end)`` — typical
of an open gen-mode assistant block. ``get_role_eos_mask`` instead
truncates BOS to ``num_eos`` so incomplete generations still get a
partial EOS mask. Mixing the two on infer-shaped sequences would either
crash SFT labels or supervise a phantom EOS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_MASK = _REPO / "src" / "alpamayo_r1" / "utils" / "get_label_mask.py"


def test_source_assistant_mask_requires_equal_counts():
    src = _MASK.read_text()
    body = src.split("def get_assistant_mask")[1].split("def get_role_eos_mask")[0]
    assert "assert len(start_indices) == len(end_indices)" in body


def test_source_role_eos_truncates_extra_bos():
    src = _MASK.read_text()
    body = src.split("def get_role_eos_mask")[1]
    assert "if num_eos < num_bos:" in body
    assert "bos_positions[0][:num_eos]" in body
    assert "bos_positions[1][:num_eos]" in body
    assert "assert len(start_indices) == len(end_indices)" not in body


def test_unpaired_assistant_block_is_infer_shaped():
    # user BOS/EOS + open assistant BOS (no EOS yet)
    bos, eos, role = 1, 2, 3
    tokens = np.array([bos, 4, 5, eos, bos, role, 7, 8])
    n_bos = int((tokens == bos).sum())
    n_eos = int((tokens == eos).sum())
    assert n_bos == 2 and n_eos == 1
    # assistant-mask contract: this sequence is illegal
    assert n_bos != n_eos
    # role-EOS contract: keep only the first BOS so the closed user turn pairs
    kept_bos = np.where(tokens == bos)[0][:n_eos]
    kept_eos = np.where(tokens == eos)[0]
    assert kept_bos.tolist() == [0]
    assert kept_eos.tolist() == [3]
