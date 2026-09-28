"""Delta encode pads the time axis (F.pad last-dim-first), not XYZ.

Weight-free. Complementary to open #26 (first-step origin pad + clamp)
and #45 (decode ignores hist / cumsum from origin).
``F.pad(fut_xyz, [0, 0, 1, 0, 0, 0])`` adds one leading time step of
zeros so ``xyz[:, 1:] - xyz[:, :-1]`` is origin-relative. Padding XYZ
(``[1, 0, 0, 0, 0, 0]``) would shift channels and tokenize garbage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DELTA = _REPO / "src" / "alpamayo_r1" / "models" / "delta_tokenizer.py"


def test_source_pads_time_then_diffs():
    src = _DELTA.read_text()
    body = src.split("def encode(")[1].split("def decode(")[0]
    assert "torch.nn.functional.pad(fut_xyz, [0, 0, 1, 0, 0, 0])" in body
    assert "xyz = xyz[:, 1:] - xyz[:, :-1]" in body
    assert "[1, 0, 0, 0, 0, 0]" not in body


def test_last_dim_first_pad_inserts_origin_row():
    # (B=1, T=2, 3); pad spec last-dim-first: (0,0) xyz, (1,0) time, (0,0) batch.
    fut = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    padded = np.pad(fut, ((0, 0), (1, 0), (0, 0)))
    assert padded.shape == (1, 3, 3)
    assert padded[0, 0].tolist() == [0.0, 0.0, 0.0]
    delta = padded[:, 1:] - padded[:, :-1]
    assert delta[0, 0].tolist() == [1.0, 2.0, 3.0]
    wrong = np.pad(fut, ((0, 0), (0, 0), (1, 0)))
    assert wrong.shape == (1, 2, 4)
