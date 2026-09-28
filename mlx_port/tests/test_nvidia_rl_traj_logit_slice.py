"""RL ``MaskDiscreteTrajectoryLogitsProcessor`` zeros a half-open traj band.

Weight-free. Complementary to open #32 (SFT ``ExpertLogitsProcessor``
traj-slice ``-inf``) and #42 (token IDs clamp into the same band). The
RL processor writes ``scores[:, offset:offset + vocab] = -inf``. An
inclusive end, or masking the whole tail, would either leak a traj id
into CoC sampling or kill the EOS / text logits next to the band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_LOGIT = _REPO / "finetune" / "rl" / "utils" / "logit_processor.py"


def test_source_masks_half_open_offset_plus_vocab():
    src = _LOGIT.read_text()
    body = src.split("class MaskDiscreteTrajectoryLogitsProcessor")[1]
    assert "scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size]" in body
    assert 'float(\n            "-inf"\n        )' in body
    assert "offset + self.traj_vocab_size + 1" not in body


def test_half_open_slice_keeps_neighbors():
    scores = np.arange(8, dtype=np.float64)
    offset, vocab = 3, 3
    masked = scores.copy()
    masked[offset : offset + vocab] = -np.inf
    assert np.isneginf(masked[3:6]).all()
    assert masked[2] == 2.0
    assert masked[6] == 6.0
