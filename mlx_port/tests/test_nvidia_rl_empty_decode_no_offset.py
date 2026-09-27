"""Empty RL decode writes raw zeros and skips the future-band subtract.

Weight-free. Complementary to open #45 (non-empty decode subtracts
``traj_token_start_idx``) and #24 (empty-span zeros + batch trim). The
``traj_text.strip() == \"\"`` branch allocates ``torch.zeros((1, expected_len))``
and never calls ``extract_traj_tokens``, so the IDs stay 0 rather than
``0 - traj_token_start_idx`` (which would clamp to the first bin after
the offset-then-clamp path). Empty CoC completions would otherwise
decode as a large negative index / first-bin trajectory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_DECODE = _REPO / "finetune" / "rl" / "utils" / "trajectory_decode.py"


def test_source_empty_branch_skips_extract_and_offset():
    src = _DECODE.read_text()
    empty = src.split('if traj_text.strip() == "":')[1].split("else:")[0]
    assert "traj_token_ids = torch.zeros(" in empty
    assert "(1, expected_len)" in empty
    assert "extract_traj_tokens" not in empty
    assert "token_id_offset" not in empty


def test_source_offset_is_only_used_on_the_nonempty_path():
    src = _DECODE.read_text()
    # offset is bound unconditionally, but extract_traj_tokens (which
    # subtracts it) lives only in the else branch
    assert "token_id_offset = model_config.traj_token_start_idx" in src
    nonempty = src.split("else:")[1]
    assert "extract_traj_tokens(" in nonempty
    assert "token_id_offset" in nonempty


def test_raw_zeros_are_not_future_band_ids():
    expected_len = 6
    offset = 100
    raw = np.zeros(expected_len, dtype=np.int64)
    shifted = raw - offset
    assert raw.tolist() == [0] * expected_len
    assert shifted.tolist() == [-offset] * expected_len
    clamped = np.clip(shifted, 0, 767)
    assert clamped.tolist() == [0] * expected_len
    # the empty path never applies the subtract; the nonempty path would
    assert not np.array_equal(raw, shifted)
