"""NVIDIA RL policy labels and last-start rollout decode.

Weight-free. Complementary to open #24 (empty-span zeros + batch trim)
and NVIDIA label-mask include-end — ``get_policy_input`` only marks the
generated span plus the appended ``traj_future_end``, and
``decode_rollout_trajectory`` takes the text *after the last*
``traj_future_start`` via ``rsplit``. A first-start or prompt-wide
mask would CE-train the prompt as a completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"
_DECODE = _REPO / "finetune" / "rl" / "utils" / "trajectory_decode.py"
_START = "<|traj_future_start|>"
_END = "<|traj_future_end|>"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_policy_labels_mask_only_gen_span_plus_end(torch):
    src = _PACKER.read_text()
    assert "new_mask[-(gen_ids.numel() + 1) :] = True  # include the end token" in src
    assert "tokenized[\"text\"] += SPECIAL_TOKENS[\"traj_future_end\"]" in src

    ids_row = torch.arange(6, dtype=torch.long)
    gen_ids = torch.tensor([70, 71, 72], dtype=torch.long)
    end_id = 99
    new_ids = torch.cat([ids_row, gen_ids, torch.tensor([end_id])], dim=0)
    new_mask = torch.zeros_like(new_ids, dtype=torch.bool)
    new_mask[-(gen_ids.numel() + 1) :] = True
    assert new_ids.tolist() == [0, 1, 2, 3, 4, 5, 70, 71, 72, 99]
    assert new_mask.tolist() == [False] * 6 + [True] * 4
    assert bool(new_mask[-1]) is True
    # A full-sequence True would CE the prompt prefix.
    assert not new_mask[:6].any()


def test_decode_rollout_uses_last_start_via_rsplit():
    src = _DECODE.read_text()
    assert 'rsplit(SPECIAL_TOKENS["traj_future_start"], maxsplit=1)[-1].split(' in src
    assert 'SPECIAL_TOKENS["traj_future_end"], maxsplit=1' in src

    # Two starts: first-start partition would keep "AAA <|traj_future_start|> BBB".
    text = f"prompt {_START} AAA {_START} BBB {_END} trailing"
    traj_text = text.rsplit(_START, maxsplit=1)[-1].split(_END, maxsplit=1)[0]
    assert traj_text.strip() == "BBB"
    first = text.split(_START, maxsplit=1)[-1].split(_END, maxsplit=1)[0]
    assert "AAA" in first
    assert first != traj_text


def test_decode_rollout_empty_after_last_start_is_blank():
    text = f"prompt {_START} kept {_START}{_END}"
    traj_text = text.rsplit(_START, maxsplit=1)[-1].split(_END, maxsplit=1)[0]
    assert traj_text.strip() == ""
    src = _DECODE.read_text()
    assert 'if traj_text.strip() == ""' in src
    assert "torch.zeros" in src
