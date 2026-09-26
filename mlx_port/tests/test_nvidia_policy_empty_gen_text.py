"""Policy packer always appends rollout text, even when ``gen_ids`` is empty.

Weight-free. Complementary to open #33 (empty ``gen_ids`` skip mask/id
extend) and #36 (policy ``cat(ids, gen, end)``). ``tokenized[\"text\"]``
gains ``rollout_output + <|traj_future_end|>`` unconditionally; tensors
stay put when the completion is empty. Log / re-tokenize paths that read
``text`` then disagree with ``input_ids``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"
_END = "<|traj_future_end|>"


def test_source_appends_text_before_the_gen_ids_guard():
    src = _PACKER.read_text()
    body = src.split("def get_policy_input")[1].split("def policy_compute_max_len")[0]
    text_idx = body.index('tokenized["text"] += rollout_output')
    end_idx = body.index('tokenized["text"] += SPECIAL_TOKENS["traj_future_end"]')
    guard_idx = body.index("if gen_ids.numel() > 0:")
    assert text_idx < end_idx < guard_idx
    assert "new_ids_row = torch.cat" in body[guard_idx:]
    assert "new_ids_row = torch.cat" not in body[:guard_idx]


def test_empty_completion_lengthens_text_only():
    tokenized = {"text": "<prompt>", "input_ids": [1, 2, 3]}
    rollout_output = ""
    gen_ids_empty = True
    if "text" in tokenized and isinstance(tokenized["text"], str):
        tokenized["text"] += rollout_output
        tokenized["text"] += _END
    if not gen_ids_empty:
        tokenized["input_ids"] = tokenized["input_ids"] + [99]
    assert tokenized["text"] == "<prompt>" + _END
    assert tokenized["input_ids"] == [1, 2, 3]
