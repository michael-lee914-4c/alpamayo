"""RL rollout requires history tensors and falls back to ``text``.

Weight-free. Complementary to open #32 (rank-3 ego unsqueeze / image-pad
collapse) and #36 (policy ``input_ids`` ``[L]``/``[1, L]``). Rollout
raises without history, encodes ``tokenized_data['text']`` when ids are
missing, and requires the fused row to be ``[1, T]`` before ``tolist()``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PACKER = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "data_packer.py"
_BASE_PACKER = _REPO / "finetune" / "rl" / "base_data_packer.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_source_rollout_requires_hist_and_text_fallback():
    src = _PACKER.read_text()
    assert 'td = sample.get("tokenized_data")' in src
    assert "if td is None:" in src
    assert 'raw_text = td.get("text")' in src
    assert "input_ids = torch.tensor(alp_tok.encode(cast(str, raw_text)), dtype=torch.long)" in src
    assert (
        'raise ValueError("Expected ego_history_xyz/ego_history_rot tensors for traj fusion.")'
        in src
    )
    assert 'raise ValueError("Expected fused_input_ids to be a 2D tensor [1, T].")' in src
    assert "token_ids = fused_input_ids[0].tolist()" in src
    # Policy uses the same text fallback, then rejects anything but [1, L].
    assert "tokenized_data.input_ids must be a torch.Tensor of shape [1, L]" in src


def test_rollout_missing_history_raises(torch):
    src = _PACKER.read_text()
    assert "if isinstance(ego_history_xyz, torch.Tensor) and isinstance(ego_history_rot, torch.Tensor):" in src

    def _require_hist(sample):
        xyz = sample.get("ego_history_xyz", None)
        rot = sample.get("ego_history_rot", None)
        if not (isinstance(xyz, torch.Tensor) and isinstance(rot, torch.Tensor)):
            raise ValueError("Expected ego_history_xyz/ego_history_rot tensors for traj fusion.")

    with pytest.raises(ValueError, match="ego_history_xyz/ego_history_rot"):
        _require_hist({"tokenized_data": {"input_ids": torch.zeros(4, dtype=torch.long)}})
    # A silent skip would leave raw pads in the vLLM prompt.
    _require_hist(
        {
            "ego_history_xyz": torch.zeros(16, 3),
            "ego_history_rot": torch.zeros(16, 3, 3),
        }
    )


def test_rollout_text_fallback_and_fused_must_be_rank2(torch):
    def _ids_from_tokenized(td, encode):
        input_ids = td.get("input_ids")
        if input_ids is None:
            raw_text = td.get("text")
            if raw_text is None:
                raise KeyError("Expected key 'input_ids' or 'text'")
            input_ids = torch.tensor(encode(raw_text), dtype=torch.long)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    encoded = _ids_from_tokenized({"text": "hello"}, encode=lambda s: [1, 2, 3])
    assert encoded.tolist() == [1, 2, 3]
    with pytest.raises(KeyError, match="input_ids"):
        _ids_from_tokenized({}, encode=lambda s: [1])

    def _tolist_prompt(fused):
        if not (isinstance(fused, torch.Tensor) and fused.ndim == 2):
            raise ValueError("Expected fused_input_ids to be a 2D tensor [1, T].")
        return fused[0].tolist()

    assert _tolist_prompt(torch.tensor([[10, 11, 12]])) == [10, 11, 12]
    with pytest.raises(ValueError, match="2D tensor"):
        _tolist_prompt(torch.tensor([10, 11, 12]))


def test_policy_text_fallback_still_requires_batch_row(torch):
    src = _BASE_PACKER.read_text()
    assert 'raw_text = tokenized.get("text")' in src
    assert "if input_ids.ndim == 1:" in src
    assert "input_ids = input_ids.unsqueeze(0)" in src
    assert "if input_ids.shape[0] != 1:" in src
    # After the 1D promote, B>1 is still rejected — a [B, L] policy
    # sample would otherwise fuse/CE the wrong row.
    ids = torch.arange(6, dtype=torch.long).reshape(2, 3)
    with pytest.raises(ValueError, match=r"\[1, L\]"):
        if ids.ndim == 2 and ids.shape[0] != 1:
            raise ValueError(f"input_ids must be shape [1, L] or [L], got {tuple(ids.shape)}")
