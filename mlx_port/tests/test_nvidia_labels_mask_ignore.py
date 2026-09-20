"""SFT Stage-1/2 write ``IGNORE_INDEX`` wherever ``labels_mask`` is False.

Weight-free. Complementary to open #36 (``nan_to_num`` after traj IGNORE)
and #34 (SFT unweighted CE). Both ``TrainableReasoningVLA.forward`` and
``TrainableAlpamayoR1.forward`` clone ``input_ids`` then
``torch.where(labels_mask, labels, IGNORE_INDEX)``. A flipped where()
would CE-train on prompt tokens or drop the future span.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_SFT_S2 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
_IGNORE = -100


def test_source_uses_where_mask_then_ignore_on_both_sft_forwards():
    for path in (_SFT_BASE, _SFT_S2):
        src = path.read_text()
        assert "labels = input_ids.clone()" in src
        assert "labels = torch.where(labels_mask, labels, IGNORE_INDEX)" in src
        assert "if labels_mask is not None:" in src
    assert "IGNORE_INDEX = -100" in (
        _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"
    ).read_text()


def test_false_mask_positions_become_ignore(torch_mod):
    torch = torch_mod
    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    labels_mask = torch.tensor([[False, True, True, False]])
    labels = input_ids.clone()
    labels = torch.where(labels_mask, labels, _IGNORE)
    assert labels.tolist() == [[_IGNORE, 20, 30, _IGNORE]]


def test_none_mask_keeps_every_token(torch_mod):
    torch = torch_mod
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    labels_mask = None
    labels = input_ids.clone()
    if labels_mask is not None:
        labels = torch.where(labels_mask, labels, _IGNORE)
    assert labels.tolist() == [[1, 2, 3]]


def test_all_false_mask_is_all_ignore(torch_mod):
    torch = torch_mod
    input_ids = torch.tensor([[7, 8]], dtype=torch.long)
    labels = torch.where(torch.zeros_like(input_ids, dtype=torch.bool), input_ids, _IGNORE)
    assert labels.tolist() == [[_IGNORE, _IGNORE]]


@pytest.fixture(scope="module")
def torch_mod():
    return pytest.importorskip("torch")
