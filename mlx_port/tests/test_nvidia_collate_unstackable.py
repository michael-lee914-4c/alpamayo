"""``basic_collation_fn`` keeps ``unstackable_keys`` as Python lists.

Weight-free. Complementary to open #25 (stack tensors / list the rest)
and #38 (mixed ``generation_mode`` / ``label_components`` raise). Raw
``image_frames`` are listed as unstackable because crops differ; a
``torch.stack`` there fails the batch or silently pads the wrong H×W.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _basic_collation_fn(batch, unstackable_keys: list[str] | None = None):
    torch = pytest.importorskip("torch")
    unstackable_keys = unstackable_keys or []
    stackable = {k: isinstance(v, torch.Tensor) for k, v in batch[0].items()}
    for k in unstackable_keys:
        if k in stackable:
            stackable[k] = False
    return {
        k: (torch.stack([row[k] for row in batch]) if stackable[k] else [row[k] for row in batch])
        for k in stackable
    }


def test_source_marks_image_frames_unstackable():
    src = _PROC.read_text()
    assert 'basic_collation_fn(data, unstackable_keys=["image_frames"])' in src
    assert "for k in unstackable_keys:" in src
    assert "stackable[k] = False" in src


def test_unstackable_tensor_stays_a_list():
    torch = pytest.importorskip("torch")
    batch = [
        {"ids": torch.tensor([1, 2]), "image_frames": torch.zeros(1, 3, 2, 2)},
        {"ids": torch.tensor([3, 4]), "image_frames": torch.ones(1, 3, 4, 4)},
    ]
    out = _basic_collation_fn(batch, unstackable_keys=["image_frames"])
    assert out["ids"].shape == (2, 2)
    torch.testing.assert_close(out["ids"], torch.tensor([[1, 2], [3, 4]]))
    assert isinstance(out["image_frames"], list)
    assert len(out["image_frames"]) == 2
    assert out["image_frames"][0].shape == (1, 3, 2, 2)
    assert out["image_frames"][1].shape == (1, 3, 4, 4)


def test_missing_unstackable_key_is_ignored():
    torch = pytest.importorskip("torch")
    batch = [{"ids": torch.tensor([7])}, {"ids": torch.tensor([8])}]
    out = _basic_collation_fn(batch, unstackable_keys=["image_frames"])
    assert out["ids"].tolist() == [[7], [8]]
    assert "image_frames" not in out
