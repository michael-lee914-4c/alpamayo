"""Already-normalized float images must not be divided by 255.

Weight-free. Complementary to open #30 (uint8 ``/255`` +
``do_rescale=False``). The float branch is ``images.float()`` only; a
shared ``/255`` would crush [0, 1] frames to ~1/255 and look like a
dark-scene or processor-rescale regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_PROC = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _images_for_vlm(images):
    torch = pytest.importorskip("torch")
    if images.dtype == torch.uint8:
        return images.float() / 255.0
    return images.float()


def test_source_uint8_divides_float_does_not():
    src = _PROC.read_text()
    assert "if images.dtype == torch.uint8:" in src
    assert "images_for_vlm = images.float() / 255.0" in src
    assert "images_for_vlm = images.float()" in src
    assert "do_rescale=False" in src
    uint8_block = src.split("if images.dtype == torch.uint8:", 1)[1]
    else_block = uint8_block.split("else:", 1)[1].split("image_inputs", 1)[0]
    assert "/ 255" not in else_block


def test_float_unit_interval_is_unchanged():
    torch = pytest.importorskip("torch")
    frames = torch.tensor([[[[0.0], [0.5], [1.0]]]], dtype=torch.float32)
    got = _images_for_vlm(frames)
    torch.testing.assert_close(got, frames)
    assert got.dtype == torch.float32


def test_uint8_is_scaled_to_unit_interval():
    torch = pytest.importorskip("torch")
    frames = torch.tensor([[[[0], [128], [255]]]], dtype=torch.uint8)
    got = _images_for_vlm(frames)
    torch.testing.assert_close(got, torch.tensor([[[[0.0], [128 / 255], [1.0]]]]))
    # A float tensor with the same numeric values must not be scaled again.
    already = frames.float()
    stayed = _images_for_vlm(already)
    torch.testing.assert_close(stayed, already)
    assert not torch.allclose(stayed, got)
