"""P2f: NVIDIA pixel budget binds; 1080p → 20×36 grid, not native 68×120."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mlx_port.processor import (
    LOCAL_QWEN_PROCESSOR_PATH,
    MAX_PIXELS,
    MIN_PIXELS,
    alpamayo_apply_chat_template,
    bind_image_pixel_budget,
    create_message,
    expected_image_grid_hw,
    get_processor,
    image_pixel_budget,
    smart_resize_hw,
)


def test_smart_resize_1080p_is_320x576():
    assert smart_resize_hw(1080, 1920) == (320, 576)
    assert expected_image_grid_hw(1080, 1920) == (20, 36)


def test_smart_resize_rejects_bad_budget():
    with pytest.raises(ValueError, match="invalid pixel budget"):
        smart_resize_hw(1080, 1920, min_pixels=200000, max_pixels=1000)


def test_bind_image_pixel_budget_overrides_16m_size():
    class _IP:
        def __init__(self):
            self.size = {"shortest_edge": 65536, "longest_edge": 16777216}
            self.min_pixels = 65536
            self.max_pixels = 16777216

    class _Proc:
        def __init__(self):
            self.image_processor = _IP()

    proc = bind_image_pixel_budget(_Proc())
    assert image_pixel_budget(proc.image_processor) == (MIN_PIXELS, MAX_PIXELS)
    assert proc.image_processor.size["longest_edge"] == MAX_PIXELS


def test_bind_image_pixel_budget_on_image_processor_itself():
    class _IP:
        def __init__(self):
            self.min_pixels = 16777216
            self.max_pixels = 16777216

    ip = bind_image_pixel_budget(_IP())
    assert image_pixel_budget(ip) == (MIN_PIXELS, MAX_PIXELS)


def test_bind_image_pixel_budget_rejects_missing():
    with pytest.raises(ValueError, match="no image_processor"):
        bind_image_pixel_budget(object())
    with pytest.raises(ValueError, match="requires a processor"):
        bind_image_pixel_budget(None)


def test_smart_resize_rejects_nonpositive_factor():
    with pytest.raises(ValueError, match="factor"):
        smart_resize_hw(64, 64, patch_size=0, merge_size=2)
    with pytest.raises(ValueError, match="factor"):
        smart_resize_hw(64, 64, patch_size=16, merge_size=0)


def test_image_pixel_budget_requires_readable_edges():
    with pytest.raises(ValueError, match="requires an image processor"):
        image_pixel_budget(None)
    with pytest.raises(ValueError, match="cannot read pixel budget"):
        image_pixel_budget(object())


def test_create_message_rejects_non_nchw_rank():
    with pytest.raises(ValueError, match="expected 4"):
        create_message(np.zeros((3, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="expected 4"):
        create_message(np.zeros((1, 16, 3, 8, 8), dtype=np.uint8))


def test_mlx_vlm_image_processor_1080p_grid_after_bind():
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLImageProcessor

    ip = Qwen3VLImageProcessor(
        patch_size=16,
        merge_size=2,
        temporal_patch_size=2,
        # Qwen3-VL preprocessor_config.json size edges (not min=max=16M —
        # that would upscale 1080p to meet min_pixels).
        min_pixels=65536,
        max_pixels=16777216,
    )
    img = Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8), mode="RGB")
    native = np.asarray(ip([img])["image_grid_thw"])
    assert native[0].tolist() == [1, 68, 120], native

    bind_image_pixel_budget(ip)
    budgeted = np.asarray(ip([img])["image_grid_thw"])
    assert budgeted[0].tolist() == [1, 20, 36], budgeted


def test_get_processor_downsamples_16_native_frames():
    if not Path(LOCAL_QWEN_PROCESSOR_PATH).exists():
        pytest.skip("local Qwen3-VL processor not present")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(LOCAL_QWEN_PROCESSOR_PATH, trust_remote_code=True)
    proc = get_processor(tok)
    assert image_pixel_budget(proc.image_processor) == (MIN_PIXELS, MAX_PIXELS)

    frames = np.zeros((16, 3, 1080, 1920), dtype=np.uint8)
    inputs = alpamayo_apply_chat_template(
        proc,
        create_message(frames),
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="np",
    )
    grid = np.asarray(inputs["image_grid_thw"])
    assert grid.shape == (16, 3), grid.shape
    assert np.array_equal(grid, np.tile([1, 20, 36], (16, 1))), grid

    ids = np.asarray(inputs["input_ids"])
    n_vision = 16 * (20 * 36 // 4)
    assert ids.size < 8000, f"expected ~3k-token prefix after downsample, got {ids.size}"
    assert ids.size > n_vision, f"prefix {ids.size} shorter than {n_vision} vision tokens"
