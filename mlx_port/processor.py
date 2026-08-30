# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message construction and processor utilities for the MLX Alpamayo port.

This module mirrors the behavior of src/alpamayo_r1/helper.py but is
framework-agnostic so it works with MLX, NumPy, or PyTorch tensors.

History configuration (used by create_message and the inference pipeline):
    DEFAULT_NUM_FRAMES = 4 camera frames per view
    DEFAULT_NUM_HISTORY_STEPS = 16 egomotion steps (1.6 s)
    DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens produced by the history tokenizer
"""

import math
from typing import Any, List, Tuple, Union

import numpy as np
from PIL import Image
from transformers import AutoProcessor

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import torch
except ImportError:
    torch = None


MIN_PIXELS = 163840
MAX_PIXELS = 196608
# Qwen3-VL preprocessor_config.json: patch 16, merge 2 → factor 32.
VISION_PATCH_SIZE = 16
VISION_MERGE_SIZE = 2
# Use the locally downloaded Qwen3-VL-8B-Instruct checkpoint (not the Hub)
LOCAL_QWEN_PROCESSOR_PATH = "/Users/michaellee/Projects/alpamayo/pre-trained/Qwen3-VL-8B-Instruct"

# =============================================================================
# History configuration constants (exposed for clarity & configurability)
# =============================================================================

# Number of discrete trajectory tokens used to represent egomotion history.
# With the default num_history_steps=16 (1.6 s @ 10 Hz), the tokenizer emits 48 tokens.
DEFAULT_HISTORY_TRAJ_TOKENS = 48

# Default number of camera frames per view (visual history ending at t0).
DEFAULT_NUM_FRAMES = 4

# Default number of egomotion history steps (1.6 s at 0.1 s step).
DEFAULT_NUM_HISTORY_STEPS = 16


def smart_resize_hw(
    height: int,
    width: int,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    patch_size: int = VISION_PATCH_SIZE,
    merge_size: int = VISION_MERGE_SIZE,
) -> Tuple[int, int]:
    """HF Qwen2/3-VL ``smart_resize`` (factor = patch × merge)."""
    if height <= 0 or width <= 0:
        raise ValueError(f"smart_resize_hw requires positive HxW, got {height}x{width}")
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError(
            f"invalid pixel budget min={min_pixels} max={max_pixels}"
        )
    factor = int(patch_size) * int(merge_size)
    if factor <= 0:
        raise ValueError(f"invalid patch/merge factor {factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def expected_image_grid_hw(
    height: int,
    width: int,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    patch_size: int = VISION_PATCH_SIZE,
    merge_size: int = VISION_MERGE_SIZE,
) -> Tuple[int, int]:
    """Patch grid (H, W) after NVIDIA pixel-budget resize."""
    rh, rw = smart_resize_hw(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        patch_size=patch_size,
        merge_size=merge_size,
    )
    return rh // patch_size, rw // patch_size


def _size_edges(size: Any) -> Tuple[Any, Any]:
    if size is None:
        return None, None
    if isinstance(size, dict):
        return size.get("shortest_edge"), size.get("longest_edge")
    return getattr(size, "shortest_edge", None), getattr(size, "longest_edge", None)


def image_pixel_budget(image_processor: Any) -> Tuple[int, int]:
    """Read ``(min_pixels, max_pixels)`` from an image/video processor."""
    if image_processor is None:
        raise ValueError("image_pixel_budget requires an image processor")
    mn = getattr(image_processor, "min_pixels", None)
    mx = getattr(image_processor, "max_pixels", None)
    short, long = _size_edges(getattr(image_processor, "size", None))
    if mn is None:
        mn = short
    if mx is None:
        mx = long
    if mn is None or mx is None:
        raise ValueError(
            f"cannot read pixel budget from {type(image_processor).__name__}"
        )
    return int(mn), int(mx)


def _write_size(image_processor: Any, min_pixels: int, max_pixels: int) -> None:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        size["shortest_edge"] = min_pixels
        size["longest_edge"] = max_pixels
        return
    if size is not None and hasattr(size, "shortest_edge"):
        try:
            size.shortest_edge = min_pixels
            size.longest_edge = max_pixels
            return
        except (AttributeError, TypeError):
            pass
    image_processor.size = {
        "shortest_edge": min_pixels,
        "longest_edge": max_pixels,
    }


def bind_image_pixel_budget(
    processor: Any,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Any:
    """Force the NVIDIA pixel budget onto the image (and video) processor.

    Qwen3-VL ``preprocessor_config.json`` ships
    ``size.longest_edge=16777216``. ``AutoProcessor.from_pretrained(...,
    min_pixels=..., max_pixels=...)`` does not override that. mlx_vlm's
    torch-free loader copies those edges into ``max_pixels``, so 1080×1920
    stays native (grid 68×120, ~2040 tokens/frame).

    Raises if nothing can be bound or the read-back does not match.
    """
    if processor is None:
        raise ValueError("bind_image_pixel_budget requires a processor")
    if min_pixels <= 0 or max_pixels <= 0 or min_pixels > max_pixels:
        raise ValueError(
            f"invalid pixel budget min={min_pixels} max={max_pixels}"
        )

    targets: List[Tuple[str, Any]] = []
    for name in ("image_processor", "video_processor"):
        sub = getattr(processor, name, None)
        if sub is not None:
            targets.append((name, sub))
    if not targets:
        if hasattr(processor, "min_pixels") or hasattr(processor, "size"):
            targets.append(("processor", processor))
        else:
            raise ValueError("bind_image_pixel_budget: no image_processor to bind")

    for name, sub in targets:
        _write_size(sub, min_pixels, max_pixels)
        sub.min_pixels = min_pixels
        sub.max_pixels = max_pixels
        got_min, got_max = image_pixel_budget(sub)
        if got_min != min_pixels or got_max != max_pixels:
            raise RuntimeError(
                f"{name} pixel budget did not bind: "
                f"wanted min={min_pixels} max={max_pixels}, "
                f"got min={got_min} max={got_max}"
            )

    names = ", ".join(name for name, _ in targets)
    print(f"[PIXELS] bound min={min_pixels} max={max_pixels} on {names}")
    return processor


def _to_numpy(frames: Any) -> np.ndarray:
    """Convert supported array types to NumPy for shape inspection."""
    if isinstance(frames, np.ndarray):
        return frames
    if mx is not None and isinstance(frames, mx.array):
        return np.array(frames)
    if torch is not None and isinstance(frames, torch.Tensor):
        return frames.detach().cpu().numpy()
    if isinstance(frames, (list, tuple)):
        return np.array(frames)
    raise TypeError(f"Unsupported frame type: {type(frames)}")


def create_message(
    frames: Any,
    num_history_traj_tokens: int = DEFAULT_HISTORY_TRAJ_TOKENS,
) -> List[dict]:
    """Construct the chat message list expected by the VLM.

    This function is an exact port of alpamayo_r1.helper.create_message.
    It builds the system + user + assistant turn structure with the
    trajectory-history placeholder tokens.

    History lengths (see module constants):
        - DEFAULT_NUM_FRAMES = 4 camera frames per view
        - DEFAULT_NUM_HISTORY_STEPS = 16 egomotion steps (1.6 s)
        - DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens (from hist_traj_tokenizer)

    Args:
        frames: Image tensor of shape (N, C, H, W). Accepts NumPy, MLX,
                PyTorch, or Python list.
        num_history_traj_tokens: Number of <|traj_history|> tokens to insert.
            Defaults to 48 (matches the current tokenizer for 16 history steps).

    Returns:
        List of chat messages in the format expected by mlx_vlm / Qwen3-VL.
    """
    arr = _to_numpy(frames)
    if arr.ndim != 4:
        raise ValueError(f"{arr.ndim=}, expected 4 (N, C, H, W)")

    # Convert each frame (C, H, W) to a PIL Image (H, W, C) for robust processing
    # by the Qwen3-VL image processor. This prevents shape misalignment issues
    # where the processor expects 3D images but receives 4D tensors.
    pil_images = []
    for frame in arr:
        # Ensure channel-first to channel-last and uint8 range
        if frame.shape[0] in (1, 3):  # (C, H, W) format
            frame = np.transpose(frame, (1, 2, 0))  # -> (H, W, C)
        if frame.dtype != np.uint8:
            # Assume float in [0, 1] or [0, 255]; scale appropriately
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        pil_images.append(Image.fromarray(frame, mode="RGB" if frame.shape[2] == 3 else "L"))

    # NOTE: we expand the padding tokens to match training, so we can
    # directly apply the native processor from the VLM.
    num_traj_token = DEFAULT_HISTORY_TRAJ_TOKENS
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer: Any, model_path: str = LOCAL_QWEN_PROCESSOR_PATH) -> Any:
    """Get the processor for the locally downloaded Qwen3-VL model.

    This is the MLX-port equivalent of alpamayo_r1.helper.get_processor.
    It loads the processor from the local checkpoint directory, injects the
    Alpamayo tokenizer, then binds the NVIDIA pixel budget. JSON
    ``size.longest_edge=16777216`` is not left in place.

    Args:
        tokenizer: The Alpamayo tokenizer (with traj tokens already added).
        model_path: Path to the local Qwen3-VL checkpoint. Defaults to the
            project-local copy at pre-trained/Qwen3-VL-8B-Instruct.

    Returns:
        A processor object with the Alpamayo tokenizer attached.
    """
    # mlx_vlm registers a torch-free Qwen3-VL processor. Without that patch,
    # HF AutoProcessor loads Qwen3VLVideoProcessor (requires torchvision).
    import mlx_vlm.models.qwen3_vl.processing_qwen3_vl  # noqa: F401

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer = tokenizer
    # JSON size.longest_edge is 16M; kwargs to from_pretrained do not override.
    bind_image_pixel_budget(processor)
    return processor


def alpamayo_apply_chat_template(
    processor: Any,
    messages: List[dict],
    tokenize: bool = True,
    add_generation_prompt: bool = False,
    continue_final_message: bool = True,
    return_dict: bool = True,
    return_tensors: str = "np",
    padding: bool = True,
) -> Any:
    """Custom apply_chat_template that guarantees a flat images list.

    This is the Alpamayo equivalent of AlpamayoPatchEmbed: a controlled
    wrapper that works around a latent bug in mlx_vlm's Qwen3-VL processor.

    The mlx_vlm implementation of apply_chat_template(..., tokenize=True)
    can internally construct a nested list-of-lists for the images when
    processing multi-image messages. That nested structure reaches
    Qwen3VLImageProcessor.__call__ as [[pil0, ..., pil15]], which is then
    converted to a (16, H, W, C) array and passed to _process_one, causing
    "too many values to unpack (expected 3)".

    This function bypasses that path entirely:
    1. Calls apply_chat_template(..., tokenize=False) to obtain the prompt text.
    2. Extracts the 16 PIL images directly from the message content as a flat list.
    3. Calls the processor with an explicit flat `images=` argument.
    4. Returns a dict with the same keys as the tokenize=True path
       (input_ids, attention_mask, pixel_values, image_grid_thw, ...).

    The result is identical in structure to what a correct processor would
    return, but with the correct (16, 3) image_grid_thw that respects the
    4-camera × 4-frame temporal grouping expected by the Alpamayo fine-tune.

    Args:
        processor: The Alpamayo-injected Qwen3-VL processor.
        messages: Chat messages as produced by create_message.
        tokenize: Must be True (the only supported mode for this helper).
        add_generation_prompt, continue_final_message: Passed through.
        return_dict, return_tensors, padding: Control the output format.

    Returns:
        A dict-like object (or BatchFeature) containing tokenized inputs
        and vision tensors, exactly as the standard tokenize=True path would.
    """
    if not tokenize or not return_dict:
        # For non-tokenize or non-dict paths we simply delegate.
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_dict=return_dict,
            return_tensors=return_tensors,
        )

    # --- The safe, controlled path ---
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )

    # Extract images as a flat list directly from the message structure.
    images = [
        item["image"]
        for item in messages[1]["content"]
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    # Call the processor with an explicit flat images list.
    # This guarantees that Qwen3VLImageProcessor receives a flat list of 16
    # individual images and therefore produces the correct image_grid_thw.
    inputs = processor(
        text=text,
        images=images,
        return_tensors=return_tensors,
        padding=padding,
    )

    return inputs


def enforce_alpamayo_temporal_grouping(
    inputs: dict,
    num_cameras: int = 4,
    num_frames_per_camera: int = 4,
) -> dict:
    """Post-process processor output to enforce Alpamayo's 4×4 temporal grouping.

    The Qwen3-VL processor (even with a flat images list) produces an
    image_grid_thw with 16 independent rows of [1, H, W] because it treats
    each of the 16 images as a single-frame temporal group.

    Alpamayo was fine-tuned with 4 cameras × 4 frames per camera, expecting
    the vision tower's Conv3D (temporal_patch_size=2) to see temporally
    coherent stacks. This function reorganizes image_grid_thw so that each
    camera's 4 frames form a single temporal group with T=4.

    Language-model RoPE still uses one HF image grid per frame-level
    image-pad run (16×[1,H,W]); grouped T>1 rows are split back in
    get_rope_index.

    For now we use T=4 per camera (4 groups total). Patch count is unchanged
    (16×H×W of the per-frame grid). Do not re-enable this on greedy e2e.

    Args:
        inputs: Dict returned by alpamayo_apply_chat_template (or any
            processor output) containing "image_grid_thw".
        num_cameras: Number of cameras (default 4).
        num_frames_per_camera: Frames per camera (default 4).

    Returns:
        The same dict with "image_grid_thw" replaced by the temporally
        grouped version. Other keys are left unchanged.
    """
    if "image_grid_thw" not in inputs:
        return inputs

    grid = np.asarray(inputs["image_grid_thw"])  # (N, 3)
    n_groups = grid.shape[0]
    expected = num_cameras * num_frames_per_camera
    if n_groups != expected:
        # Not the 4×4 case we know how to handle; leave unchanged.
        return inputs

    # Each original row is [1, H, W]. We want to collapse every
    # `num_frames_per_camera` consecutive rows into one row [T, H, W]
    # where T = num_frames_per_camera.
    h, w = grid[0, 1], grid[0, 2]
    t = num_frames_per_camera

    new_grid = np.zeros((num_cameras, 3), dtype=np.int64)
    for cam in range(num_cameras):
        new_grid[cam] = [t, h, w]

    inputs = dict(inputs)  # shallow copy so we don't mutate caller's dict
    inputs["image_grid_thw"] = new_grid
    return inputs


def dump_vision_inputs(label: str, inputs: dict) -> None:
    """Print grid / pixel stats for a processor output dict."""
    ids = np.asarray(inputs.get("input_ids"))
    print(f"[{label}] input_ids shape={getattr(ids, 'shape', None)}")
    if "image_grid_thw" in inputs:
        grid = np.asarray(inputs["image_grid_thw"])
        print(f"[{label}] image_grid_thw shape={grid.shape}\n{grid}")
    else:
        print(f"[{label}] image_grid_thw missing")
    for key in ("pixel_values", "pixel_values_videos"):
        if key not in inputs:
            continue
        arr = np.asarray(inputs[key])
        print(
            f"[{label}] {key} shape={arr.shape} dtype={arr.dtype} "
            f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
            f"mean={float(arr.mean()):.4f}"
        )


def _frame_to_hwc_uint8(frame: Any) -> np.ndarray:
    """CHW/HWC tensor or array → HWC uint8, matching create_message."""
    if torch is not None and isinstance(frame, torch.Tensor):
        arr = frame.detach().cpu().numpy()
    else:
        arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def compare_nvidia_tokenization(mlx_inputs: dict, frames, tokenizer) -> None:
    """Diff MLX tokenize vs NVIDIA helper.create_message + the same processor.

    Native ``apply_chat_template(..., tokenize=True)`` hits the nested-image
    unpack bug on this stack, so images are flattened the same way as
    ``alpamayo_apply_chat_template``. This still compares chat text and
    pixels from NVIDIA's raw tensors vs MLX's PIL path.
    """
    from alpamayo_r1 import helper

    print("[PARITY] NVIDIA-style tokenize (helper.create_message + flat processor)")
    nv_messages = helper.create_message(frames)
    nv_proc = get_processor(tokenizer, LOCAL_QWEN_PROCESSOR_PATH)
    print(f"[PARITY] processor={LOCAL_QWEN_PROCESSOR_PATH}")

    nv_text = nv_proc.apply_chat_template(
        nv_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    print(f"[PARITY] nvidia_text_len={len(nv_text)}")
    print(f"[PARITY] nvidia_text_tail={nv_text[-180:]!r}")

    nv_images = []
    for item in nv_messages[1]["content"]:
        if isinstance(item, dict) and item.get("type") == "image":
            arr = _frame_to_hwc_uint8(item["image"])
            nv_images.append(Image.fromarray(arr, mode="RGB" if arr.shape[2] == 3 else "L"))
    print(f"[PARITY] nvidia_images={len(nv_images)} first_hw={nv_images[0].size if nv_images else None}")

    nv_inputs = nv_proc(
        text=nv_text,
        images=nv_images,
        return_tensors="np",
        padding=True,
    )
    dump_vision_inputs("NVIDIA", nv_inputs)

    mlx_ids = np.asarray(mlx_inputs["input_ids"])
    nv_ids = np.asarray(nv_inputs["input_ids"])
    print(f"[PARITY] input_ids mlx={mlx_ids.shape} nvidia={nv_ids.shape}")
    if mlx_ids.shape == nv_ids.shape:
        n_diff = int((mlx_ids != nv_ids).sum())
        print(f"[PARITY] input_ids mismatches={n_diff}/{mlx_ids.size}")
        if n_diff:
            bad = np.argwhere(mlx_ids != nv_ids)
            for r, c in bad[:12]:
                print(
                    f"[PARITY]   pos[{int(r)},{int(c)}] mlx={int(mlx_ids[r, c])} "
                    f"nvidia={int(nv_ids[r, c])}"
                )
    else:
        print("[PARITY] input_ids shapes differ; no elementwise compare")

    mlx_grid = np.asarray(mlx_inputs.get("image_grid_thw"))
    nv_grid = np.asarray(nv_inputs.get("image_grid_thw"))
    if mlx_grid.shape == nv_grid.shape:
        print(f"[PARITY] image_grid_thw equal={np.array_equal(mlx_grid, nv_grid)}")
    else:
        print(f"[PARITY] image_grid_thw mlx={mlx_grid.shape} nvidia={nv_grid.shape}")

    for key in ("pixel_values", "pixel_values_videos"):
        if key not in mlx_inputs or key not in nv_inputs:
            continue
        a = np.asarray(mlx_inputs[key], dtype=np.float32)
        b = np.asarray(nv_inputs[key], dtype=np.float32)
        if a.shape != b.shape:
            print(f"[PARITY] {key} shapes mlx={a.shape} nvidia={b.shape}")
            continue
        diff = np.abs(a - b)
        print(
            f"[PARITY] {key} max_abs_diff={float(diff.max()):.6f} "
            f"mean_abs_diff={float(diff.mean()):.6f}"
        )

    # Raw dataset range so we know whether PIL uint8 conversion quantized anything.
    raw = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
    print(
        f"[PARITY] raw_frames shape={raw.shape} dtype={raw.dtype} "
        f"min={float(raw.min()):.4f} max={float(raw.max()):.4f}"
    )
