"""Single-image CoC ablation vs human GT on the local PAI-CoC subset.

Uses one front-wide frame at the CoC event timestamp (not the 16-image
4×4 stack). Stage 1 gate: generated CoC must be readable and align with
the human label before we re-open the multi-camera / expert path.
"""

from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest
from PIL import Image

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from mlx_port.processor import create_message, get_processor, alpamayo_apply_chat_template
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.inference import sample_trajectories_from_data_with_vlm_rollout
from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    format_gt_report,
    load_clip_gt,
    score_coc,
)


LOCAL_DIR = "/Volumes/MicronSSD/pai_coc"
ALPAMAYO_PATH = "/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B"
FRONT_WIDE_INDEX = 1  # after NVIDIA camera_indices sort
REPORT_IMAGE = Path("reports/stage1_single_image_front_wide.png")


def _front_wide_t0_frame(data) -> np.ndarray:
    """Return (1, C, H, W) uint8 tensor: current front-wide frame only."""
    frames = data["image_frames"]
    indices = data["camera_indices"]
    cam_pos = int((indices == FRONT_WIDE_INDEX).nonzero(as_tuple=True)[0][0])
    # Last temporal slot is t0 (NVIDIA: [t0-0.3, t0-0.2, t0-0.1, t0])
    frame = frames[cam_pos, -1]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"expected (C,H,W), got {frame.shape}")
    return frame[None, ...]


@pytest.mark.slow
@pytest.mark.parametrize("temperature", [0.0])
def test_single_image_coc_vs_gt(temperature):
    gt = load_clip_gt(DEFAULT_EVAL_CLIP_ID)
    t0_us = int(gt["events"][0]["event_start_timestamp"])
    print(f"\n[Single-image CoC] clip={DEFAULT_EVAL_CLIP_ID}")
    print(f"[Single-image CoC] t0_us={t0_us} (CoC event, not default 5.1s)")
    print("[Single-image CoC] GT:")
    for text in gt["gt_coc_texts"]:
        print(f"  - {text}")

    data = load_physical_aiavdataset(
        DEFAULT_EVAL_CLIP_ID,
        t0_us=t0_us,
        local_dir=LOCAL_DIR,
        maybe_stream=True,
        num_frames=4,
    )
    single = _front_wide_t0_frame(data)
    print(f"[Single-image CoC] front-wide t0 frame shape={single.shape} dtype={single.dtype}")

    # Save the exact image used for the HTML report
    vis = np.transpose(single[0], (1, 2, 0))
    if vis.dtype != np.uint8:
        vis = vis.astype(np.uint8) if vis.max() > 1.0 else (vis * 255).astype(np.uint8)
    REPORT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(REPORT_IMAGE)
    print(f"[Single-image CoC] wrote {REPORT_IMAGE}")

    messages = create_message(single)
    model = AlpamayoR1MLX.from_pretrained(
        ALPAMAYO_PATH,
        load_expert=False,
        dtype=mx.bfloat16,
    )
    processor = get_processor(model.tokenizer)
    inputs = alpamayo_apply_chat_template(
        processor,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="np",
    )
    # No 4×4 temporal grouping — one image stays one [1, H, W] row.
    if "image_grid_thw" in inputs:
        print("[Single-image CoC] image_grid_thw:\n", inputs["image_grid_thw"])
    if "input_ids" in inputs:
        print("[Single-image CoC] input_ids shape:", inputs["input_ids"].shape)

    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            arr = inputs[key]
            if hasattr(arr, "shape") and len(arr.shape) == 5 and arr.shape[-1] == 3:
                inputs[key] = np.transpose(arr, (0, 4, 1, 2, 3))

    pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data={
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        top_p=1.0,
        temperature=temperature,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
        vlm_only=True,
    )

    pred_coc = extra["cot"][0] if extra and "cot" in extra else None
    print("\n[Single-image CoC] Pred:\n", pred_coc)
    print("\n[GT comparison]")
    print(
        format_gt_report(
            gt,
            pred_coc=pred_coc,
            pred_xyz=pred_xyz,
            ego_future_xyz=data.get("ego_future_xyz"),
        )
    )

    score = score_coc(pred_coc, gt["gt_coc_texts"])
    print(
        f"[Single-image CoC] readable={score['readable']} "
        f"jaccard={score['jaccard']:.3f} gt_coverage={score['gt_coverage']:.3f}"
    )

    assert pred_coc is not None and len(pred_coc) > 0
    # Soft gate: record scores. Hard fail only if generation is empty.
    # Stage 1 is not done until readable=True and gt_coverage is meaningful.
