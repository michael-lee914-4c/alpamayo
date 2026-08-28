"""End-to-end inference test for the MLX port of AlpamayoR1.

This mirrors src/alpamayo_r1/test_inference.py using the MLX-native components.
It loads a CoC-labeled clip from the local PAI-CoC subset, runs the full rollout, and prints the CoC.
NVIDIA's published example clip is not in the CoC subset, so it is not used here.

History lengths used (matching NVIDIA defaults):
    - Camera frames: DEFAULT_NUM_FRAMES = 4 per camera (visual history)
    - Egomotion:     DEFAULT_NUM_HISTORY_STEPS = 16 steps (1.6 s @ 10 Hz)
    - Trajectory tokens: DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens
"""

import numpy as np
import mlx.core as mx
import pytest

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from mlx_port.processor import (
    create_message,
    get_processor,
    alpamayo_apply_chat_template,
    DEFAULT_NUM_FRAMES,
)
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.inference import sample_trajectories_from_data_with_vlm_rollout
from mlx_port.profiling import profile_section, is_profiling_enabled
from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    clean_pred_coc,
    format_gt_report,
    load_clip_gt,
    score_coc,
)


# CoC-labeled clip in local chunks 0–249 (cameras + egomotion downloaded).
# NVIDIA test_inference.py uses 030c760c-… / 5.1s, which has no CoC label.
CLIP_ID = DEFAULT_EVAL_CLIP_ID
LOCAL_DIR = "/Volumes/MicronSSD/pai_coc"
# Last confirmed greedy (temperature=0, 16×[1,H,W], event t0) CoC on CLIP_ID.
EXPECTED_GREEDY_COC_PREFIX = "Stop for the stop sign"


@pytest.mark.slow
@pytest.mark.parametrize("max_gen_len", [256])
def test_end_to_end_inference_prints_coc_vlm_only(max_gen_len):
    """Run inference on one clip using only the VLM (no diffusion expert loaded).

    This is a lightweight end-to-end test that exercises the full MLX pipeline up to
    VLM generation (CoC) and a stub expert path. The real diffusion expert architecture
    has not been implemented yet, so expert weights are not loaded (`load_expert=False`).

    Once the real expert is implemented, a separate test (or this one updated) will load
    the full pre-trained expert weights.
    """
    print(f"\n[End-to-End Test] Loading dataset for clip_id: {CLIP_ID}...")
    gt = load_clip_gt(CLIP_ID)
    t0_us = int(gt["events"][0]["event_start_timestamp"])
    print(f"[End-to-End Test] t0_us={t0_us} (CoC event, not default 5.1s)")
    print("[End-to-End Test] GT CoC labels:")
    for text in gt["gt_coc_texts"]:
        print(f"  - {text}")
    data = load_physical_aiavdataset(
        CLIP_ID,
        t0_us=t0_us,
        local_dir=LOCAL_DIR,
        maybe_stream=True,
        num_frames=DEFAULT_NUM_FRAMES,
    )
    print("[End-to-End Test] Dataset loaded.")
    frames = data["image_frames"]
    print(
        f"[End-to-End Test] image_frames shape={tuple(frames.shape)} "
        f"(expect 4 cameras × {DEFAULT_NUM_FRAMES} frames)"
    )

    # Create chat messages (MLX version of helper.create_message)
    messages = create_message(frames.flatten(0, 1))
    n_images = sum(1 for m in messages for c in m.get("content", []) if c.get("type") == "image")
    print(f"[End-to-End Test] chat images={n_images} (expect 16)")

    # Load MLX model (weights from local Alpamayo-R1-10B checkpoint)
    print("[End-to-End Test] Loading AlpamayoR1MLX (this may take a while)...")
    with profile_section("model_load", enabled=is_profiling_enabled()):
        model = AlpamayoR1MLX.from_pretrained(
            "/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B",
            load_expert=False,  # Expert weights not loaded (real expert architecture not yet implemented)
            dtype=mx.bfloat16,  # Enforce bfloat16 for all inference (Row 10)
        )

    # Processor (injects Alpamayo tokenizer)
    with profile_section("get_processor", enabled=is_profiling_enabled()):
        processor = get_processor(model.tokenizer)
    # History fusion uses DeltaTrajectoryTokenizerMLX (16 × xyz = 48 tokens).

    # Use the Alpamayo-controlled apply_chat_template helper.
    # This guarantees a flat images list and therefore a correct image_grid_thw,
    # exactly analogous to how AlpamayoPatchEmbed guarantees correct Conv3D input.
    with profile_section("apply_chat_template", enabled=is_profiling_enabled()):
        inputs = alpamayo_apply_chat_template(
            processor,
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="np",
        )

    # NVIDIA processor emits 16×[1,H,W]. Do not collapse to 4×[4,H,W].

    # Normalize pixel_values to channels-first for the Alpamayo-loaded Conv3D.
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            arr = inputs[key]
            if hasattr(arr, "shape") and len(arr.shape) == 5 and arr.shape[-1] == 3:
                inputs[key] = np.transpose(arr, (0, 4, 1, 2, 3))

    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }

    # Run rollout with return_extra=True so we get the CoC
    print("[End-to-End Test] Running greedy VLM rollout (temperature=0, no grouping)...")
    pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data=model_inputs,
        top_p=1.0,
        temperature=0.0,
        num_traj_samples=1,
        max_generation_length=max_gen_len,
        return_extra=True,
        vlm_only=True,  # early exit after CoC generation; no action expert / diffusion
    )

    # Report the true peak memory observed during the entire VLM generation.
    # This captures spikes that may occur inside long forward passes even if
    # memory later declines before the next profiler sample.
    from mlx_port.profiling import get_global_memory_peak

    peak = get_global_memory_peak()
    if peak["total"] > 0:
        print(
            f"\n[MEMORY PEAK] Global high-water mark during test: "
            f"total={peak['total']/1e9:.2f}GB  "
            f"resident={peak['resident']/1e9:.2f}GB  "
            f"compressed={peak['compressed']/1e9:.2f}GB"
        )
        # Prominently expose the Metal peak (often the dominant consumer on Apple Silicon)
        print(
            f"[METAL PEAK ] Highest Metal active memory observed: "
            f"{peak['metal']/1e9:.2f}GB  "
            f"(final Metal memory was lower; this is the true high-water mark during VLM execution)"
        )

    pred_coc = None
    if extra and "cot" in extra:
        pred_coc = extra["cot"][0]
        pred_clean = clean_pred_coc(pred_coc)
        print("\nChain-of-Causation (per trajectory):\n", pred_coc)
        score = score_coc(pred_coc, gt["gt_coc_texts"])
        print(
            f"[End-to-End Test] readable={score['readable']} "
            f"jaccard={score['jaccard']:.3f} gt_coverage={score['gt_coverage']:.3f} "
            f"cleaned={pred_clean!r}"
        )
        assert score["readable"], f"greedy CoC is not readable: {pred_coc!r}"
        assert pred_clean.startswith(EXPECTED_GREEDY_COC_PREFIX), (
            f"greedy CoC drifted from pinned output.\n"
            f"  expected prefix: {EXPECTED_GREEDY_COC_PREFIX!r}\n"
            f"  got: {pred_clean!r}"
        )
    else:
        print("\n[End-to-End Test] No CoC extracted (extra=", extra, ")")

    print("\n[GT comparison]")
    print(
        format_gt_report(
            gt,
            pred_coc=pred_coc,
            pred_xyz=pred_xyz,
            ego_future_xyz=data.get("ego_future_xyz"),
        )
    )

    # Print future trajectory (pred_xyz / pred_rot) produced by the action expert
    print("\nFuture Trajectory (from action expert):")
    if pred_xyz is not None:
        xyz_np = np.asarray(pred_xyz)
        print("  pred_xyz shape:", xyz_np.shape)
        print("  pred_xyz[0, :5]:\n", xyz_np[0, :5])
    if pred_rot is not None:
        rot_np = np.asarray(pred_rot)
        print("  pred_rot shape:", rot_np.shape)
        print("  pred_rot[0, :5]:\n", rot_np[0, :5])

    # Basic shape sanity checks
    # In vlm_only mode, pred_xyz/pred_rot are intentionally None
    if pred_xyz is not None or pred_rot is not None:
        assert pred_xyz is not None
        assert pred_rot is not None
    print("[End-to-End Test] Inference completed successfully.")

    # Explicit cleanup to release Metal memory held by MLX objects
    print("[End-to-End Test] Cleaning up to release memory...")
    import gc
    for name in ("model", "processor", "data", "model_inputs", "generated", "cache", "vlm_outputs", "extra"):
        if name in locals():
            del locals()[name]
    gc.collect()
    mx.clear_cache()
    print("[End-to-End Test] Memory cleanup done.")