"""End-to-end inference test for the MLX port of AlpamayoR1.

This mirrors src/alpamayo_r1/test_inference.py using the MLX-native components.
It loads a CoC-labeled clip from the local PAI-CoC subset, runs the full rollout, and prints the CoC.
NVIDIA's published example clip is not in the CoC subset, so it is not used here.

History lengths used (matching NVIDIA defaults):
    - Camera frames: DEFAULT_NUM_FRAMES = 4 per camera (visual history)
    - Egomotion:     DEFAULT_NUM_HISTORY_STEPS = 16 steps (1.6 s @ 10 Hz)
    - Trajectory tokens: DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens
"""

import gc

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
from mlx_port.profiling import profile_section, is_profiling_enabled, get_global_memory_peak
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
CHECKPOINT = "/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B"
# Last confirmed greedy (temperature=0, 16×[1,H,W], event t0) CoC on CLIP_ID.
EXPECTED_GREEDY_COC_PREFIX = "Stop for the stop sign"
# NVIDIA test_inference.py
NVIDIA_TEMPERATURE = 0.6
NVIDIA_TOP_P = 0.98
NVIDIA_SEED = 42


def _load_clip():
    gt = load_clip_gt(CLIP_ID)
    t0_us = int(gt["events"][0]["event_start_timestamp"])
    print(f"\n[End-to-End Test] clip={CLIP_ID}")
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
    frames = data["image_frames"]
    print(
        f"[End-to-End Test] image_frames shape={tuple(frames.shape)} "
        f"(expect 4 cameras × {DEFAULT_NUM_FRAMES} frames)"
    )
    return gt, data


def _prepare_inputs(model, data):
    frames = data["image_frames"]
    messages = create_message(frames.flatten(0, 1))
    n_images = sum(1 for m in messages for c in m.get("content", []) if c.get("type") == "image")
    print(f"[End-to-End Test] chat images={n_images} (expect 16)")
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
    # NVIDIA processor emits 16×[1,H,W]. Do not collapse to 4×[4,H,W].
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs:
            arr = inputs[key]
            if hasattr(arr, "shape") and len(arr.shape) == 5 and arr.shape[-1] == 3:
                inputs[key] = np.transpose(arr, (0, 4, 1, 2, 3))
    return {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }


def _report_memory():
    peak = get_global_memory_peak()
    if peak["total"] > 0:
        print(
            f"\n[MEMORY PEAK] Global high-water mark during test: "
            f"total={peak['total']/1e9:.2f}GB  "
            f"resident={peak['resident']/1e9:.2f}GB  "
            f"compressed={peak['compressed']/1e9:.2f}GB"
        )
        print(
            f"[METAL PEAK ] Highest Metal active memory observed: "
            f"{peak['metal']/1e9:.2f}GB"
        )
    return peak


def _print_traj(pred_xyz, pred_rot, data):
    print("\nFuture Trajectory (from action expert):")
    if pred_xyz is not None:
        xyz_np = np.asarray(pred_xyz)
        print("  pred_xyz shape:", xyz_np.shape)
        print("  pred_xyz[0, :5]:\n", xyz_np.reshape(-1, xyz_np.shape[-1])[:5])
    else:
        print("  pred_xyz=None  (expert step_fn not run)")
    if pred_rot is not None:
        rot_np = np.asarray(pred_rot)
        print("  pred_rot shape:", rot_np.shape)
    gt_fut = data.get("ego_future_xyz")
    if gt_fut is not None:
        fut = np.asarray(gt_fut)
        print(f"  gt ego_future_xyz shape={fut.shape}")


def _cleanup(*objs):
    print("[End-to-End Test] Cleaning up to release memory...")
    del objs
    gc.collect()
    mx.clear_cache()
    print("[End-to-End Test] Memory cleanup done.")


@pytest.mark.slow
@pytest.mark.parametrize("max_gen_len", [256])
def test_end_to_end_inference_prints_coc_vlm_only(max_gen_len):
    """Greedy CoC (T=0). Pins the last confirmed sentence on CLIP_ID."""
    gt, data = _load_clip()
    print("[End-to-End Test] Loading AlpamayoR1MLX (this may take a while)...")
    with profile_section("model_load", enabled=is_profiling_enabled()):
        model = AlpamayoR1MLX.from_pretrained(
            CHECKPOINT,
            load_expert=False,
            dtype=mx.bfloat16,
        )
    model_inputs = _prepare_inputs(model, data)

    print("[End-to-End Test] Running greedy VLM rollout (temperature=0, no grouping)...")
    pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data=model_inputs,
        top_p=1.0,
        temperature=0.0,
        num_traj_samples=1,
        max_generation_length=max_gen_len,
        return_extra=True,
        vlm_only=True,
    )
    _report_memory()

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
        pytest.fail("greedy rollout returned no CoC")

    print("\n[GT comparison]")
    print(
        format_gt_report(
            gt,
            pred_coc=pred_coc,
            pred_xyz=pred_xyz,
            ego_future_xyz=data.get("ego_future_xyz"),
        )
    )
    _print_traj(pred_xyz, pred_rot, data)
    assert pred_xyz is None and pred_rot is None
    print("[End-to-End Test] Inference completed successfully.")
    _cleanup(model, data, model_inputs, extra)


@pytest.mark.slow
@pytest.mark.parametrize("max_gen_len", [256])
def test_end_to_end_inference_temperature_coc_and_traj(max_gen_len):
    """NVIDIA test_inference.py draw: T=0.6, top_p=0.98, seed 42, one sample.

    CoC is sampled (not greedy). Stage 1b wires expert step_fn so this
    test can print pred_xyz / minADE; until then pred_xyz stays None.
    """
    gt, data = _load_clip()
    print("[End-to-End Test] Loading AlpamayoR1MLX (this may take a while)...")
    with profile_section("model_load", enabled=is_profiling_enabled()):
        model = AlpamayoR1MLX.from_pretrained(
            CHECKPOINT,
            load_expert=False,
            dtype=mx.bfloat16,
        )
    model_inputs = _prepare_inputs(model, data)

    mx.random.seed(NVIDIA_SEED)
    print(
        f"[End-to-End Test] NVIDIA sampling: T={NVIDIA_TEMPERATURE} "
        f"top_p={NVIDIA_TOP_P} seed={NVIDIA_SEED} num_traj_samples=1"
    )
    pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data=model_inputs,
        top_p=NVIDIA_TOP_P,
        temperature=NVIDIA_TEMPERATURE,
        num_traj_samples=1,
        max_generation_length=max_gen_len,
        return_extra=True,
        vlm_only=False,
    )
    peak = _report_memory()

    assert extra and extra.get("cot"), f"temperature rollout returned no CoC: {extra!r}"
    pred_coc = extra["cot"][0]
    pred_clean = clean_pred_coc(pred_coc)
    score = score_coc(pred_coc, gt["gt_coc_texts"])
    print("\nChain-of-Causation (temperature sample):\n", pred_coc)
    print(
        f"[End-to-End Test] readable={score['readable']} "
        f"jaccard={score['jaccard']:.3f} gt_coverage={score['gt_coverage']:.3f} "
        f"cleaned={pred_clean!r}"
    )
    assert score["readable"], f"temperature CoC is not readable: {pred_coc!r}"
    # Do not pin the sentence: T>0 is a draw even with a fixed seed.

    print("\n[GT comparison]")
    print(
        format_gt_report(
            gt,
            pred_coc=pred_coc,
            pred_xyz=pred_xyz,
            ego_future_xyz=data.get("ego_future_xyz"),
        )
    )
    _print_traj(pred_xyz, pred_rot, data)
    print("[End-to-End Test] Temperature sample completed successfully.")
    if peak["total"] > 0:
        print(
            f"[End-to-End Test] RECORD "
            f"cleaned={pred_clean!r} "
            f"jaccard={score['jaccard']:.3f} "
            f"gt_coverage={score['gt_coverage']:.3f} "
            f"pred_xyz={'yes' if pred_xyz is not None else 'None'} "
            f"rss_gb={peak['resident']/1e9:.2f} "
            f"metal_gb={peak['metal']/1e9:.2f}"
        )
    _cleanup(model, data, model_inputs, extra)

