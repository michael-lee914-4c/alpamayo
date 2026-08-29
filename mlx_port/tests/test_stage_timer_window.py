"""T1.1 warmup + 3 warm windows on the default clip (Stage 1c G1).

Discard the first window after load. Record the next three. Print the median.
Does not replace the cold-trial numbers already in reports/stage1c_progress.html.
"""

from __future__ import annotations

import gc

import mlx.core as mx
import pytest

from mlx_port.inference import sample_trajectories_from_data_with_vlm_rollout
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.stage_timers import G1_DOMINANT, G1_MS_KEYS, median_stage_times, print_stage_table
from mlx_port.tests.test_end_to_end_inference import (
    CHECKPOINT,
    CLIP_ID,
    _load_clip,
    _prepare_inputs,
)


def _one_greedy_window(model, model_inputs, max_gen_len: int) -> dict:
    pred_xyz, _pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
        model=model,
        data=model_inputs,
        top_p=1.0,
        temperature=0.0,
        num_traj_samples=1,
        max_generation_length=max_gen_len,
        return_extra=True,
        vlm_only=False,
    )
    if pred_xyz is None:
        raise RuntimeError("warmup timing window returned no trajectory")
    if extra is None or "stage_times" not in extra:
        raise RuntimeError("stage_times missing; ALPAMAYO_STAGE_TIMERS must be on")
    gc.collect()
    mx.clear_cache()
    return extra["stage_times"]


@pytest.mark.slow
@pytest.mark.parametrize("max_gen_len", [256])
def test_greedy_window_warmup_then_three_warm_trials(max_gen_len, monkeypatch):
    monkeypatch.setenv("ALPAMAYO_STAGE_TIMERS", "1")
    gt, data = _load_clip()
    print(f"[WARM] clip={CLIP_ID}  load once, discard 1 window, time 3")
    print("[WARM] GT CoC:")
    for text in gt["gt_coc_texts"]:
        print(f"  - {text}")

    model = AlpamayoR1MLX.from_pretrained(
        CHECKPOINT,
        load_expert=True,
        dtype=mx.bfloat16,
    )
    model_inputs = _prepare_inputs(model, data)

    warmup = _one_greedy_window(model, model_inputs, max_gen_len)
    print("[WARMUP] discarded (first window after load)")
    print_stage_table(warmup)

    trials = []
    for i in range(3):
        times = _one_greedy_window(model, model_inputs, max_gen_len)
        trials.append(times)
        print(f"[TRIAL {i + 1}/3]")
        print_stage_table(times)

    median = median_stage_times(trials)
    print("[MEDIAN] of 3 warm trials")
    print_stage_table(median)

    for key in (*G1_MS_KEYS, "decode_tok", "fm_steps", "dominant_stage"):
        assert key in median
    assert median["decode_tok"] > 0
    assert median["fm_steps"] == 10
    assert median["dominant_stage"] in G1_DOMINANT
    print(
        f"[WARM] median dominant={median['dominant_stage']}  "
        f"total_ms={median['total_ms']:.1f}"
    )
