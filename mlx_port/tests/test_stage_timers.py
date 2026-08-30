"""Stage 1c T1.1 clock math — no model, no Metal."""

import time

from mlx_port.stage_timers import (
    G1_DOMINANT,
    StageClock,
    bind_clock,
    current_clock,
    is_stage_timers_enabled,
    median_stage_times,
    print_stage_table,
    reset_compiled,
    set_compiled,
    time_stage,
    vlm_step_stage,
)


def test_as_dict_fills_g1_fields_and_names_dominant():
    reset_compiled()
    clock = StageClock(
        encode_ms=100.0,
        prefill_ms=800.0,
        decode_ms=250.0,
        fm_ms=400.0,
        convert_ms=20.0,
        decode_tok=50,
        fm_steps=10,
    )
    times = clock.as_dict()
    for key in (
        "encode_ms",
        "prefill_ms",
        "decode_ms",
        "decode_tok",
        "ms_per_tok",
        "fm_ms",
        "fm_steps",
        "ms_per_fm_step",
        "dominant_stage",
        "compiled",
        "dtype",
    ):
        assert key in times
    assert times["ms_per_tok"] == 5.0
    assert times["ms_per_fm_step"] == 40.0
    assert times["dominant_stage"] == "prefill"
    assert times["compiled"] == {
        "encode": False,
        "prefill": False,
        "decode": False,
        "fm": False,
    }
    assert times["dtype"] == "bfloat16"


def test_set_compiled_updates_as_dict():
    reset_compiled()
    try:
        set_compiled("prefill", True)
        assert StageClock().as_dict()["compiled"]["prefill"] is True
        assert StageClock().as_dict()["compiled"]["fm"] is False
    finally:
        reset_compiled()
    try:
        set_compiled("not-a-stage", True)
    except ValueError as exc:
        assert "unknown compile stage" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown stage")


def test_zero_clock_does_not_divide_by_zero():
    times = StageClock().as_dict()
    assert times["ms_per_tok"] == 0.0
    assert times["ms_per_fm_step"] == 0.0
    assert times["dominant_stage"] == "python-overhead"


def test_time_stage_is_noop_without_bound_clock():
    assert current_clock() is None
    with time_stage("encode"):
        pass
    assert current_clock() is None


def test_time_stage_records_when_bound():
    clock = StageClock()
    with bind_clock(clock):
        assert current_clock() is clock
        with time_stage("decode"):
            time.sleep(0.01)
        clock.decode_tok = 2
    assert current_clock() is None
    assert clock.decode_ms >= 8.0
    times = clock.as_dict()
    assert times["decode_tok"] == 2
    print_stage_table(times)


def test_env_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ALPAMAYO_STAGE_TIMERS", raising=False)
    assert is_stage_timers_enabled() is False
    monkeypatch.setenv("ALPAMAYO_STAGE_TIMERS", "1")
    assert is_stage_timers_enabled() is True


def test_median_stage_times_is_middle_trial_not_mean():
    base = StageClock(
        encode_ms=100.0,
        prefill_ms=800.0,
        decode_ms=250.0,
        fm_ms=400.0,
        convert_ms=20.0,
        decode_tok=12,
        fm_steps=10,
    ).as_dict()
    low = dict(base, encode_ms=90.0, prefill_ms=500.0, total_ms=1260.0)
    high = dict(base, encode_ms=300.0, prefill_ms=9000.0, total_ms=9970.0)
    mid = median_stage_times([low, base, high])
    assert mid["encode_ms"] == 100.0
    assert mid["prefill_ms"] == 800.0
    assert mid["ms_per_tok"] == 20.83
    assert mid["dominant_stage"] == "prefill"
    assert mid["dominant_stage"] in G1_DOMINANT


def test_vlm_step_stage_prompt_is_prefill_one_token_is_decode():
    prompt = type("Ids", (), {"shape": (1, 32777)})()
    first_decode = type("Ids", (), {"shape": (1, 1)})()
    assert vlm_step_stage(prompt) == "prefill"
    assert vlm_step_stage(first_decode) == "decode"


def test_vlm_step_stage_rejects_empty_ids():
    try:
        vlm_step_stage(None)
    except ValueError as exc:
        assert "input_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing input_ids")
    empty = type("Ids", (), {"shape": (1, 0)})()
    try:
        vlm_step_stage(empty)
    except ValueError as exc:
        assert "seq_len=0" in str(exc)
    else:
        raise AssertionError("expected ValueError for seq_len=0")


def test_median_stage_times_rejects_empty():
    try:
        median_stage_times([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError for an empty trial list")
