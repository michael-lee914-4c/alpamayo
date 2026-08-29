"""Stage 1c T1.1 clock math — no model, no Metal."""

import time

from mlx_port.stage_timers import (
    StageClock,
    bind_clock,
    current_clock,
    is_stage_timers_enabled,
    print_stage_table,
    time_stage,
)


def test_as_dict_fills_g1_fields_and_names_dominant():
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
