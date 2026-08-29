"""Stage wall-clock for one inference window (Stage 1c T1.1 / G1).

Enable with ``ALPAMAYO_STAGE_TIMERS=1``. When a clock is bound, vision encode
ends with ``mx.eval`` so encode time is not lazy-attributed to prefill.
Token outputs are unchanged; this is a host barrier only.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator

STAGE_TIMER_ENV = "ALPAMAYO_STAGE_TIMERS"


def is_stage_timers_enabled() -> bool:
    return os.environ.get(STAGE_TIMER_ENV, "0").lower() in ("1", "true", "yes", "on")


@dataclass
class StageClock:
    encode_ms: float = 0.0
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    fm_ms: float = 0.0
    convert_ms: float = 0.0
    decode_tok: int = 0
    fm_steps: int = 0
    notes: list[str] = field(default_factory=list)

    def add_ms(self, name: str, ms: float) -> None:
        attr = {
            "encode": "encode_ms",
            "prefill": "prefill_ms",
            "decode": "decode_ms",
            "fm": "fm_ms",
            "action": "fm_ms",
            "convert": "convert_ms",
            "python-overhead": "convert_ms",
        }[name]
        setattr(self, attr, getattr(self, attr) + float(ms))

    def add_seconds(self, name: str, seconds: float) -> None:
        self.add_ms(name, seconds * 1000.0)

    def as_dict(self) -> dict:
        ms_per_tok = self.decode_ms / self.decode_tok if self.decode_tok else 0.0
        ms_per_fm = self.fm_ms / self.fm_steps if self.fm_steps else 0.0
        stages = {
            "encode": self.encode_ms,
            "prefill": self.prefill_ms,
            "decode": self.decode_ms,
            "action": self.fm_ms,
            "python-overhead": self.convert_ms,
        }
        dominant = max(stages, key=stages.get)
        if stages[dominant] <= 0.0:
            dominant = "python-overhead"
        return {
            "encode_ms": round(self.encode_ms, 1),
            "prefill_ms": round(self.prefill_ms, 1),
            "decode_ms": round(self.decode_ms, 1),
            "decode_tok": int(self.decode_tok),
            "ms_per_tok": round(ms_per_tok, 2),
            "fm_ms": round(self.fm_ms, 1),
            "fm_steps": int(self.fm_steps),
            "ms_per_fm_step": round(ms_per_fm, 2),
            "convert_ms": round(self.convert_ms, 1),
            "total_ms": round(sum(stages.values()), 1),
            "dominant_stage": dominant,
            "compiled": {
                "encode": False,
                "prefill": False,
                "decode": False,
                "fm": False,
            },
            "dtype": "bfloat16",
        }


_current: ContextVar[StageClock | None] = ContextVar("alpamayo_stage_clock", default=None)


def current_clock() -> StageClock | None:
    return _current.get()


@contextmanager
def bind_clock(clock: StageClock | None) -> Iterator[StageClock | None]:
    token = _current.set(clock)
    try:
        yield clock
    finally:
        _current.reset(token)


@contextmanager
def time_stage(name: str) -> Iterator[None]:
    clock = current_clock()
    if clock is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        clock.add_seconds(name, time.perf_counter() - t0)


def print_stage_table(times: dict) -> None:
    print(
        "[STAGE] "
        f"encode_ms={times['encode_ms']:.1f}  "
        f"prefill_ms={times['prefill_ms']:.1f}  "
        f"decode_ms={times['decode_ms']:.1f}  "
        f"decode_tok={times['decode_tok']}  "
        f"ms_per_tok={times['ms_per_tok']:.2f}  "
        f"fm_ms={times['fm_ms']:.1f}  "
        f"fm_steps={times['fm_steps']}  "
        f"ms_per_fm_step={times['ms_per_fm_step']:.2f}  "
        f"convert_ms={times['convert_ms']:.1f}  "
        f"total_ms={times['total_ms']:.1f}  "
        f"dominant={times['dominant_stage']}"
    )
