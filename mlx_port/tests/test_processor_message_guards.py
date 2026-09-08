"""create_message exclusive-flag guards. No weights. PIL + NumPy only."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np

def _ensure_transformers_stub() -> None:
    try:
        import transformers  # noqa: F401
        return
    except ImportError:
        pass
    existing = sys.modules.get("transformers")
    if existing is not None and hasattr(existing, "AutoTokenizer"):
        return
    stub = ModuleType("transformers")
    stub.AutoProcessor = object
    stub.AutoTokenizer = object
    stub.StoppingCriteria = type("StoppingCriteria", (), {})
    gen = ModuleType("transformers.generation")
    logits = ModuleType("transformers.generation.logits_process")

    class _LogitsProcessor:
        pass

    logits.LogitsProcessor = _LogitsProcessor
    stub.generation = gen
    sys.modules["transformers"] = stub
    sys.modules["transformers.generation"] = gen
    sys.modules["transformers.generation.logits_process"] = logits


_ensure_transformers_stub()

from mlx_port.processor import create_message


def test_create_message_teacher_cot_exclusive_with_sft_stage():
    frames = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    try:
        create_message(frames, teacher_cot="yield", sft_stage="stage1")
    except ValueError as exc:
        assert "exclusive" in str(exc)
    else:
        raise AssertionError("expected ValueError when teacher_cot and sft_stage are both set")


def test_create_message_rejects_unknown_sft_stage_and_empty_teacher():
    frames = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    try:
        create_message(frames, sft_stage="stage2")
    except ValueError as exc:
        assert "stage1" in str(exc)
    else:
        raise AssertionError("expected ValueError for sft_stage other than stage1")
    try:
        create_message(frames, teacher_cot="   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError for whitespace-only teacher_cot")


def test_create_message_infer_stops_at_cot_start():
    frames = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    infer = create_message(frames)
    assert infer[2]["content"][0]["text"] == "<|cot_start|>"
    teacher = create_message(frames, teacher_cot=" yield to the pedestrian ")
    assert teacher[2]["content"][0]["text"] == (
        "<|cot_start|>yield to the pedestrian<|cot_end|><|traj_future_start|>"
    )
