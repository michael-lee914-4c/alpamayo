"""NVIDIA generate-stop, tokenize defaults, and recursive device cast.

Weight-free. Complementary to MLX ``StopAfterEOS`` tests in
``test_inference_row7.py`` — this is the CUDA helper path (#17–#25 did not
cover NVIDIA ``StopAfterEOS``, ``tokenize_messages`` kwargs, or ``to_device``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _stub_transformers_if_missing() -> None:
    try:
        import transformers  # noqa: F401

        if hasattr(transformers, "AutoTokenizer") and hasattr(transformers, "StoppingCriteria"):
            return
    except ImportError:
        pass
    stub = sys.modules.get("transformers")
    if stub is None or not hasattr(stub, "AutoTokenizer"):
        stub = ModuleType("transformers")
    stub.AutoProcessor = getattr(stub, "AutoProcessor", object)
    stub.AutoTokenizer = getattr(stub, "AutoTokenizer", object)
    stub.StoppingCriteria = getattr(stub, "StoppingCriteria", type("StoppingCriteria", (), {}))
    sys.modules["transformers"] = stub


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_token_utils():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    try:
        from alpamayo_r1.models.token_utils import StopAfterEOS
    except Exception as exc:
        pytest.skip(f"NVIDIA token_utils unavailable: {exc}")
    return StopAfterEOS


def _load_helper():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    try:
        from alpamayo_r1.helper import to_device, tokenize_messages
    except Exception as exc:
        pytest.skip(f"NVIDIA helper unavailable: {exc}")
    return to_device, tokenize_messages


def test_nv_stop_after_eos_waits_one_token_and_all_rows(torch):
    """NVIDIA stops only after every row has already seen EOS on a prior call.

    The extra token after the first EOS is generated; there is no ``reset()``.
    """
    StopAfterEOS = _load_token_utils()
    stop = StopAfterEOS(eos_token_id=7)
    assert stop(torch.tensor([[1, 2, 7], [3, 4, 5]]), None) is False
    assert stop.eos_found.tolist() == [True, False]
    # Row 1 emits EOS now. Decision is made *before* the update, so still False.
    assert stop(torch.tensor([[1, 2, 7, 9], [3, 4, 5, 7]]), None) is False
    assert stop.eos_found.tolist() == [True, True]
    assert stop(torch.tensor([[1, 2, 7, 9, 0], [3, 4, 5, 7, 0]]), None) is True
    # Sticky: NVIDIA has no reset, so the next sample would stop immediately.
    assert stop(torch.tensor([[99]]), None) is True


def test_tokenize_messages_continues_final_and_skips_gen_prompt(torch):
    _to_device, tokenize_messages = _load_helper()
    del _to_device, torch

    captured: dict = {}

    class _Proc:
        def apply_chat_template(self, messages, **kwargs):
            captured.update(kwargs)
            return {"ok": messages}

    out = tokenize_messages(_Proc(), [{"role": "assistant", "content": "x"}])
    assert out == {"ok": [{"role": "assistant", "content": "x"}]}
    assert captured["tokenize"] is True
    assert captured["add_generation_prompt"] is False
    assert captured["continue_final_message"] is True
    assert captured["return_dict"] is True
    assert captured["return_tensors"] == "pt"

    tokenize_messages(_Proc(), [], add_generation_prompt=True, continue_final_message=False)
    assert captured["add_generation_prompt"] is True
    assert captured["continue_final_message"] is False


def test_to_device_casts_nested_tensors_and_leaves_text(torch):
    to_device, _tokenize = _load_helper()
    del _tokenize
    payload = {
        "ids": torch.tensor([1, 2], dtype=torch.int64),
        "nested": [{"x": torch.tensor([0.5], dtype=torch.float32)}, "keep", b"bytes"],
    }
    out = to_device(payload, device="cpu", dtype=torch.float64)
    assert out["ids"].dtype == torch.float64
    assert out["nested"][0]["x"].dtype == torch.float64
    assert out["nested"][1] == "keep"
    assert out["nested"][2] == b"bytes"
    assert to_device(3, device="cpu") == 3
