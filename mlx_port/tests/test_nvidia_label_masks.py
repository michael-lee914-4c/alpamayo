"""NVIDIA SFT label / CoC-extract contracts used by CUDA train and RL.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #17–#22,
which covered the MLX ports of these helpers — not the NVIDIA source.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SPECIAL_TOKEN_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC), str(_REPO / "finetune")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _ensure_base_model_specials() -> None:
    """Use the real NVIDIA module when hydra/transformers are present."""
    _ensure_src_on_path()
    if "alpamayo_r1.models.base_model" in sys.modules:
        return
    try:
        import alpamayo_r1.models.base_model  # noqa: F401
        return
    except Exception:
        sys.modules.pop("alpamayo_r1.models.base_model", None)
    import alpamayo_r1  # noqa: F401

    models = ModuleType("alpamayo_r1.models")
    models.__path__ = [str(_SRC / "alpamayo_r1" / "models")]
    sys.modules["alpamayo_r1.models"] = models
    sys.modules["alpamayo_r1.models.base_model"] = SimpleNamespace(
        SPECIAL_TOKENS={k: f"<|{k}|>" for k in _SPECIAL_TOKEN_KEYS},
        IGNORE_INDEX=-100,
    )


def _stub_transformers_if_missing() -> None:
    try:
        import transformers  # noqa: F401
        if hasattr(transformers, "AutoTokenizer"):
            return
    except ImportError:
        pass
    stub = sys.modules.get("transformers")
    if stub is None or not hasattr(stub, "AutoTokenizer"):
        stub = ModuleType("transformers")
    stub.AutoProcessor = getattr(stub, "AutoProcessor", object)
    stub.AutoTokenizer = getattr(stub, "AutoTokenizer", object)
    stub.StoppingCriteria = getattr(stub, "StoppingCriteria", type("StoppingCriteria", (), {}))
    gen = sys.modules.get("transformers.generation") or ModuleType("transformers.generation")
    logits = sys.modules.get("transformers.generation.logits_process") or ModuleType(
        "transformers.generation.logits_process"
    )
    if not hasattr(logits, "LogitsProcessor"):
        logits.LogitsProcessor = type("LogitsProcessor", (), {})
    stub.generation = gen
    sys.modules["transformers"] = stub
    sys.modules["transformers.generation"] = gen
    sys.modules["transformers.generation.logits_process"] = logits


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


class _Tok:
    """Minimal convert_tokens_to_ids map for NVIDIA label helpers."""

    def __init__(self, extra: dict[str, int] | None = None):
        self._ids = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "assistant": 3,
            "system": 4,
            "user": 5,
            "<|traj_future_start|>": 10,
            "<|traj_future_end|>": 11,
            "<|traj_future|>": 12,
            "<|cot_start|>": 20,
            "<|cot_end|>": 21,
        }
        if extra:
            self._ids.update(extra)

    def convert_tokens_to_ids(self, name):
        if name not in self._ids:
            raise KeyError(name)
        return self._ids[name]


def _load_label_mask():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    _ensure_base_model_specials()
    try:
        from alpamayo_r1.utils.get_label_mask import (
            fill_masks_between_special_tokens,
            get_assistant_mask,
            get_label_mask,
            get_role_eos_mask,
        )
    except Exception as exc:
        pytest.skip(f"NVIDIA get_label_mask unavailable: {exc}")
    return SimpleNamespace(
        fill=fill_masks_between_special_tokens,
        get_label_mask=get_label_mask,
        get_assistant_mask=get_assistant_mask,
        get_role_eos_mask=get_role_eos_mask,
    )


def _load_token_utils():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    path = _SRC / "alpamayo_r1" / "models" / "token_utils.py"
    spec = importlib.util.spec_from_file_location("nv_token_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nv_token_utils"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA token_utils unavailable: {exc}")
    return module


def _load_conversation():
    _ensure_src_on_path()
    _ensure_base_model_specials()
    path = _SRC / "alpamayo_r1" / "chat_template" / "conversation.py"
    spec = importlib.util.spec_from_file_location("nv_conversation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nv_conversation"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA conversation unavailable: {exc}")
    return module


def test_get_label_mask_includes_end_and_unions_components(torch):
    nv = _load_label_mask()
    tok = _Tok()
    # cot span 2..4, traj span 6..9 (end tokens must be True).
    ids = torch.tensor([[0, 1, 20, 77, 21, 5, 10, 12, 12, 11, 2]], dtype=torch.long)
    cot = nv.get_label_mask(ids, tok, ["cot"])
    assert cot.tolist() == [[False, False, True, True, True, False, False, False, False, False, False]]
    both = nv.get_label_mask(ids, tok, ["cot", "traj_future"])
    assert both.tolist() == [[False, False, True, True, True, False, True, True, True, True, False]]


def test_get_label_mask_rejects_unknown_component(torch):
    nv = _load_label_mask()
    tok = _Tok()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    try:
        nv.get_label_mask(ids, tok, ["not_a_component"])
    except AssertionError as exc:
        assert "not_a_component_start" in str(exc)
    else:
        raise AssertionError("expected AssertionError for an unknown label component")


def test_fill_masks_zips_start_end_and_drops_unpaired_start(torch):
    """NVIDIA zips start/end nonzero rows. An extra start without an end is ignored."""
    nv = _load_label_mask()
    tok = _Tok()
    labels = torch.zeros((1, 8), dtype=torch.bool)
    ids = torch.tensor([[0, 10, 12, 11, 10, 12, 12, 0]], dtype=torch.long)
    filled = nv.fill(
        "<|traj_future_start|>",
        "<|traj_future_end|>",
        ids,
        tok,
        labels,
    )
    # Only the first complete [start, end] pair is filled; the second start has no end.
    assert filled.tolist() == [[False, True, True, True, False, False, False, False]]


def test_get_assistant_mask_skips_header_includes_eos_and_ignores_user(torch):
    nv = _load_label_mask()
    tok = _Tok()
    # BOS + role + newline-equivalent (3-token header), then body, then EOS.
    # user block 0..3, assistant block 4..10 with body at 7..9 and EOS at 10.
    ids = torch.tensor([1, 5, 0, 2, 1, 3, 0, 30, 31, 32, 2], dtype=torch.long)
    mask = nv.get_assistant_mask(tok, ids)
    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]


def test_get_assistant_mask_rejects_unpaired_bos_eos(torch):
    nv = _load_label_mask()
    tok = _Tok()
    try:
        nv.get_assistant_mask(tok, torch.tensor([1, 3, 0, 1, 3, 0, 2], dtype=torch.long))
    except AssertionError as exc:
        assert "bos" in str(exc).lower() and "eos" in str(exc).lower()
    else:
        raise AssertionError("expected AssertionError when BOS/EOS counts differ")


def test_get_role_eos_mask_truncates_extra_bos_and_skips_user(torch):
    nv = _load_label_mask()
    tok = _Tok()
    # user EOS at 3; assistant still open (BOS at 4, no EOS) — infer-style truncate.
    ids = torch.tensor([[1, 5, 9, 2, 1, 3, 30, 31]], dtype=torch.long)
    mask = nv.get_role_eos_mask(ids, tok)
    assert mask.tolist() == [[False, False, False, False, False, False, False, False]]

    closed = torch.tensor([[1, 5, 9, 2, 1, 3, 30, 2]], dtype=torch.long)
    closed_mask = nv.get_role_eos_mask(closed, tok)
    assert closed_mask.tolist() == [[False, False, False, False, False, False, False, True]]


def test_nv_extract_between_special_tokens_last_start_and_missing_end():
    """NVIDIA missing-end → empty; last start wins. MLX (#17) returns the whole text."""
    tu = _load_token_utils()
    got = tu.extract_between_special_tokens(
        [
            "noise <|cot_start|>keep me<|cot_end|> tail",
            "<|cot_start|>first<|cot_start|>second<|cot_end|>",
            "no end marker at all <|cot_start|>orphan",
            "prefix only <|cot_end|> after",
        ],
        "cot",
    )
    assert got == ["keep me", "second", "", "prefix only"]


def test_nv_replace_padding_after_eos_first_only_and_no_eos_noop(torch):
    tu = _load_token_utils()
    ids = torch.tensor([[1, 2, 0, 5, 0, 6], [7, 8, 9, 10, 11, 12]], dtype=torch.long)
    out = tu.replace_padding_after_eos(ids, eos_token_id=0, pad_token_id=-100)
    assert out.tolist() == [[1, 2, 0, -100, -100, -100], [7, 8, 9, 10, 11, 12]]
    assert out is ids


def test_get_component_str_ask_omits_end_and_rejects_both_payloads():
    conv = _load_conversation()
    assert conv.get_component_str("<s>", "<e>", ask_for_component=True) == "<s>"
    assert (
        conv.get_component_str("<s>", "<e>", padding_str="<p><p>") == "<s><p><p><e>"
    )
    assert conv.get_component_str("<s>", "<e>", content_str="why") == "<s>why<e>"
    try:
        conv.get_component_str("<s>", "<e>", content_str="a", padding_str="b")
    except AssertionError as exc:
        assert "Exactly one" in str(exc)
    else:
        raise AssertionError("expected AssertionError when both content and padding are set")
    try:
        conv.get_component_str("<s>", "<e>")
    except AssertionError as exc:
        assert "Exactly one" in str(exc)
    else:
        raise AssertionError("expected AssertionError when neither content nor padding is set")


def test_construct_user_prompt_train_uses_order_gen_uses_prompt_list():
    conv = _load_conversation()
    train = conv.construct_user_prompt(
        ["image", "traj_history", "prompt", "traj_future"],
        ["cot", "traj_future"],
        generation_mode=False,
    )
    assert train == [{"type": "text", "text": "output the future trajectory."}]
    gen = conv.construct_user_prompt(
        ["image", "traj_history", "prompt", "traj_future"],
        ["cot", "traj_future"],
        generation_mode=True,
    )
    assert gen == [
        {
            "type": "text",
            "text": (
                "output the chain-of-thought reasoning of the driving process, "
                "then output the future trajectory."
            ),
        }
    ]


def test_construct_traj_future_ask_is_start_only():
    conv = _load_conversation()
    pads = conv.construct_traj_future(3, ask_for_component=False)
    assert pads[0]["text"] == (
        "<|traj_future_start|><|traj_future|><|traj_future|><|traj_future|><|traj_future_end|>"
    )
    ask = conv.construct_traj_future(3, ask_for_component=True)
    assert ask[0]["text"] == "<|traj_future_start|>"


def test_construct_cot_requires_data_unless_asking():
    conv = _load_conversation()
    try:
        conv.construct_cot({}, ask_for_component=False)
    except AssertionError as exc:
        assert "cot not found" in str(exc)
    else:
        raise AssertionError("expected AssertionError when teacher CoC is missing")
    ask = conv.construct_cot({}, ask_for_component=True)
    assert ask[0]["text"] == "<|cot_start|>"
    filled = conv.construct_cot({"cot": "yield"}, ask_for_component=False)
    assert filled[0]["text"] == "<|cot_start|>yield<|cot_end|>"


def test_build_conversation_generation_asks_only_last_assistant_span():
    conv = _load_conversation()
    msgs = conv.build_conversation(
        {"cot": "yield"},
        num_tokens_per_history_traj=2,
        num_tokens_per_future_traj=3,
        components_order=["traj_history", "prompt", "cot", "traj_future"],
        components_prompt=["cot", "traj_future"],
        generation_mode=True,
    )
    assistant = "".join(part["text"] for part in msgs[2]["content"])
    assert assistant == "<|cot_start|>yield<|cot_end|><|traj_future_start|>"
    user = "".join(part["text"] for part in msgs[1]["content"])
    assert user.endswith(
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
