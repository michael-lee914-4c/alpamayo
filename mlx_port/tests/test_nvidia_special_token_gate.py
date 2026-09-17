"""Model tokenizer special-token gate vs QwenProcessor.

Weight-free. Complementary to open #24 (conversation / label masks that
assume ``SPECIAL_TOKENS`` exist). ``ReasoningVLAConfig.add_special_tokens``
defaults to False and then only ``TRAJ_TOKEN`` values are added. The
Qwen processor always adds the full ``SPECIAL_TOKENS`` set. A processor
built from the model-config path with the default flag will not have
``<|cot_start|>`` / ``<|meta_action_start|>``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"


def _traj_and_special() -> tuple[set[str], set[str]]:
    # Keep this in sync with src/alpamayo_r1/models/base_model.py.
    traj = {
        "history": "<|traj_history|>",
        "future": "<|traj_future|>",
        "history_start": "<|traj_history_start|>",
        "future_start": "<|traj_future_start|>",
        "history_end": "<|traj_history_end|>",
        "future_end": "<|traj_future_end|>",
    }
    keys = [
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
    special = {k: "<|" + k + "|>" for k in keys}
    return set(traj.values()), set(special.values())


def test_add_special_tokens_defaults_false_and_gates_traj_only():
    src = _BASE.read_text()
    assert "add_special_tokens: bool = False" in src
    assert "if self.add_special_tokens:" in src
    assert "tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)" in src
    assert "if config.add_special_tokens:" in src


def test_qwen_processor_always_adds_full_special_set():
    src = _QWEN.read_text()
    assert "special_tokens = list(SPECIAL_TOKENS.values())" in src
    assert "tokenizer.add_tokens(special_tokens, special_tokens=True)" in src
    assert "if self.add_special_tokens" not in src
    assert "TRAJ_TOKEN.values()" not in src


def test_default_model_tokenizer_is_missing_cot_and_meta_action():
    traj_vals, special_vals = _traj_and_special()
    src = _BASE.read_text()
    assert 'TRAJ_TOKEN = {' in src
    assert '"cot_start",' in src
    assert "SPECIAL_TOKENS = {k: \"<|\" + k + \"|>\" for k in SPECIAL_TOKENS_KEYS}" in src

    assert traj_vals < special_vals
    missing_when_gated = special_vals - traj_vals
    assert "<|cot_start|>" in missing_when_gated
    assert "<|cot_end|>" in missing_when_gated
    assert "<|meta_action_start|>" in missing_when_gated
    assert "<|image_pad|>" in missing_when_gated
    assert "<|traj_future_start|>" not in missing_when_gated
