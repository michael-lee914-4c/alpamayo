"""Policy/rollout keys rewrite to HF names and trim FSDP vocab padding.

Weight-free. Complementary to open #38 (``vlm.*`` shard filter). A missed
``reasoning_vla.`` / ``qkv_proj`` rewrite or a padded ``lm_head`` row copy
loads the wrong slice with no runtime error.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MAP = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "weight_mapper.py"


def _apply_rewrites(name: str, rules: list[tuple[str, str]]) -> str:
    for old, new in rules:
        name = name.replace(old, new, 1)
    return name


_POLICY_RULES = [
    ("reasoning_vla.", ""),
    ("vlm.", ""),
    ("model.language_model.", "model."),
    ("model.visual.", "visual."),
]


def _rollout_map_local_key_to_hf_key(name: str) -> str:
    if name.startswith("llm.model."):
        name = name.replace("llm.model.", "model.", 1)
    elif name.startswith("llm.lm_head."):
        name = name.replace("llm.lm_head.", "lm_head.", 1)
    elif name.startswith("model.vlm.model.visual."):
        name = name.replace("model.vlm.model.visual.", "visual.", 1)
    elif name.startswith("vlm.model.visual."):
        name = name.replace("vlm.model.visual.", "visual.", 1)
    elif name.startswith("vlm."):
        name = name[len("vlm.") :]
    if name.startswith("language_model.model."):
        name = name.replace("language_model.model.", "model.", 1)
    elif name.startswith("language_model.lm_head."):
        name = name.replace("language_model.lm_head.", "lm_head.", 1)
    elif name.startswith("language_model."):
        name = name.replace("language_model.", "model.", 1)
    if name.startswith("visual.") and ".attn.qkv_proj." in name:
        name = name.replace(".attn.qkv_proj.", ".attn.qkv.", 1)
    return _apply_rewrites(name, _POLICY_RULES)


def _trim_vocab(name: str, tensor, vocab_size: int):
    if (
        name in ("model.embed_tokens.weight", "lm_head.weight")
        and tensor.ndim == 2
        and tensor.shape[0] > vocab_size
        and tensor.shape[1] > 0
    ):
        return tensor[:vocab_size]
    return tensor


def test_source_rewrites_qkv_and_trims_padded_vocab():
    src = _MAP.read_text()
    assert '("reasoning_vla.", ""),' in src
    assert '("vlm.", ""),' in src
    assert '("model.language_model.", "model."),' in src
    assert '("model.visual.", "visual."),' in src
    assert 'name = name.replace(".attn.qkv_proj.", ".attn.qkv.", 1)' in src
    assert 'name in ("model.embed_tokens.weight", "lm_head.weight")' in src
    assert "trimmed.append((name, t[:vocab_size]))" in src


def test_policy_and_rollout_rewrites_collapse_to_hf_keys():
    policy = _apply_rewrites(
        "reasoning_vla.vlm.model.language_model.layers.0.weight", _POLICY_RULES
    )
    assert policy == "model.layers.0.weight"
    visual = _apply_rewrites(
        "reasoning_vla.vlm.model.visual.blocks.0.attn.qkv.weight", _POLICY_RULES
    )
    assert visual == "visual.blocks.0.attn.qkv.weight"
    rollout = _rollout_map_local_key_to_hf_key(
        "model.vlm.model.visual.blocks.0.attn.qkv_proj.weight"
    )
    assert rollout == "visual.blocks.0.attn.qkv.weight"
    assert _rollout_map_local_key_to_hf_key("llm.lm_head.weight") == "lm_head.weight"


def test_fsdp_padded_lm_head_is_trimmed_to_vocab():
    torch = pytest.importorskip("torch")
    padded = torch.arange(150 * 4, dtype=torch.float32).reshape(150, 4)
    trimmed = _trim_vocab("lm_head.weight", padded, vocab_size=100)
    assert trimmed.shape == (100, 4)
    torch.testing.assert_close(trimmed, padded[:100])
    kept = _trim_vocab("expert.weight", padded, vocab_size=100)
    assert kept.shape == (150, 4)
