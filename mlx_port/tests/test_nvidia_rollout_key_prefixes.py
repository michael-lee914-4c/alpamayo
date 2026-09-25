"""vLLM rollout weight names are rewritten before the policy mapper.

Weight-free. Complementary to open #42 (policy ``reasoning_vla.`` / ``qkv_proj``
rewrites + FSDP vocab trim). ``rollout_map_local_key_to_hf_key`` first strips
``llm.model.``, ``llm.lm_head.``, ``model.vlm.model.visual.``,
``vlm.model.visual.``, and ``language_model.*``, then vision ``qkv_proj → qkv``.
A missing prefix leaves shards unmatched and silently skips vision / LM-head
sync between policy and rollout.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MAP = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "weight_mapper.py"


def _rollout_prefix_rewrite(name: str) -> str:
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
    return name


def test_source_has_rollout_prefix_branches():
    src = _MAP.read_text()
    assert 'if name.startswith("llm.model."):' in src
    assert 'elif name.startswith("llm.lm_head."):' in src
    assert 'elif name.startswith("model.vlm.model.visual."):' in src
    assert 'elif name.startswith("vlm.model.visual."):' in src
    assert 'elif name.startswith("vlm."):' in src
    assert 'if name.startswith("language_model.model."):' in src
    assert 'name.replace(".attn.qkv_proj.", ".attn.qkv.", 1)' in src
    assert "processed_name = self.policy_map_local_key_to_hf_key(name)" in src


def test_rollout_prefixes_normalize_to_hf_keyspace():
    cases = [
        ("llm.model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.weight"),
        ("llm.lm_head.weight", "lm_head.weight"),
        ("model.vlm.model.visual.patch_embed.weight", "visual.patch_embed.weight"),
        ("vlm.model.visual.patch_embed.weight", "visual.patch_embed.weight"),
        ("vlm.model.layers.0.mlp.down_proj.weight", "model.layers.0.mlp.down_proj.weight"),
        ("language_model.model.embed_tokens.weight", "model.embed_tokens.weight"),
        ("language_model.lm_head.weight", "lm_head.weight"),
        ("language_model.layers.0.input_layernorm.weight", "model.layers.0.input_layernorm.weight"),
        (
            "visual.blocks.0.attn.qkv_proj.weight",
            "visual.blocks.0.attn.qkv.weight",
        ),
    ]
    for raw, expected in cases:
        assert _rollout_prefix_rewrite(raw) == expected, raw


def test_non_visual_qkv_proj_is_not_rewritten():
    # The qkv_proj → qkv rewrite is gated on a visual. prefix.
    raw = "model.layers.0.self_attn.qkv_proj.weight"
    assert _rollout_prefix_rewrite(raw) == raw
