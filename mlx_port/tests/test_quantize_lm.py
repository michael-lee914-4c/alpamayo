"""T3.1: language-tower affine 4-bit; vision / expert / lm_head stay dense."""

import json
import os

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import LanguageModel, Qwen3VLDecoderLayer

from mlx_port.models.compiled_backbone import install_compiled_prefill
from mlx_port.models.quantize_lm import (
    DEFAULT_LM4_DIRNAME,
    LM4_CONFIG_NAME,
    LM4_DIR_ENV,
    LM4_WEIGHTS_NAME,
    QUANT_BITS,
    QUANT_GROUP_SIZE,
    QUANT_MODE,
    QUANT_MODE_ENV,
    QUANT_SPEC,
    apply_language_tower_quant,
    keep_dense,
    language_tower_predicate,
    lm4_checkpoint_ready,
    lm_quant_enabled,
    load_language_tower,
    mark_language_tower_dense,
    quantize_language_tower,
    resolve_lm4_dir,
    resolve_quant_mode,
    save_language_tower,
)
from mlx_port.stage_timers import quantized_flags, reset_quantized


def _tiny_text_config() -> TextConfig:
    return TextConfig(
        model_type="qwen3_vl",
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=32,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=10000.0,
        max_position_embeddings=128,
        rope_scaling={"type": "default", "mrope_section": [6, 5, 5]},
    )


def test_lm_quant_enabled_env_is_t31_only():
    prev = os.environ.pop(QUANT_MODE_ENV, None)
    try:
        assert lm_quant_enabled(True) is True
        assert lm_quant_enabled(False) is False
        os.environ[QUANT_MODE_ENV] = "none"
        assert lm_quant_enabled(True) is False
        os.environ[QUANT_MODE_ENV] = "lm4"
        assert lm_quant_enabled(False) is True
        os.environ[QUANT_MODE_ENV] = "all4"
        assert lm_quant_enabled(True) is False
        assert resolve_quant_mode() == "all4"
        os.environ[QUANT_MODE_ENV] = "nvfp4"
        try:
            lm_quant_enabled(True)
        except ValueError as exc:
            assert "not supported" in str(exc)
        else:
            raise AssertionError("expected ValueError for leftover ALPAMAYO_QUANT")
    finally:
        if prev is None:
            os.environ.pop(QUANT_MODE_ENV, None)
        else:
            os.environ[QUANT_MODE_ENV] = prev


def test_mark_language_tower_dense_sets_bf16_flags():
    reset_quantized()
    mark_language_tower_dense()
    flags = quantized_flags()
    assert flags["lm"] == "bf16"
    assert flags["vision"] == "bf16"
    assert flags["expert"] == "bf16"


def test_keep_dense_matches_t31_substrings():
    assert keep_dense("lm_head")
    assert keep_dense("model.embed_tokens")
    assert keep_dense("vision_tower.patch_embed.proj")
    assert keep_dense("expert.language_model.model.layers.0.self_attn.q_proj")
    assert keep_dense("action_in_proj.encoder")
    assert not keep_dense("model.layers.0.self_attn.q_proj")
    assert not keep_dense("model.layers.0.inner.mlp.gate_proj")


def test_predicate_keeps_head_and_embeds_dense():
    lin = nn.Linear(64, 64)
    emb = nn.Embedding(32, 64)
    assert language_tower_predicate("lm_head", lin) is False
    assert language_tower_predicate("model.embed_tokens", emb) is False
    spec = language_tower_predicate("model.layers.0.self_attn.q_proj", lin)
    assert spec == {"bits": QUANT_BITS, "group_size": QUANT_GROUP_SIZE, "mode": QUANT_MODE}
    assert language_tower_predicate("model.layers.0.input_layernorm", object()) is False


def test_quantize_language_tower_refuses_vlm_and_full_model():
    class FakeVLM:
        vision_tower = object()
        lm_head = object()

    class FakeFull:
        vlm = object()
        expert = object()
        lm_head = object()

    class FakeExpert:
        language_model = object()

    try:
        quantize_language_tower(FakeVLM())
    except ValueError as exc:
        assert "not the full VLM" in str(exc)
    else:
        raise AssertionError("expected ValueError for vision_tower")

    try:
        quantize_language_tower(FakeFull())
    except ValueError as exc:
        assert "not AlpamayoR1MLX" in str(exc)
    else:
        raise AssertionError("expected ValueError for full model")

    try:
        quantize_language_tower(FakeExpert())
    except ValueError as exc:
        assert "lm_head" in str(exc)
    else:
        raise AssertionError("expected ValueError for expert")


def test_quantize_tiny_language_model_4bit_keeps_head_and_embed():
    reset_quantized()
    lm = LanguageModel(_tiny_text_config())
    mx.eval(lm.parameters())
    summary = quantize_language_tower(lm)
    assert summary["n_quantized_linear"] == 14  # 2 layers × (qkv o + gate up down)
    assert isinstance(lm.lm_head, nn.Linear)
    assert isinstance(lm.model.embed_tokens, nn.Embedding)
    q = lm.model.layers[0].self_attn.q_proj
    assert isinstance(q, nn.QuantizedLinear)
    assert q.bits == QUANT_BITS
    assert q.group_size == QUANT_GROUP_SIZE
    assert q.mode == QUANT_MODE
    flags = quantized_flags()
    assert flags["lm"] == "affine-4-gs64"
    assert flags["vision"] == "bf16"
    assert flags["expert"] == "bf16"
    again = quantize_language_tower(lm)
    assert again["n_quantized_linear"] == summary["n_quantized_linear"]


def test_quantize_after_compiled_prefill_wrap_uses_inner_path():
    reset_quantized()
    lm = LanguageModel(_tiny_text_config())
    mx.eval(lm.parameters())
    n = install_compiled_prefill(lm)
    assert n == 2
    summary = quantize_language_tower(lm)
    assert summary["n_quantized_linear"] == 14
    inner = lm.model.layers[0].inner
    assert isinstance(inner.self_attn.q_proj, nn.QuantizedLinear)
    assert isinstance(inner.mlp.down_proj, nn.QuantizedLinear)
    assert isinstance(lm.lm_head, nn.Linear)


def test_resolve_lm4_dir_kwarg_env_and_default(tmp_path):
    default = resolve_lm4_dir("/ckpt/alpamayo")
    assert default.endswith(f"/ckpt/alpamayo/{DEFAULT_LM4_DIRNAME}")
    assert resolve_lm4_dir("/ckpt/alpamayo", str(tmp_path / "custom")) == str(
        (tmp_path / "custom").resolve()
    )
    prev = os.environ.pop(LM4_DIR_ENV, None)
    try:
        os.environ[LM4_DIR_ENV] = str(tmp_path / "envdir")
        assert resolve_lm4_dir("/ckpt/alpamayo") == str((tmp_path / "envdir").resolve())
    finally:
        if prev is None:
            os.environ.pop(LM4_DIR_ENV, None)
        else:
            os.environ[LM4_DIR_ENV] = prev


def test_lm4_checkpoint_ready_rejects_incomplete_pair(tmp_path):
    dest = tmp_path / "mlx_lm4"
    dest.mkdir()
    assert lm4_checkpoint_ready(str(dest)) is False
    (dest / LM4_WEIGHTS_NAME).write_bytes(b"x")
    try:
        lm4_checkpoint_ready(str(dest))
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for weights without config")


def test_save_and_load_language_tower_roundtrip(tmp_path):
    reset_quantized()
    src = LanguageModel(_tiny_text_config())
    mx.eval(src.parameters())
    install_compiled_prefill(src)
    quantize_language_tower(src)
    dest = str(tmp_path / "mlx_lm4")
    saved = save_language_tower(src, dest)
    assert saved["n_quantized_linear"] == 14
    assert os.path.isfile(os.path.join(dest, LM4_WEIGHTS_NAME))
    assert os.path.isfile(os.path.join(dest, LM4_CONFIG_NAME))
    assert lm4_checkpoint_ready(dest) is True

    reset_quantized()
    dst = LanguageModel(_tiny_text_config())
    mx.eval(dst.parameters())
    install_compiled_prefill(dst)
    loaded = load_language_tower(dst, dest)
    assert loaded["n_quantized_linear"] == 14
    assert loaded["source"] == "disk"
    assert isinstance(dst.lm_head, nn.Linear)
    assert isinstance(dst.model.layers[0].inner.self_attn.q_proj, nn.QuantizedLinear)
    src_w = src.model.layers[0].inner.self_attn.q_proj.weight
    dst_w = dst.model.layers[0].inner.self_attn.q_proj.weight
    assert bool((src_w == dst_w).all())
    assert bool((src.lm_head.weight == dst.lm_head.weight).all())
    flags = quantized_flags()
    assert flags["lm"] == QUANT_SPEC


def test_apply_language_tower_quant_live_packs_then_loads(tmp_path):
    dest = str(tmp_path / "mlx_lm4")
    reset_quantized()
    first = LanguageModel(_tiny_text_config())
    mx.eval(first.parameters())
    install_compiled_prefill(first)
    out = apply_language_tower_quant(first, dest)
    assert out["source"] == "live-pack"
    assert lm4_checkpoint_ready(dest)

    reset_quantized()
    second = LanguageModel(_tiny_text_config())
    mx.eval(second.parameters())
    install_compiled_prefill(second)
    again = apply_language_tower_quant(second, dest)
    assert again["source"] == "disk"
    src_w = first.model.layers[0].inner.self_attn.q_proj.weight
    dst_w = second.model.layers[0].inner.self_attn.q_proj.weight
    assert bool((src_w == dst_w).all())


def test_load_language_tower_rejects_spec_mismatch(tmp_path):
    reset_quantized()
    src = LanguageModel(_tiny_text_config())
    mx.eval(src.parameters())
    install_compiled_prefill(src)
    dest = str(tmp_path / "mlx_lm4")
    quantize_language_tower(src)
    save_language_tower(src, dest)
    cfg_path = os.path.join(dest, LM4_CONFIG_NAME)
    with open(cfg_path) as f:
        cfg = json.loads(f.read())
    cfg["bits"] = 8
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    reset_quantized()
    dst = LanguageModel(_tiny_text_config())
    mx.eval(dst.parameters())
    install_compiled_prefill(dst)
    try:
        load_language_tower(dst, dest)
    except ValueError as exc:
        assert "bits" in str(exc)
    else:
        raise AssertionError("expected ValueError for bits mismatch")


def test_resize_embeddings_grows_lm_head_with_embed():
    from mlx_port.vlm_loader import _resize_embeddings

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(4, 8, bias=False)

    class VLM:
        def __init__(self):
            self.language_model = LM()

    vlm = VLM()
    mx.eval(vlm.language_model.parameters())
    _resize_embeddings(vlm, 12)
    assert vlm.language_model.model.embed_tokens.weight.shape == (12, 4)
    assert vlm.language_model.lm_head.weight.shape == (12, 4)


def test_quantize_does_not_touch_a_sibling_expert():
    reset_quantized()

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LanguageModel(_tiny_text_config())
            self.expert = Qwen3VLDecoderLayer(_tiny_text_config(), 0)

    holder = Holder()
    mx.eval(holder.parameters())
    expert_q = holder.expert.self_attn.q_proj
    assert isinstance(expert_q, nn.Linear)
    quantize_language_tower(holder.language_model)
    assert holder.expert.self_attn.q_proj is expert_q
    assert isinstance(holder.expert.self_attn.q_proj, nn.Linear)
