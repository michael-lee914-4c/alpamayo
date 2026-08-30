"""all4: full VLM + expert affine-4; action-in/out stay dense."""

import os

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import LanguageModel

from mlx_port.models.compiled_backbone import install_compiled_prefill
from mlx_port.models.expert_mlx import AlpamayoExpert
from mlx_port.models.quantize_all import (
    ALL4_DIR_ENV,
    ALL4_EXPERT_WEIGHTS_NAME,
    ALL4_VLM_WEIGHTS_NAME,
    DEFAULT_ALL4_DIRNAME,
    all4_checkpoint_ready,
    all4_predicate,
    apply_expert_all4,
    apply_vlm_all4,
    last_dim_packable,
    load_expert_all4,
    load_vlm_all4,
    quantize_expert_all4,
    quantize_vlm_all4,
    resolve_all4_dir,
    save_expert_all4,
    save_vlm_all4,
)
from mlx_port.models.quantize_lm import QUANT_MODE_ENV, resolve_quant_mode
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


class _TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.linear_fc2 = nn.Linear(16, 32)


class _TinyVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = LanguageModel(_tiny_text_config())
        self.vision_tower = _TinyVision()


def test_resolve_quant_mode_accepts_all4():
    prev = os.environ.pop(QUANT_MODE_ENV, None)
    try:
        assert resolve_quant_mode() == "none"
        assert resolve_quant_mode(quantize_all=True) == "all4"
        assert resolve_quant_mode(quantize_lm=True) == "lm4"
        try:
            resolve_quant_mode(quantize_lm=True, quantize_all=True)
        except ValueError as exc:
            assert "exclusive" in str(exc)
        else:
            raise AssertionError("expected ValueError for exclusive kwargs")
        os.environ[QUANT_MODE_ENV] = "all4"
        assert resolve_quant_mode() == "all4"
        assert resolve_quant_mode(quantize_lm=True) == "all4"
    finally:
        if prev is None:
            os.environ.pop(QUANT_MODE_ENV, None)
        else:
            os.environ[QUANT_MODE_ENV] = prev


def test_last_dim_packable_and_predicate():
    pack = nn.Linear(64, 32)
    skip = nn.Linear(16, 32)
    assert last_dim_packable(pack)
    assert not last_dim_packable(skip)
    spec = all4_predicate("vision_tower.fc1", pack)
    assert spec["bits"] == 4 and spec["group_size"] == 64
    assert all4_predicate("vision_tower.linear_fc2", skip) is False
    assert all4_predicate("action_in_proj.encoder", pack) is False
    assert all4_predicate("lm_head", pack) != False


def test_quantize_vlm_all4_refuses_language_only_and_full_model():
    try:
        quantize_vlm_all4(LanguageModel(_tiny_text_config()))
    except ValueError as exc:
        assert "full VLM" in str(exc)
    else:
        raise AssertionError("expected ValueError for language-only")

    class FakeFull:
        vlm = object()
        expert = object()
        vision_tower = object()
        language_model = object()

    try:
        quantize_vlm_all4(FakeFull())
    except ValueError as exc:
        assert "not AlpamayoR1MLX" in str(exc)
    else:
        raise AssertionError("expected ValueError for full model")


def test_quantize_expert_all4_refuses_vlm():
    vlm = _TinyVLM()
    try:
        quantize_expert_all4(vlm)
    except ValueError as exc:
        assert "not the VLM" in str(exc)
    else:
        raise AssertionError("expected ValueError for VLM")


def test_quantize_tiny_vlm_packs_head_embed_and_skips_fc2():
    reset_quantized()
    vlm = _TinyVLM()
    mx.eval(vlm.parameters())
    install_compiled_prefill(vlm.language_model)
    summary = quantize_vlm_all4(vlm)
    assert summary["n_quantized_linear"] == 16  # 14 decoder + lm_head + vision fc1
    assert summary["n_quantized_embedding"] == 1
    assert isinstance(vlm.language_model.lm_head, nn.QuantizedLinear)
    assert isinstance(vlm.language_model.model.embed_tokens, nn.QuantizedEmbedding)
    assert isinstance(vlm.vision_tower.fc1, nn.QuantizedLinear)
    assert isinstance(vlm.vision_tower.linear_fc2, nn.Linear)
    assert any("linear_fc2" in p for p in summary["unpacked_last_dim_paths"])
    flags = quantized_flags()
    assert flags["lm"] == "affine-4-gs64"
    assert flags["vision"] == "affine-4-gs64"


def test_quantize_tiny_expert():
    reset_quantized()
    expert = AlpamayoExpert(_tiny_text_config())
    mx.eval(expert.parameters())
    summary = quantize_expert_all4(expert)
    assert summary["n_quantized_linear"] == 14
    assert isinstance(expert.language_model.model.layers[0].self_attn.q_proj, nn.QuantizedLinear)
    assert quantized_flags()["expert"] == "affine-4-gs64"


def test_all4_save_load_roundtrip(tmp_path):
    reset_quantized()
    src_vlm = _TinyVLM()
    mx.eval(src_vlm.parameters())
    install_compiled_prefill(src_vlm.language_model)
    quantize_vlm_all4(src_vlm)
    dest = str(tmp_path / "mlx_all4")
    vlm_saved = save_vlm_all4(src_vlm, dest)
    src_ex = AlpamayoExpert(_tiny_text_config())
    mx.eval(src_ex.parameters())
    quantize_expert_all4(src_ex)
    save_expert_all4(src_ex, dest, vlm_saved)
    assert all4_checkpoint_ready(dest) is True

    reset_quantized()
    dst_vlm = _TinyVLM()
    mx.eval(dst_vlm.parameters())
    install_compiled_prefill(dst_vlm.language_model)
    loaded_v = load_vlm_all4(dst_vlm, dest)
    assert loaded_v["source"] == "disk"
    assert loaded_v["n_quantized_linear"] == 16
    src_w = src_vlm.vision_tower.fc1.weight
    dst_w = dst_vlm.vision_tower.fc1.weight
    assert bool((src_w == dst_w).all())
    assert bool((src_vlm.vision_tower.linear_fc2.weight == dst_vlm.vision_tower.linear_fc2.weight).all())

    dst_ex = AlpamayoExpert(_tiny_text_config())
    mx.eval(dst_ex.parameters())
    loaded_e = load_expert_all4(dst_ex, dest)
    assert loaded_e["n_quantized_linear"] == 14
    src_eq = src_ex.language_model.model.layers[0].self_attn.q_proj.weight
    dst_eq = dst_ex.language_model.model.layers[0].self_attn.q_proj.weight
    assert bool((src_eq == dst_eq).all())


def test_apply_all4_live_packs_then_loads(tmp_path):
    dest = str(tmp_path / "mlx_all4")
    reset_quantized()
    v1 = _TinyVLM()
    mx.eval(v1.parameters())
    install_compiled_prefill(v1.language_model)
    out_v = apply_vlm_all4(v1, dest)
    assert out_v["source"] == "live-pack"
    e1 = AlpamayoExpert(_tiny_text_config())
    mx.eval(e1.parameters())
    out_e = apply_expert_all4(e1, dest, out_v)
    assert out_e["source"] == "live-pack"
    assert all4_checkpoint_ready(dest)

    reset_quantized()
    v2 = _TinyVLM()
    mx.eval(v2.parameters())
    install_compiled_prefill(v2.language_model)
    again_v = apply_vlm_all4(v2, dest)
    assert again_v["source"] == "disk"
    e2 = AlpamayoExpert(_tiny_text_config())
    mx.eval(e2.parameters())
    again_e = apply_expert_all4(e2, dest, again_v)
    assert again_e["source"] == "disk"


def test_all4_checkpoint_ready_rejects_incomplete(tmp_path):
    dest = tmp_path / "mlx_all4"
    dest.mkdir()
    assert all4_checkpoint_ready(str(dest)) is False
    (dest / ALL4_VLM_WEIGHTS_NAME).write_bytes(b"x")
    try:
        all4_checkpoint_ready(str(dest))
    except FileNotFoundError as exc:
        assert "incomplete" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for vlm without expert/config")


def test_resolve_all4_dir_kwarg_env_and_default(tmp_path):
    default = resolve_all4_dir("/ckpt/alpamayo")
    assert default.endswith(f"/ckpt/alpamayo/{DEFAULT_ALL4_DIRNAME}")
    assert resolve_all4_dir("/ckpt/alpamayo", str(tmp_path / "custom")) == str(
        (tmp_path / "custom").resolve()
    )
    prev = os.environ.pop(ALL4_DIR_ENV, None)
    try:
        os.environ[ALL4_DIR_ENV] = str(tmp_path / "envdir")
        assert resolve_all4_dir("/ckpt/alpamayo") == str((tmp_path / "envdir").resolve())
    finally:
        if prev is None:
            os.environ.pop(ALL4_DIR_ENV, None)
        else:
            os.environ[ALL4_DIR_ENV] = prev
