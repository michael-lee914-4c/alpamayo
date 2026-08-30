"""T2.2: compiled prefill is invoked by layer(...) and matches eager."""

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_vlm.models.qwen3_vl.config import TextConfig
from mlx_vlm.models.qwen3_vl.language import Qwen3VLDecoderLayer, Qwen3VLModel

from mlx_port.models.compiled_backbone import (
    CompiledPrefillLayer,
    install_compiled_prefill,
    prefill_compile_captures,
    reset_prefill_compile_captures,
    wrap_decoder_layer_prefill,
)
from mlx_port.stage_timers import compiled_flags, reset_compiled


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


def _hidden_and_pos(seq: int, hidden: int = 64):
    h = mx.random.normal((1, seq, hidden)).astype(mx.bfloat16)
    pos = mx.broadcast_to(mx.arange(seq).reshape(1, 1, seq), (3, 1, seq))
    return h, pos


def test_layer_call_uses_compiled_prefill_class():
    """Qwen3VLModel calls layer(...); that must hit CompiledPrefillLayer.__call__."""
    inner = Qwen3VLDecoderLayer(_tiny_text_config(), 0)
    wrapped = wrap_decoder_layer_prefill(inner)
    assert isinstance(wrapped, CompiledPrefillLayer)
    assert type(wrapped).__call__ is CompiledPrefillLayer.__call__
    assert type(inner).__call__ is Qwen3VLDecoderLayer.__call__


def test_compiled_prefill_matches_eager_and_fills_cache():
    mx.random.seed(0)
    reset_prefill_compile_captures()
    cfg = _tiny_text_config()
    eager = Qwen3VLDecoderLayer(cfg, 0)
    inner = Qwen3VLDecoderLayer(cfg, 0)
    inner.update(eager.parameters())
    mx.eval(eager.parameters(), inner.parameters())
    compiled = wrap_decoder_layer_prefill(inner)

    h, pos = _hidden_and_pos(16)
    cache_e = KVCache()
    cache_c = KVCache()
    y_e = eager(h, "causal", cache_e, pos)
    y_c = compiled(h, "causal", cache_c, pos)
    mx.eval(y_e, y_c)

    assert tuple(y_c.shape) == tuple(y_e.shape)
    assert cache_c.offset == cache_e.offset == 16
    delta = float(mx.abs(y_e.astype(mx.float32) - y_c.astype(mx.float32)).max())
    assert delta == 0.0, f"compiled prefill drifted from eager: {delta}"
    assert prefill_compile_captures() == 1
    compiled(h, "causal", cache_c, pos)
    assert prefill_compile_captures() == 1


def test_wrap_decoder_layer_prefill_is_idempotent():
    layer = Qwen3VLDecoderLayer(_tiny_text_config(), 0)
    first = wrap_decoder_layer_prefill(layer)
    second = wrap_decoder_layer_prefill(first)
    assert first is second
    assert isinstance(first, CompiledPrefillLayer)


def test_decode_after_compiled_prefill_appends_kv():
    mx.random.seed(1)
    cfg = _tiny_text_config()
    layer = wrap_decoder_layer_prefill(Qwen3VLDecoderLayer(cfg, 0))
    mx.eval(layer.parameters())
    cache = KVCache()
    h, pos = _hidden_and_pos(8)
    mx.eval(layer(h, "causal", cache, pos))
    assert cache.offset == 8

    h1, pos1 = _hidden_and_pos(1)
    pos1 = pos1 + 8
    mx.eval(layer(h1, None, cache, pos1))
    assert cache.offset == 9


def test_qwen3vl_model_loop_hits_compiled_wrapper():
    """The stock loop is `layer(h, mask, c, pos)`, not layer.__call__ = ..."""
    reset_prefill_compile_captures()
    reset_compiled()
    try:
        model = Qwen3VLModel(_tiny_text_config())
        mx.eval(model.parameters())
        lm = type("LM", (), {"model": model})()
        assert install_compiled_prefill(lm) == 2
        assert all(isinstance(layer, CompiledPrefillLayer) for layer in model.layers)

        seq = 8
        h, pos = _hidden_and_pos(seq)
        cache = [KVCache(), KVCache()]
        out = model(
            mx.zeros((1, seq), dtype=mx.int32),
            inputs_embeds=h,
            mask="causal",
            cache=cache,
            position_ids=pos,
        )
        mx.eval(out)
        assert prefill_compile_captures() == 2
        assert cache[0].offset == seq
        assert cache[1].offset == seq
    finally:
        reset_compiled()
        reset_prefill_compile_captures()


def test_install_compiled_prefill_wraps_all_layers_and_sets_flag():
    reset_compiled()
    try:
        model = Qwen3VLModel(_tiny_text_config())
        mx.eval(model.parameters())
        lm = type("LM", (), {"model": model})()
        n = install_compiled_prefill(lm)
        assert n == 2
        assert all(isinstance(layer, CompiledPrefillLayer) for layer in model.layers)
        assert compiled_flags()["prefill"] is True
        assert install_compiled_prefill(lm) == 0
    finally:
        reset_compiled()


def test_install_compiled_prefill_rejects_missing_model():
    try:
        install_compiled_prefill(None)
    except ValueError as exc:
        assert "language model" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing model")
