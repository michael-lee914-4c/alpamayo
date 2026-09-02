"""T4.1 QLoRA: wrap decoder + vision 27-block/merger/deepstack; packed ints stay frozen."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

from mlx_port.lora import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    DENSE_WEIGHTS_NAME,
    LORA_LEAVES,
    VISION_BLOCK_LEAVES,
    VISION_MERGER_LEAVES,
    assert_only_lora_trainable,
    decoder_layer_inner,
    freeze_expert_base_unfreeze_lora,
    freeze_vision_features,
    has_expert_lora,
    has_vision_lora,
    inject_backbone_lora,
    inject_expert_lora,
    inject_vision_lora,
    load_lora_adapters,
    lora_save_steps,
    packed_weight_fingerprint,
    save_dense_trainables,
    save_lora_adapters,
    sft_lora_update,
)
from mlx_port.models.compiled_backbone import (
    CompiledPrefillLayer,
    uninstall_compiled_prefill,
)


class _Attn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)


class _Mlp(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, d * 2, bias=False)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)


class _Layer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _Mlp(d)

    def __call__(self, x):
        a = self.self_attn.q_proj(x)
        a = a + self.self_attn.k_proj(x)
        a = a + self.self_attn.v_proj(x)
        a = self.self_attn.o_proj(a)
        m = self.mlp.gate_proj(x) * self.mlp.up_proj(x)
        return a + self.mlp.down_proj(m)


class _InnerLM(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.layers = [_Layer(d) for _ in range(n)]


class _LM(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.model = _InnerLM(n, d)


class TinyVLM(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 32, n: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.language_model = _LM(n, d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.vision_tower = nn.Linear(d, d, bias=False)

    def __call__(self, input_ids, cache=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.language_model.model.layers:
            h = decoder_layer_inner(layer)(h)
        return SimpleNamespace(logits=self.lm_head(h))


class TinyHost(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 32, n: int = 2):
        super().__init__()
        self.vlm = TinyVLM(vocab=vocab, d=d, n=n)
        self.expert = nn.Linear(d, d, bias=False)


class TinyExpert(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.language_model = _LM(n, d)

    @property
    def layers(self):
        return self.language_model.model.layers

    def __call__(self, inputs_embeds, position_ids=None, cache=None, mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = decoder_layer_inner(layer)(x)
        return x


class TinyStage2Host(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 32, n_vlm: int = 2, n_expert: int = 2):
        super().__init__()
        self.vlm = TinyVLM(vocab=vocab, d=d, n=n_vlm)
        self.expert = TinyExpert(d=d, n=n_expert)
        self.action_in_proj = nn.Linear(d, d, bias=False)
        self.action_out_proj = nn.Linear(d, 2, bias=False)


class _VAttn(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3, bias=True)
        self.proj = nn.Linear(d, d)

    def __call__(self, x):
        qkv = self.qkv(x)
        return self.proj(qkv[..., : x.shape[-1]])


class _VMlp(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(d, hidden, bias=True)
        self.linear_fc2 = nn.Linear(hidden, d, bias=True)

    def __call__(self, x):
        return self.linear_fc2(self.linear_fc1(x))


class _VBlock(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.attn = _VAttn(d)
        self.mlp = _VMlp(d, hidden)

    def __call__(self, x):
        return self.attn(x) + self.mlp(x)


class _Merger(nn.Module):
    def __init__(self, d: int, out: int):
        super().__init__()
        self.linear_fc1 = nn.Linear(d, d)
        self.linear_fc2 = nn.Linear(d, out)

    def __call__(self, x):
        return self.linear_fc2(self.linear_fc1(x))


class TinyVisionTower(nn.Module):
    def __init__(
        self,
        n_blocks: int = 2,
        d: int = 16,
        hidden: int = 32,
        n_deep: int = 3,
        out: int = 16,
    ):
        super().__init__()
        self.patch_embed = SimpleNamespace(proj=nn.Linear(d, d, bias=False))
        self.blocks = [_VBlock(d, hidden) for _ in range(n_blocks)]
        self.merger = _Merger(d, out)
        self.deepstack_merger_list = [_Merger(d, out) for _ in range(n_deep)]

    def __call__(self, pixels, grid=None):
        x = mx.array(pixels)
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])
        for block in self.blocks:
            x = block(x)
        hidden = self.merger(x)
        deepstack = [mer(x) for mer in self.deepstack_merger_list]
        return hidden, deepstack


class TinyVisionVLM(TinyVLM):
    def __init__(
        self,
        vocab: int = 16,
        d: int = 16,
        n: int = 2,
        n_blocks: int = 2,
        n_deep: int = 3,
    ):
        super().__init__(vocab=vocab, d=d, n=n)
        self.vision_tower = TinyVisionTower(
            n_blocks=n_blocks, d=d, hidden=d * 2, n_deep=n_deep, out=d
        )

    def __call__(self, input_ids, cache=None, **kwargs):
        h = self.embed(input_ids)
        pixels = kwargs.get("pixel_values")
        if pixels is not None:
            hidden, deepstack = self.vision_tower(pixels, kwargs.get("image_grid_thw"))
            h = h + hidden.mean()
            for feat in deepstack:
                h = h + feat.mean()
        for layer in self.language_model.model.layers:
            h = decoder_layer_inner(layer)(h)
        return SimpleNamespace(logits=self.lm_head(h))


class TinyVisionHost(nn.Module):
    def __init__(
        self,
        vocab: int = 16,
        d: int = 16,
        n: int = 2,
        n_blocks: int = 2,
        n_deep: int = 3,
    ):
        super().__init__()
        self.vlm = TinyVisionVLM(
            vocab=vocab, d=d, n=n, n_blocks=n_blocks, n_deep=n_deep
        )
        self.expert = nn.Linear(d, d, bias=False)


def _quantize_leaf(mod: nn.Module, name: str) -> None:
    leaf = getattr(mod, name)
    if not isinstance(leaf, nn.Linear):
        raise RuntimeError(f"{name} is {type(leaf).__name__}, not Linear")
    setattr(mod, name, nn.QuantizedLinear.from_linear(leaf, group_size=32, bits=4))


def _quantized_host(n: int = 2, d: int = 32) -> TinyHost:
    host = TinyHost(n=n, d=d)
    for layer in host.vlm.language_model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _quantize_leaf(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _quantize_leaf(layer.mlp, name)
    return host


def test_inject_wraps_seven_leaves_per_layer_and_skips_expert():
    host = TinyHost(n=2, d=32)
    info = inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    assert info["n_wrapped"] == 14
    assert info["n_trainable"] == 28
    for layer in host.vlm.language_model.model.layers:
        inner = decoder_layer_inner(layer)
        names = []
        for path, mod in inner.named_modules():
            leaf = path.split(".")[-1]
            if leaf in LORA_LEAVES:
                assert isinstance(mod, LoRALinear), path
                names.append(leaf)
        assert tuple(sorted(names)) == tuple(sorted(LORA_LEAVES))
    assert not isinstance(host.expert, LoRALinear)
    assert not isinstance(host.vlm.vision_tower, LoRALinear)
    assert not isinstance(host.vlm.lm_head, LoRALinear)
    assert_only_lora_trainable(host)


def test_inject_unwraps_compiled_prefill_layer():
    host = TinyHost(n=2, d=32)
    layers = host.vlm.language_model.model.layers
    layers[0] = CompiledPrefillLayer(layers[0])
    assert uninstall_compiled_prefill(host.vlm.language_model) == 1
    assert not isinstance(layers[0], CompiledPrefillLayer)
    layers[0] = CompiledPrefillLayer(layers[0])
    info = inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    assert info["n_wrapped"] == 14
    assert not isinstance(layers[0], CompiledPrefillLayer)
    assert isinstance(layers[0].self_attn.q_proj, LoRALinear)


def test_inject_raises_on_wrong_layer_count_and_missing_leaf():
    host = TinyHost(n=2, d=32)
    try:
        inject_backbone_lora(host, expected_layers=36, vision=False)
    except RuntimeError as exc:
        assert "36" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for 36 vs 2 layers")

    host2 = TinyHost(n=1, d=32)
    del host2.vlm.language_model.model.layers[0].self_attn.o_proj
    try:
        inject_backbone_lora(host2, expected_layers=1, vision=False)
    except RuntimeError as exc:
        assert "o_proj" in str(exc) or "7" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when a LoRA leaf is missing")


def test_inject_refuses_missing_vlm_and_double_wrap():
    try:
        inject_backbone_lora(nn.Linear(4, 4), expected_layers=1, vision=False)
    except ValueError as exc:
        assert "vlm" in str(exc)
    else:
        raise AssertionError("expected ValueError without model.vlm")

    host = TinyHost(n=1, d=32)
    inject_backbone_lora(host, rank=4, expected_layers=1, vision=False)
    try:
        inject_backbone_lora(host, rank=4, expected_layers=1, vision=False)
    except RuntimeError as exc:
        assert "already LoRALinear" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on second inject")


def test_packed_weight_unchanged_after_lora_step():
    mx.random.seed(0)
    host = _quantized_host(n=2, d=32)
    inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    fp0 = packed_weight_fingerprint(host)
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    opt = optim.Adam(learning_rate=1e-2)
    before = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    loss = sft_lora_update(host, {"input_ids": ids}, opt, stage="stage1")
    fp1 = packed_weight_fingerprint(host)
    after = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    assert fp1 == fp0
    assert float(loss.loss) >= 0.0
    assert loss.times.fwd_bwd_ms > 0.0
    assert loss.times.adam_ms >= 0.0
    assert loss.times.n_vlm_forwards == 1
    moved = [k for k in before if not np.allclose(before[k], after[k], atol=0.0)]
    if not moved:
        raise AssertionError("LoRA A/B did not change after an optimizer step")
    assert all("lora_" in k for k in moved)
    assert_only_lora_trainable(host)


def test_packed_fingerprint_raises_without_quantized_linear():
    host = TinyHost(n=1, d=32)
    try:
        packed_weight_fingerprint(host)
    except RuntimeError as exc:
        assert "QuantizedLinear" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when nothing is packed")


def test_freeze_vision_features_is_noop_without_pixels():
    host = TinyHost(n=1, d=32)
    batch = {"input_ids": mx.array([[1, 2, 3]], dtype=mx.int32)}
    out = freeze_vision_features(host, batch)
    assert out is batch or out["input_ids"] is batch["input_ids"]
    assert "cached_image_features" not in out


def test_time_train_step_lora_flags_in_help():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.time_train_step", "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    for flag in (
        "--lora",
        "--lora-steps",
        "--lora-rank",
        "--lora-vision",
        "--lora-save-dir",
        "--lora-save-every",
        "--no-lora-save",
    ):
        if flag not in proc.stdout:
            raise AssertionError(f"{flag} missing from help")
    if "merger" not in proc.stdout.lower() or "deepstack" not in proc.stdout.lower():
        raise AssertionError("--lora-vision help must mention merger + deepstack")


def test_time_train_step_lora_requires_stage1():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--lora",
            "--stage",
            "joint",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --lora is not stage1")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "stage1" in err


def test_time_train_step_lora_vision_requires_lora():
    import subprocess
    import sys

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--lora-vision",
            "merger",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --lora-vision is set without --lora")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "--lora" in err


def _expected_vision_wraps(n_blocks: int, n_deep: int, scope: str = "full") -> int:
    n_merger = len(VISION_MERGER_LEAVES) * (1 + n_deep)
    if scope == "merger":
        return n_merger
    return n_blocks * len(VISION_BLOCK_LEAVES) + n_merger


def test_inject_vision_wraps_blocks_merger_deepstack():
    host = TinyVisionHost(n=2, d=16, n_blocks=2, n_deep=3)
    info = inject_vision_lora(
        host, rank=4, expected_blocks=2, expected_deepstack=3
    )
    expect = _expected_vision_wraps(2, 3)
    assert info["n_wrapped"] == expect
    assert info["n_trainable"] == expect * 2
    tower = host.vlm.vision_tower
    for i, block in enumerate(tower.blocks):
        names = []
        for path, mod in block.named_modules():
            leaf = path.split(".")[-1]
            if leaf in VISION_BLOCK_LEAVES:
                assert isinstance(mod, LoRALinear), path
                names.append(leaf)
        assert tuple(sorted(set(names))) == tuple(sorted(VISION_BLOCK_LEAVES))
    assert isinstance(tower.merger.linear_fc1, LoRALinear)
    assert isinstance(tower.merger.linear_fc2, LoRALinear)
    for mer in tower.deepstack_merger_list:
        assert isinstance(mer.linear_fc1, LoRALinear)
        assert isinstance(mer.linear_fc2, LoRALinear)
    assert not isinstance(tower.patch_embed.proj, LoRALinear)
    assert not isinstance(host.expert, LoRALinear)
    assert not isinstance(host.vlm.lm_head, LoRALinear)
    assert has_vision_lora(host)
    assert_only_lora_trainable(host)


def test_inject_backbone_lora_wraps_decoder_and_vision():
    host = TinyVisionHost(n=2, d=16, n_blocks=2, n_deep=3)
    info = inject_backbone_lora(
        host,
        rank=4,
        expected_layers=2,
        expected_vision_blocks=2,
        expected_deepstack=3,
    )
    assert info["n_decoder_wrapped"] == 14
    assert info["n_vision_wrapped"] == _expected_vision_wraps(2, 3)
    assert info["n_wrapped"] == 14 + info["n_vision_wrapped"]
    assert has_vision_lora(host)
    assert isinstance(host.vlm.language_model.model.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(host.vlm.vision_tower.blocks[0].attn.qkv, LoRALinear)
    assert not isinstance(host.expert, LoRALinear)
    assert info["vision_scope"] == "full"


def test_inject_vision_merger_scope_skips_blocks():
    host = TinyVisionHost(n=2, d=16, n_blocks=2, n_deep=3)
    info = inject_vision_lora(
        host, rank=4, expected_blocks=2, expected_deepstack=3, scope="merger"
    )
    expect = _expected_vision_wraps(2, 3, scope="merger")
    assert expect == 8
    assert info["n_wrapped"] == expect
    assert info["n_trainable"] == expect * 2
    assert info["vision_scope"] == "merger"
    tower = host.vlm.vision_tower
    for block in tower.blocks:
        assert not isinstance(block.attn.qkv, LoRALinear)
        assert not isinstance(block.attn.proj, LoRALinear)
        assert not isinstance(block.mlp.linear_fc1, LoRALinear)
        assert not isinstance(block.mlp.linear_fc2, LoRALinear)
    assert isinstance(tower.merger.linear_fc1, LoRALinear)
    assert isinstance(tower.merger.linear_fc2, LoRALinear)
    for mer in tower.deepstack_merger_list:
        assert isinstance(mer.linear_fc1, LoRALinear)
        assert isinstance(mer.linear_fc2, LoRALinear)
    assert has_vision_lora(host)
    assert_only_lora_trainable(host)


def test_inject_backbone_lora_vision_merger_scope():
    host = TinyVisionHost(n=2, d=16, n_blocks=2, n_deep=3)
    info = inject_backbone_lora(
        host,
        rank=4,
        expected_layers=2,
        expected_vision_blocks=2,
        expected_deepstack=3,
        vision_scope="merger",
    )
    assert info["n_decoder_wrapped"] == 14
    assert info["n_vision_wrapped"] == 8
    assert info["n_wrapped"] == 22
    assert info["vision_scope"] == "merger"
    assert isinstance(host.vlm.language_model.model.layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(host.vlm.vision_tower.blocks[0].attn.qkv, LoRALinear)
    assert isinstance(host.vlm.vision_tower.merger.linear_fc1, LoRALinear)
    assert isinstance(host.vlm.vision_tower.deepstack_merger_list[0].linear_fc2, LoRALinear)


def test_inject_vision_raises_on_bad_scope():
    host = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=3)
    try:
        inject_vision_lora(host, expected_blocks=2, expected_deepstack=3, scope="blocks")
    except ValueError as exc:
        assert "merger" in str(exc) or "full" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown vision scope")


def test_inject_vision_raises_on_wrong_counts_and_double_wrap():
    host = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=3)
    try:
        inject_vision_lora(host, expected_blocks=27, expected_deepstack=3)
    except RuntimeError as exc:
        assert "27" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for 27 vs 2 vision blocks")

    host2 = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=2)
    try:
        inject_vision_lora(host2, expected_blocks=2, expected_deepstack=3)
    except RuntimeError as exc:
        assert "deepstack" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for deepstack count")

    host3 = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=3)
    inject_vision_lora(host3, rank=4, expected_blocks=2, expected_deepstack=3)
    try:
        inject_vision_lora(host3, rank=4, expected_blocks=2, expected_deepstack=3)
    except RuntimeError as exc:
        assert "already LoRALinear" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on second vision inject")


def test_inject_backbone_lora_raises_without_vision_blocks():
    host = TinyHost(n=1, d=32)
    try:
        inject_backbone_lora(host, expected_layers=1, vision=True)
    except ValueError as exc:
        assert "blocks" in str(exc)
    else:
        raise AssertionError("expected ValueError when vision_tower has no blocks")


def test_freeze_vision_features_raises_when_vision_lora():
    host = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=3)
    inject_vision_lora(host, rank=4, expected_blocks=2, expected_deepstack=3)
    batch = {
        "pixel_values": mx.random.normal((4, 16)),
        "image_grid_thw": mx.array([[1, 2, 2]], dtype=mx.int32),
    }
    try:
        freeze_vision_features(host, batch)
    except RuntimeError as exc:
        assert "vision LoRA" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when stop-grad would kill vision LoRA")


def test_sft_lora_update_trains_vision_adapters_and_keeps_encode_on_tape():
    mx.random.seed(1)
    host = TinyVisionHost(n=2, d=16, n_blocks=2, n_deep=3)
    inject_backbone_lora(
        host,
        rank=4,
        expected_layers=2,
        expected_vision_blocks=2,
        expected_deepstack=3,
        vision_scope="merger",
    )
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    pixels = mx.random.normal((4, 16))
    batch = {
        "input_ids": ids,
        "pixel_values": pixels,
        "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
    }
    opt = optim.Adam(learning_rate=1e-2)
    before = {
        k: np.array(v)
        for k, v in tree_flatten(host.trainable_parameters())
        if "vision_tower" in k
    }
    if not before:
        raise AssertionError("no trainable vision LoRA parameters")
    if any("blocks" in k for k in before):
        raise AssertionError("merger scope must not train vision blocks")
    block_qkv = np.array(host.vlm.vision_tower.blocks[0].attn.qkv.weight)
    loss = sft_lora_update(host, batch, opt, stage="stage1")
    after = {
        k: np.array(v)
        for k, v in tree_flatten(host.trainable_parameters())
        if "vision_tower" in k
    }
    assert float(loss.loss) >= 0.0
    assert "cached_image_features" not in batch
    moved = [k for k in before if not np.allclose(before[k], after[k], atol=0.0)]
    if not moved:
        raise AssertionError("vision LoRA A/B did not change; encode was off the tape")
    assert all("lora_" in k for k in moved)
    assert any("merger" in k or "deepstack" in k for k in moved)
    assert np.allclose(block_qkv, np.array(host.vlm.vision_tower.blocks[0].attn.qkv.weight))
    assert_only_lora_trainable(host)


def test_sft_lora_update_refuses_cached_features_with_vision_lora():
    host = TinyVisionHost(n=1, d=16, n_blocks=2, n_deep=3)
    inject_vision_lora(host, rank=4, expected_blocks=2, expected_deepstack=3)
    batch = {
        "input_ids": mx.array([[1, 2, 3, 4]], dtype=mx.int32),
        "pixel_values": mx.random.normal((4, 16)),
        "cached_image_features": mx.zeros((4, 16)),
    }
    opt = optim.Adam(learning_rate=1e-2)
    try:
        sft_lora_update(host, batch, opt, stage="stage1")
    except RuntimeError as exc:
        assert "cached" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for cached features + vision LoRA")


def test_packed_vision_weight_unchanged_after_lora_step():
    mx.random.seed(2)
    # last dim must be ÷ group_size 32 so QuantizedLinear.from_linear can pack
    host = TinyVisionHost(n=2, d=32, n_blocks=2, n_deep=3)
    tower = host.vlm.vision_tower
    for block in tower.blocks:
        _quantize_leaf(block.attn, "qkv")
        _quantize_leaf(block.attn, "proj")
        _quantize_leaf(block.mlp, "linear_fc1")
        _quantize_leaf(block.mlp, "linear_fc2")
    _quantize_leaf(tower.merger, "linear_fc1")
    _quantize_leaf(tower.merger, "linear_fc2")
    for mer in tower.deepstack_merger_list:
        _quantize_leaf(mer, "linear_fc1")
        _quantize_leaf(mer, "linear_fc2")
    for layer in host.vlm.language_model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _quantize_leaf(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _quantize_leaf(layer.mlp, name)
    inject_backbone_lora(
        host,
        rank=4,
        expected_layers=2,
        expected_vision_blocks=2,
        expected_deepstack=3,
    )
    fp0 = packed_weight_fingerprint(host)
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    opt = optim.Adam(learning_rate=1e-2)
    loss = sft_lora_update(
        host,
        {
            "input_ids": ids,
            "pixel_values": mx.random.normal((4, 32)),
            "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int32),
        },
        opt,
        stage="stage1",
    )
    fp1 = packed_weight_fingerprint(host)
    assert fp1 == fp0
    assert float(loss.loss) >= 0.0
    assert_only_lora_trainable(host)


def test_lora_save_steps_always_includes_last():
    assert lora_save_steps(10, 10) == [10]
    assert lora_save_steps(25, 10) == [10, 20, 25]
    assert lora_save_steps(7, 10) == [7]
    try:
        lora_save_steps(10, 0)
    except ValueError as exc:
        assert "save-every" in str(exc)
    else:
        raise AssertionError("expected ValueError for save-every=0")
    try:
        lora_save_steps(0, 10)
    except ValueError as exc:
        assert "n_steps" in str(exc)
    else:
        raise AssertionError("expected ValueError for n_steps=0")


def test_save_and_load_lora_roundtrip(tmp_path):
    mx.random.seed(3)
    trained = TinyHost(n=2, d=32)
    inject_backbone_lora(trained, rank=4, expected_layers=2, vision=False)
    opt = optim.Adam(learning_rate=1e-2)
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    sft_lora_update(trained, {"input_ids": ids}, opt, stage="stage1")
    before = {k: np.array(v) for k, v in tree_flatten(trained.trainable_parameters())}
    if all(np.allclose(v, 0) for k, v in before.items() if k.endswith("lora_b")):
        raise AssertionError("expected LoRA B to move after a train step")
    info = save_lora_adapters(
        trained,
        tmp_path,
        step=10,
        rank=4,
        scale=20.0,
        vision_scope="none",
    )
    assert (tmp_path / ADAPTER_WEIGHTS_NAME).is_file()
    assert (tmp_path / ADAPTER_CONFIG_NAME).is_file()
    names = {p.name for p in tmp_path.glob("*.safetensors")}
    if names != {ADAPTER_WEIGHTS_NAME}:
        raise AssertionError(f"must overwrite one file, found {names}")
    assert info["path"].endswith(ADAPTER_WEIGHTS_NAME)
    assert info["n_arrays"] == 28
    save_lora_adapters(
        trained,
        tmp_path,
        step=20,
        rank=4,
        scale=20.0,
        vision_scope="none",
    )
    names = {p.name for p in tmp_path.glob("*.safetensors")}
    if names != {ADAPTER_WEIGHTS_NAME}:
        raise AssertionError(f"second save must overwrite the same file, found {names}")

    fresh = TinyHost(n=2, d=32)
    inject_backbone_lora(fresh, rank=4, expected_layers=2, vision=False)
    zeros = {k: np.array(v) for k, v in tree_flatten(fresh.trainable_parameters())}
    if not all(np.allclose(v, 0) for k, v in zeros.items() if k.endswith("lora_b")):
        raise AssertionError("fresh LoRA B must start at zero")
    cfg = load_lora_adapters(fresh, tmp_path)
    assert cfg["step"] == 20
    assert cfg["rank"] == 4
    after = {k: np.array(v) for k, v in tree_flatten(fresh.trainable_parameters())}
    for key, val in before.items():
        if not np.allclose(val, after[key]):
            raise AssertionError(f"roundtrip mismatch on {key}")


def test_load_lora_raises_on_key_mismatch(tmp_path):
    host = TinyHost(n=2, d=32)
    inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    save_lora_adapters(
        host, tmp_path, step=1, rank=4, scale=20.0, vision_scope="none"
    )
    other = TinyHost(n=1, d=32)
    inject_backbone_lora(other, rank=4, expected_layers=1, vision=False)
    try:
        load_lora_adapters(other, tmp_path)
    except RuntimeError as exc:
        assert "key set" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on adapter key mismatch")


def test_save_lora_raises_without_adapters(tmp_path):
    host = TinyHost(n=1, d=16)
    try:
        save_lora_adapters(
            host, tmp_path, step=1, rank=4, scale=20.0, vision_scope="none"
        )
    except RuntimeError as exc:
        assert "trainable" in str(exc).lower() or "LoRA" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when no LoRA is injected")


def _quantize_expert_leaves(host: TinyStage2Host) -> None:
    for layer in host.expert.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _quantize_leaf(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _quantize_leaf(layer.mlp, name)


def test_inject_expert_lora_wraps_decoder_skips_action_and_vlm():
    host = TinyStage2Host(n_vlm=2, n_expert=2, d=32)
    assert not has_expert_lora(host)
    info = inject_expert_lora(host, rank=4, expected_layers=2)
    assert info["n_wrapped"] == 14
    assert info["n_trainable"] == 28
    assert has_expert_lora(host)
    for layer in host.expert.layers:
        names = []
        for path, mod in decoder_layer_inner(layer).named_modules():
            leaf = path.split(".")[-1]
            if leaf in LORA_LEAVES:
                assert isinstance(mod, LoRALinear), path
                names.append(leaf)
        assert tuple(sorted(names)) == tuple(sorted(LORA_LEAVES))
    assert not isinstance(host.action_in_proj, LoRALinear)
    assert not isinstance(host.action_out_proj, LoRALinear)
    assert not isinstance(host.vlm.language_model.model.layers[0].self_attn.q_proj, LoRALinear)
    assert not isinstance(host.vlm.lm_head, LoRALinear)
    flat = dict(tree_flatten(host.trainable_parameters()))
    expert_lora = [k for k in flat if k.startswith("expert.") and "lora_" in k]
    assert len(expert_lora) == 28
    assert not any(k.startswith("action_") for k in flat)
    assert any(k.startswith("vlm.") for k in flat)


def test_inject_backbone_lora_still_skips_expert_stack():
    host = TinyStage2Host(n_vlm=2, n_expert=2, d=32)
    inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    assert not has_expert_lora(host)
    assert isinstance(host.expert.layers[0].self_attn.q_proj, nn.Linear)
    assert isinstance(host.vlm.language_model.model.layers[0].self_attn.q_proj, LoRALinear)


def test_inject_expert_lora_raises_on_linear_expert_and_layer_count():
    host = TinyHost(n=2, d=32)
    try:
        inject_expert_lora(host, rank=4, expected_layers=2)
    except ValueError as exc:
        assert "expert" in str(exc)
    else:
        raise AssertionError("expected ValueError when expert has no decoder layers")

    host2 = TinyStage2Host(n_vlm=1, n_expert=2, d=32)
    try:
        inject_expert_lora(host2, rank=4, expected_layers=36)
    except RuntimeError as exc:
        assert "36" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for 36 vs 2 expert layers")


def test_inject_expert_lora_refuses_double_wrap():
    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    inject_expert_lora(host, rank=4, expected_layers=1)
    try:
        inject_expert_lora(host, rank=4, expected_layers=1)
    except RuntimeError as exc:
        assert "already LoRALinear" in str(exc)
    else:
        raise AssertionError("expected RuntimeError on second expert inject")


def test_packed_expert_fingerprint_unchanged_after_expert_lora_step():
    mx.random.seed(0)
    host = TinyStage2Host(n_vlm=1, n_expert=2, d=32)
    _quantize_expert_leaves(host)
    inject_expert_lora(host, rank=4, expected_layers=2)
    from mlx_port.train_step import freeze_vlm

    freeze_vlm(host)
    freeze_expert_base_unfreeze_lora(host)
    assert_only_lora_trainable(host)
    fp0 = packed_weight_fingerprint(host)
    before = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    action_in0 = np.array(host.action_in_proj.weight)
    opt = optim.Adam(learning_rate=1e-2)

    def loss_fn(m):
        x = mx.ones((1, 4, 32), dtype=mx.float32)
        return m.expert(x).mean()

    loss, grads = nn.value_and_grad(host, loss_fn)(host)
    opt.update(host, grads)
    mx.eval(loss, host.parameters(), opt.state)
    fp1 = packed_weight_fingerprint(host)
    after = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    assert fp1 == fp0
    moved = [k for k in before if not np.allclose(before[k], after[k], atol=0.0)]
    if not moved:
        raise AssertionError("expert LoRA A/B did not change after an optimizer step")
    assert all("lora_" in k and k.startswith("expert.") for k in moved)
    if not np.allclose(action_in0, np.array(host.action_in_proj.weight)):
        raise AssertionError("action_in_proj moved during expert LoRA step")


def test_save_load_expert_lora_excludes_vlm_keys(tmp_path):
    mx.random.seed(4)
    host = TinyStage2Host(n_vlm=2, n_expert=2, d=32)
    inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    inject_expert_lora(host, rank=4, expected_layers=2)
    from mlx_port.train_step import freeze_vlm

    freeze_vlm(host)
    freeze_expert_base_unfreeze_lora(host)
    assert_only_lora_trainable(host)
    opt = optim.Adam(learning_rate=1e-2)

    def loss_fn(m):
        x = mx.ones((1, 4, 32), dtype=mx.float32)
        return m.expert(x).mean()

    loss, grads = nn.value_and_grad(host, loss_fn)(host)
    opt.update(host, grads)
    mx.eval(loss, host.parameters())
    before = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    info = save_lora_adapters(
        host,
        tmp_path,
        step=1,
        rank=4,
        scale=20.0,
        vision_scope="none",
        extra={"target": "expert"},
    )
    assert info["n_arrays"] == 28
    saved = mx.load(str(tmp_path / ADAPTER_WEIGHTS_NAME))
    assert all(k.startswith("expert.") for k in saved)
    assert not any(k.startswith("vlm.") for k in saved)
    cfg = json.loads((tmp_path / ADAPTER_CONFIG_NAME).read_text())
    assert cfg["target"] == "expert"

    fresh = TinyStage2Host(n_vlm=2, n_expert=2, d=32)
    inject_expert_lora(fresh, rank=4, expected_layers=2)
    freeze_vlm(fresh)
    freeze_expert_base_unfreeze_lora(fresh)
    load_lora_adapters(fresh, tmp_path)
    after = {k: np.array(v) for k, v in tree_flatten(fresh.trainable_parameters())}
    for key, val in before.items():
        if not np.allclose(val, after[key]):
            raise AssertionError(f"expert adapter roundtrip mismatch on {key}")


def test_load_expert_adapters_reject_vlm_file(tmp_path):
    vlm_host = TinyHost(n=2, d=32)
    inject_backbone_lora(vlm_host, rank=4, expected_layers=2, vision=False)
    save_lora_adapters(
        vlm_host, tmp_path, step=1, rank=4, scale=20.0, vision_scope="none"
    )
    host = TinyStage2Host(n_vlm=2, n_expert=2, d=32)
    inject_expert_lora(host, rank=4, expected_layers=2)
    from mlx_port.train_step import freeze_vlm

    freeze_vlm(host)
    freeze_expert_base_unfreeze_lora(host)
    try:
        load_lora_adapters(host, tmp_path)
    except RuntimeError as exc:
        assert "key set" in str(exc)
    else:
        raise AssertionError("expected RuntimeError loading VLM adapters onto expert LoRA")


def test_expert_lora_train_action_proj_unfreezes_action_proj():
    host = TinyStage2Host(n_vlm=1, n_expert=2, d=32)
    inject_expert_lora(host, rank=4, expected_layers=2)
    from mlx_port.train_step import freeze_vlm

    freeze_vlm(host)
    freeze_expert_base_unfreeze_lora(host, train_action_proj=True)
    flat = dict(tree_flatten(host.trainable_parameters()))
    assert any(k.startswith("expert.") and "lora_" in k for k in flat)
    assert any(k.startswith("action_in_proj.") for k in flat)
    assert any(k.startswith("action_out_proj.") for k in flat)
    assert not any(k.startswith("vlm.") for k in flat)
    try:
        assert_only_lora_trainable(host)
    except RuntimeError as exc:
        assert "non-LoRA" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when action proj is trainable")


def test_save_expert_lora_with_dense_action_proj(tmp_path):
    mx.random.seed(5)
    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    inject_expert_lora(host, rank=4, expected_layers=1)
    from mlx_port.train_step import freeze_vlm

    freeze_vlm(host)
    freeze_expert_base_unfreeze_lora(host, train_action_proj=True)
    try:
        save_lora_adapters(
            host, tmp_path, step=1, rank=4, scale=20.0, vision_scope="none"
        )
    except RuntimeError as exc:
        assert "LoRA" in str(exc) or "trainable" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError saving LoRA while action proj trains")
    info = save_lora_adapters(
        host,
        tmp_path,
        step=1,
        rank=4,
        scale=20.0,
        vision_scope="none",
        extra={"train_action_proj": True},
        allow_extra_trainables=True,
    )
    assert info["n_arrays"] == 14
    saved = mx.load(str(tmp_path / ADAPTER_WEIGHTS_NAME))
    assert all("lora_" in k for k in saved)
    dense_info = save_dense_trainables(host, tmp_path, step=1)
    assert dense_info["n_arrays"] >= 2
    dense = mx.load(str(tmp_path / DENSE_WEIGHTS_NAME))
    assert any(k.startswith("action_in_proj.") for k in dense)
    assert any(k.startswith("action_out_proj.") for k in dense)
    assert not any("lora_" in k for k in dense)
