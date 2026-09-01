"""T4.1: QLoRA on the language decoder, vision tower, and (opt-in) expert decoder.

Language: walks ``vlm.language_model.model.layers`` (unwraps CompiledPrefillLayer)
and replaces q/k/v/o/gate/up/down with mlx_lm ``LoRALinear``.

Vision ``scope="full"``: every block ``qkv`` / ``proj`` / ``linear_fc1`` /
``linear_fc2`` (27×), plus ``merger`` and the 3 deepstack mergers.
``scope="merger"``: ``merger`` + 3 deepstack only (blocks stay frozen).
Conv3D ``patch_embed``, LayerNorms, ``pos_embed``, ``lm_head``, and
token embeddings are not wrapped.

Expert (``inject_expert_lora``): same 36×7 decoder leaves under
``expert.language_model.model.layers``. ``action_in_proj`` / ``action_out_proj``
stay dense (not LoRA-wrapped). After freeze, only expert ``lora_a`` /
``lora_b`` train unless Stage-2 ``train_action_proj`` also Adam-steps
action in/out. Packed ``QuantizedLinear.weight`` stays frozen.

When vision LoRA is present, encode stays on the tape. Language-only LoRA
still encodes once and ``stop_gradient``s.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.tuner.lora import LoRALinear

from mlx_port.models.compiled_backbone import uninstall_compiled_prefill
from mlx_port.train_step import (
    TrainStepTimes,
    TrainUpdateOutput,
    run_value_and_grad_update,
    sft_train_step,
)

LORA_LEAVES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
VISION_BLOCK_LEAVES = ("qkv", "proj", "linear_fc1", "linear_fc2")
VISION_MERGER_LEAVES = ("linear_fc1", "linear_fc2")
VISION_SCOPES = ("full", "merger")
EXPECTED_DECODER_LAYERS = 36
EXPECTED_VISION_BLOCKS = 27
EXPECTED_DEEPSTACK = 3
DEFAULT_RANK = 8
DEFAULT_SCALE = 20.0
DEFAULT_VISION_SCOPE = "full"
ADAPTER_WEIGHTS_NAME = "adapters.safetensors"
ADAPTER_CONFIG_NAME = "adapter_config.json"
DENSE_WEIGHTS_NAME = "dense.safetensors"
DEFAULT_SAVE_EVERY = 10


def decoder_layer_inner(layer: Any) -> Any:
    if getattr(layer, "_alpamayo_prefill_compiled", False):
        inner = getattr(layer, "inner", None)
        if inner is None:
            raise RuntimeError("CompiledPrefillLayer has no inner decoder layer")
        return inner
    return layer


def _language_layers(model: Any) -> list[Any]:
    if model is None:
        raise ValueError("inject_backbone_lora requires a model")
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        raise ValueError("inject_backbone_lora requires model.vlm")
    lm = getattr(vlm, "language_model", None)
    if lm is None or not hasattr(lm, "model"):
        raise ValueError("inject_backbone_lora requires vlm.language_model.model")
    layers = getattr(lm.model, "layers", None)
    if not layers:
        raise ValueError("language_model.model.layers is missing")
    return list(layers)


def _expert_layers(model: Any) -> list[Any]:
    expert = getattr(model, "expert", None)
    if expert is None:
        raise ValueError("inject_expert_lora requires model.expert")
    layers = getattr(expert, "layers", None)
    if layers:
        try:
            return list(layers)
        except TypeError as exc:
            raise ValueError(
                "inject_expert_lora: expert.layers is not a layer list"
            ) from exc
    lm = getattr(expert, "language_model", None)
    if lm is None or not hasattr(lm, "model"):
        raise ValueError(
            "inject_expert_lora requires expert.layers or "
            "expert.language_model.model.layers"
        )
    layers = getattr(lm.model, "layers", None)
    if not layers:
        raise ValueError("expert.language_model.model.layers is missing")
    return list(layers)


def _maybe_expert_layers(model: Any) -> list[Any]:
    expert = getattr(model, "expert", None)
    if expert is None:
        return []
    layers = getattr(expert, "layers", None)
    if layers:
        try:
            return list(layers)
        except TypeError:
            return []
    lm = getattr(expert, "language_model", None)
    if lm is None or not hasattr(lm, "model"):
        return []
    layers = getattr(lm.model, "layers", None)
    return list(layers) if layers else []


def _vision_tower(model: Any) -> Any:
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        raise ValueError("inject_vision_lora requires model.vlm")
    tower = getattr(vlm, "vision_tower", None)
    if tower is None:
        raise ValueError("inject_vision_lora requires vlm.vision_tower")
    return tower


def _wrap_module_leaves(
    module: nn.Module,
    leaves: tuple[str, ...],
    *,
    rank: int,
    scale: float,
    dropout: float,
    where: str,
) -> int:
    updates: list[tuple[str, nn.Module]] = []
    for path, mod in module.named_modules():
        leaf = path.split(".")[-1]
        if leaf not in leaves:
            continue
        if isinstance(mod, LoRALinear):
            raise RuntimeError(f"{where} {path} is already LoRALinear")
        if not isinstance(mod, (nn.Linear, nn.QuantizedLinear)):
            continue
        updates.append(
            (
                path,
                LoRALinear.from_base(mod, r=int(rank), scale=float(scale), dropout=dropout),
            )
        )
    module.update_modules(tree_unflatten(updates))
    return len(updates)


def resolve_vision_scope(scope: str) -> str:
    if scope not in VISION_SCOPES:
        raise ValueError(
            f"vision scope must be one of {VISION_SCOPES}, got {scope!r}"
        )
    return scope


def inject_vision_lora(
    model: Any,
    *,
    rank: int = DEFAULT_RANK,
    scale: float = DEFAULT_SCALE,
    dropout: float = 0.0,
    expected_blocks: int | None = EXPECTED_VISION_BLOCKS,
    expected_deepstack: int | None = EXPECTED_DEEPSTACK,
    freeze: bool = True,
    scope: str = DEFAULT_VISION_SCOPE,
) -> dict[str, int]:
    """Wrap vision LoRA leaves. ``scope='full'`` is 27 blocks + mergers; ``'merger'`` is merger + 3 deepstack."""
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    scope = resolve_vision_scope(scope)
    tower = _vision_tower(model)
    blocks = getattr(tower, "blocks", None)
    if not blocks:
        raise ValueError("vision_tower.blocks is missing")
    if expected_blocks is not None and len(blocks) != int(expected_blocks):
        raise RuntimeError(
            f"expected {expected_blocks} vision blocks, got {len(blocks)}"
        )
    merger = getattr(tower, "merger", None)
    if merger is None:
        raise ValueError("vision_tower.merger is missing")
    deepstack = getattr(tower, "deepstack_merger_list", None)
    if not deepstack:
        raise ValueError("vision_tower.deepstack_merger_list is missing")
    if expected_deepstack is not None and len(deepstack) != int(expected_deepstack):
        raise RuntimeError(
            f"expected {expected_deepstack} deepstack mergers, got {len(deepstack)}"
        )

    n_wrapped = 0
    if scope == "full":
        for i, block in enumerate(blocks):
            n = _wrap_module_leaves(
                block,
                VISION_BLOCK_LEAVES,
                rank=rank,
                scale=scale,
                dropout=dropout,
                where=f"vision block {i}",
            )
            if n != len(VISION_BLOCK_LEAVES):
                raise RuntimeError(
                    f"vision block {i} expected {len(VISION_BLOCK_LEAVES)} LoRA leaves "
                    f"{VISION_BLOCK_LEAVES}, got {n}"
                )
            n_wrapped += n

    n_merger = _wrap_module_leaves(
        merger,
        VISION_MERGER_LEAVES,
        rank=rank,
        scale=scale,
        dropout=dropout,
        where="vision merger",
    )
    if n_merger != len(VISION_MERGER_LEAVES):
        raise RuntimeError(
            f"vision merger expected {len(VISION_MERGER_LEAVES)} LoRA leaves, got {n_merger}"
        )
    n_wrapped += n_merger

    for j, mer in enumerate(deepstack):
        n = _wrap_module_leaves(
            mer,
            VISION_MERGER_LEAVES,
            rank=rank,
            scale=scale,
            dropout=dropout,
            where=f"deepstack merger {j}",
        )
        if n != len(VISION_MERGER_LEAVES):
            raise RuntimeError(
                f"deepstack merger {j} expected {len(VISION_MERGER_LEAVES)} "
                f"LoRA leaves, got {n}"
            )
        n_wrapped += n

    n_merger_leaves = len(VISION_MERGER_LEAVES) * (1 + len(deepstack))
    expect = n_merger_leaves
    if scope == "full":
        expect += len(blocks) * len(VISION_BLOCK_LEAVES)
    if n_wrapped != expect:
        raise RuntimeError(
            f"wrapped {n_wrapped} vision leaves, expected {expect} (scope={scope})"
        )

    patch = getattr(getattr(tower, "patch_embed", None), "proj", None)
    if isinstance(patch, LoRALinear):
        raise RuntimeError("patch_embed.proj must stay unwrapped (Conv3D / not a LoRA leaf)")

    if freeze:
        freeze_base_unfreeze_lora(model)
        n_train = count_trainable(model)
        if n_train != n_wrapped * 2:
            raise RuntimeError(
                f"expected {n_wrapped * 2} trainable arrays (lora_a/b), got {n_train}"
            )
        n_elem = count_trainable_elements(model)
        print(
            f"[LORA] wrapped {n_wrapped} vision leaves scope={scope} rank={rank} "
            f"trainable={n_train} arrays / {n_elem} elems (lora_a/b only)"
        )
        return {
            "n_wrapped": n_wrapped,
            "n_trainable": n_train,
            "n_elements": n_elem,
            "rank": int(rank),
            "vision_scope": scope,
        }
    return {"n_wrapped": n_wrapped, "rank": int(rank), "vision_scope": scope}


def inject_backbone_lora(
    model: Any,
    *,
    rank: int = DEFAULT_RANK,
    scale: float = DEFAULT_SCALE,
    dropout: float = 0.0,
    expected_layers: int | None = EXPECTED_DECODER_LAYERS,
    uninstall_compile: bool = True,
    vision: bool = True,
    vision_scope: str = DEFAULT_VISION_SCOPE,
    expected_vision_blocks: int | None = EXPECTED_VISION_BLOCKS,
    expected_deepstack: int | None = EXPECTED_DEEPSTACK,
) -> dict[str, int]:
    """Wrap decoder q/k/v/o/gate/up/down and, by default, the full vision LoRA set.

    ``uninstall_compile`` is on by default: compiled prefill + ``value_and_grad``
    at the PAI window (seq≈3000) OOMs. Infer compile stays on the infer path.
    ``vision=True`` + ``vision_scope='full'`` wraps 27 blocks + merger + 3 deepstack.
    ``vision_scope='merger'`` wraps merger + 3 deepstack only.
    """
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    if uninstall_compile:
        vlm = getattr(model, "vlm", None)
        lm = getattr(vlm, "language_model", None) if vlm is not None else None
        if lm is not None and hasattr(lm, "model"):
            n_un = uninstall_compiled_prefill(lm)
            if n_un:
                print(
                    f"[LORA] eager train: uninstalled {n_un} CompiledPrefillLayer "
                    "(compile+grad is infer-only)"
                )
    layers = _language_layers(model)
    if expected_layers is not None and len(layers) != int(expected_layers):
        raise RuntimeError(
            f"expected {expected_layers} decoder layers, got {len(layers)}"
        )

    n_lang = 0
    for i, layer in enumerate(layers):
        inner = decoder_layer_inner(layer)
        n = _wrap_module_leaves(
            inner,
            LORA_LEAVES,
            rank=rank,
            scale=scale,
            dropout=dropout,
            where=f"layer {i}",
        )
        if n != len(LORA_LEAVES):
            raise RuntimeError(
                f"layer {i} expected {len(LORA_LEAVES)} LoRA leaves "
                f"{LORA_LEAVES}, got {n}"
            )
        n_lang += n

    expect_lang = len(layers) * len(LORA_LEAVES)
    if n_lang != expect_lang:
        raise RuntimeError(f"wrapped {n_lang} decoder leaves, expected {expect_lang}")

    n_vis = 0
    scope = "none"
    if vision:
        scope = resolve_vision_scope(vision_scope)
        vis_info = inject_vision_lora(
            model,
            rank=rank,
            scale=scale,
            dropout=dropout,
            expected_blocks=expected_vision_blocks,
            expected_deepstack=expected_deepstack,
            freeze=False,
            scope=scope,
        )
        n_vis = int(vis_info["n_wrapped"])

    freeze_base_unfreeze_lora(model)
    n_wrapped = n_lang + n_vis
    n_train = count_trainable(model)
    if n_train != n_wrapped * 2:
        raise RuntimeError(
            f"expected {n_wrapped * 2} trainable arrays (lora_a/b), got {n_train}"
        )
    n_elem = count_trainable_elements(model)
    extra = f" + {n_vis} vision({scope})" if n_vis else ""
    print(
        f"[LORA] wrapped {n_lang} decoder{extra} leaves rank={rank} "
        f"trainable={n_train} arrays / {n_elem} elems (lora_a/b only)"
    )
    return {
        "n_wrapped": n_wrapped,
        "n_decoder_wrapped": n_lang,
        "n_vision_wrapped": n_vis,
        "n_trainable": n_train,
        "n_elements": n_elem,
        "rank": int(rank),
        "scale": float(scale),
        "vision_scope": scope,
    }


def has_expert_lora(model: Any) -> bool:
    expert = getattr(model, "expert", None)
    if expert is None or not hasattr(expert, "named_modules"):
        return False
    for _, mod in expert.named_modules():
        if isinstance(mod, LoRALinear):
            return True
    return False


def freeze_expert_base_unfreeze_lora(
    model: Any, *, train_action_proj: bool = False
) -> None:
    """Freeze packed expert; unfreeze expert LoRA A/B. Optionally Adam action in/out."""
    expert = getattr(model, "expert", None)
    if expert is None or not hasattr(expert, "freeze"):
        raise RuntimeError("freeze_expert_base_unfreeze_lora: model.expert is missing")
    expert.freeze()
    n = 0
    for _, mod in expert.named_modules():
        if isinstance(mod, LoRALinear):
            mod.unfreeze(keys=["lora_a", "lora_b"])
            if hasattr(mod, "linear") and mod.linear is not None:
                mod.linear.freeze()
            n += 1
    if n < 1:
        raise RuntimeError("freeze_expert_base_unfreeze_lora found no LoRALinear")
    for name in ("action_in_proj", "action_out_proj"):
        mod = getattr(model, name, None)
        if mod is None:
            if train_action_proj:
                raise RuntimeError(f"train_action_proj requires model.{name}")
            continue
        if not hasattr(mod, "freeze") or not hasattr(mod, "unfreeze"):
            raise RuntimeError(f"model.{name} has no freeze/unfreeze")
        if train_action_proj:
            mod.unfreeze()
        else:
            mod.freeze()


def unfreeze_action_proj(model: Any) -> None:
    """Adam-step leftover dense action in/out. Packed QuantizedLinear stays frozen."""
    freeze_expert_base_unfreeze_lora(model, train_action_proj=True)


def inject_expert_lora(
    model: Any,
    *,
    rank: int = DEFAULT_RANK,
    scale: float = DEFAULT_SCALE,
    dropout: float = 0.0,
    expected_layers: int | None = EXPECTED_DECODER_LAYERS,
    freeze: bool = True,
) -> dict[str, int]:
    """Wrap expert decoder q/k/v/o/gate/up/down. Action in/out stay dense.

    Does not call ``model.freeze()`` — VLM (and any Stage-1 LoRA) stay as they
    are. ``freeze=True`` freezes the expert subtree and action proj, then
    unfreezes expert ``lora_a`` / ``lora_b`` only.
    """
    if rank < 1:
        raise ValueError(f"LoRA rank must be >= 1, got {rank}")
    layers = _expert_layers(model)
    if expected_layers is not None and len(layers) != int(expected_layers):
        raise RuntimeError(
            f"expected {expected_layers} expert decoder layers, got {len(layers)}"
        )

    n_wrapped = 0
    for i, layer in enumerate(layers):
        inner = decoder_layer_inner(layer)
        n = _wrap_module_leaves(
            inner,
            LORA_LEAVES,
            rank=rank,
            scale=scale,
            dropout=dropout,
            where=f"expert layer {i}",
        )
        if n != len(LORA_LEAVES):
            raise RuntimeError(
                f"expert layer {i} expected {len(LORA_LEAVES)} LoRA leaves "
                f"{LORA_LEAVES}, got {n}"
            )
        n_wrapped += n

    expect = len(layers) * len(LORA_LEAVES)
    if n_wrapped != expect:
        raise RuntimeError(
            f"wrapped {n_wrapped} expert decoder leaves, expected {expect}"
        )

    for name in ("action_in_proj", "action_out_proj"):
        mod = getattr(model, name, None)
        if isinstance(mod, LoRALinear):
            raise RuntimeError(f"{name} must stay unwrapped (not a decoder leaf)")
        if mod is not None and hasattr(mod, "named_modules"):
            for path, inner in mod.named_modules():
                if isinstance(inner, LoRALinear):
                    raise RuntimeError(
                        f"{name}.{path} must stay unwrapped (not a decoder leaf)"
                    )

    if freeze:
        freeze_expert_base_unfreeze_lora(model)
        n_train = 0
        n_elem = 0
        for key, val in tree_flatten(model.trainable_parameters()):
            if not key.startswith("expert."):
                continue
            if "lora_a" not in key and "lora_b" not in key:
                raise RuntimeError(
                    f"expert trainable set must be LoRA A/B only; got {key}"
                )
            n_train += 1
            n_elem += int(val.size)
        if n_train != n_wrapped * 2:
            raise RuntimeError(
                f"expected {n_wrapped * 2} expert LoRA arrays, got {n_train}"
            )
        print(
            f"[LORA] wrapped {n_wrapped} expert decoder leaves rank={rank} "
            f"trainable={n_train} arrays / {n_elem} elems (lora_a/b only; "
            "action in/out dense + frozen)"
        )
        return {
            "n_wrapped": n_wrapped,
            "n_trainable": n_train,
            "n_elements": n_elem,
            "rank": int(rank),
            "scale": float(scale),
        }
    return {
        "n_wrapped": n_wrapped,
        "rank": int(rank),
        "scale": float(scale),
    }


def freeze_base_unfreeze_lora(model: Any) -> None:
    """Freeze every parameter, then unfreeze only LoRA A/B."""
    model.freeze()
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.unfreeze(keys=["lora_a", "lora_b"])
            if hasattr(mod, "linear") and mod.linear is not None:
                mod.linear.freeze()
            n += 1
    if n < 1:
        raise RuntimeError("freeze_base_unfreeze_lora found no LoRALinear")
    assert_only_lora_trainable(model)


def assert_only_lora_trainable(model: Any) -> None:
    flat = dict(tree_flatten(model.trainable_parameters()))
    if not flat:
        raise RuntimeError("no trainable parameters")
    bad = [k for k in flat if "lora_a" not in k and "lora_b" not in k]
    if bad:
        raise RuntimeError(
            "trainable parameters must be LoRA A/B only; "
            f"got non-LoRA keys {bad[:8]}"
        )


def count_trainable(model: Any) -> int:
    return len(tree_flatten(model.trainable_parameters()))


def count_trainable_elements(model: Any) -> int:
    return sum(int(v.size) for _, v in tree_flatten(model.trainable_parameters()))


def lora_adapter_weights(model: Any) -> dict[str, mx.array]:
    """Trainable LoRA A/B only. Ignores leftover dense trainables (action proj)."""
    weights = {
        k: v
        for k, v in tree_flatten(model.trainable_parameters())
        if "lora_a" in k or "lora_b" in k
    }
    if not weights:
        raise RuntimeError("no LoRA parameters to save")
    return weights


def dense_trainable_weights(model: Any) -> dict[str, mx.array]:
    """Trainable non-LoRA arrays (action in/out). Raises if the set is empty."""
    weights = {
        k: v
        for k, v in tree_flatten(model.trainable_parameters())
        if "lora_a" not in k and "lora_b" not in k
    }
    if not weights:
        raise RuntimeError("no dense trainable parameters to save")
    return weights


def lora_trainable_weights(model: Any) -> dict[str, mx.array]:
    """LoRA A/B only. Raises if the trainable set is empty or includes a base weight."""
    assert_only_lora_trainable(model)
    return lora_adapter_weights(model)


def lora_save_steps(n_steps: int, every: int) -> list[int]:
    """1-based completed-step indices that overwrite the same adapter file.

    Always includes the last step so a short run still writes once.
    """
    if int(n_steps) < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if int(every) < 1:
        raise ValueError(f"save-every must be >= 1, got {every}")
    n_steps = int(n_steps)
    every = int(every)
    steps = [i for i in range(every, n_steps + 1, every)]
    if n_steps not in steps:
        steps.append(n_steps)
    return steps


def save_dense_trainables(
    model: Any,
    directory: str | Path,
    *,
    step: int,
) -> dict[str, Any]:
    """Overwrite ``dense.safetensors`` with trainable non-LoRA arrays (action in/out)."""
    if int(step) < 1:
        raise ValueError(f"step must be >= 1 (completed steps), got {step}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = dense_trainable_weights(model)
    mx.eval(*weights.values())
    path = directory / DENSE_WEIGHTS_NAME
    mx.save_safetensors(str(path), weights)
    print(f"[DENSE] saved step={int(step)} n_arrays={len(weights)} → {path}")
    return {
        "directory": str(directory),
        "path": str(path),
        "n_arrays": len(weights),
        "n_elements": int(sum(int(v.size) for v in weights.values())),
        "step": int(step),
    }


def save_lora_adapters(
    model: Any,
    directory: str | Path,
    *,
    step: int,
    rank: int,
    scale: float,
    vision_scope: str,
    extra: dict[str, Any] | None = None,
    allow_extra_trainables: bool = False,
) -> dict[str, Any]:
    """Overwrite ``adapters.safetensors`` with the current LoRA A/B.

    Same filename every save. Packed ``QuantizedLinear.weight`` is not written.
    Raises if the trainable set is not LoRA A/B only, unless
    ``allow_extra_trainables`` (Stage-2 LoRA + dense action proj).
    """
    if int(step) < 1:
        raise ValueError(f"step must be >= 1 (completed steps), got {step}")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if allow_extra_trainables:
        weights = lora_adapter_weights(model)
    else:
        weights = lora_trainable_weights(model)
    mx.eval(*weights.values())
    path = directory / ADAPTER_WEIGHTS_NAME
    mx.save_safetensors(str(path), weights)
    config = {
        "fine_tune_type": "lora",
        "rank": int(rank),
        "scale": float(scale),
        "vision_scope": str(vision_scope),
        "n_arrays": len(weights),
        "n_elements": int(sum(int(v.size) for v in weights.values())),
        "step": int(step),
    }
    if extra:
        overlap = set(extra) & set(config)
        if overlap:
            raise ValueError(f"adapter extra keys collide with reserved fields: {sorted(overlap)}")
        config.update(extra)
    (directory / ADAPTER_CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
    print(f"[LORA] saved step={int(step)} n_arrays={len(weights)} → {path}")
    return {
        "directory": str(directory),
        "path": str(path),
        "n_arrays": len(weights),
        "step": int(step),
        "config": config,
    }


def load_lora_adapters(
    model: Any,
    directory: str | Path,
    *,
    weights_name: str = ADAPTER_WEIGHTS_NAME,
) -> dict[str, Any]:
    """Load a saved adapter after ``inject_backbone_lora``. Key sets must match."""
    directory = Path(directory)
    cfg_path = directory / ADAPTER_CONFIG_NAME
    w_path = directory / weights_name
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {ADAPTER_CONFIG_NAME} in {directory}")
    if not w_path.is_file():
        raise FileNotFoundError(f"missing {weights_name} in {directory}")
    config = json.loads(cfg_path.read_text())
    current = lora_trainable_weights(model)
    saved = mx.load(str(w_path))
    if not isinstance(saved, dict):
        raise RuntimeError(f"{w_path} did not load as a weight dict")
    missing = sorted(set(saved) - set(current))
    extra = sorted(set(current) - set(saved))
    if missing or extra:
        raise RuntimeError(
            f"adapter key set != model LoRA key set; "
            f"missing_on_model={missing[:6]} extra_on_model={extra[:6]}"
        )
    if int(config.get("n_arrays", -1)) != len(saved):
        raise RuntimeError(
            f"adapter_config n_arrays={config.get('n_arrays')} "
            f"!= file arrays {len(saved)}"
        )
    model.load_weights(str(w_path), strict=False)
    mx.eval(model.parameters())
    return config


def has_vision_lora(model: Any) -> bool:
    vlm = getattr(model, "vlm", None)
    tower = getattr(vlm, "vision_tower", None) if vlm is not None else None
    if tower is None or not hasattr(tower, "named_modules"):
        return False
    for _, mod in tower.named_modules():
        if isinstance(mod, LoRALinear):
            return True
    return False


def packed_weight_fingerprint(model: Any) -> str:
    """Hash of every QuantizedLinear.weight under language, vision, and expert decoders."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256()
    n = 0
    for layer in _language_layers(model):
        inner = decoder_layer_inner(layer)
        for _, mod in inner.named_modules():
            linear = getattr(mod, "linear", mod)
            if isinstance(linear, nn.QuantizedLinear):
                digest.update(np.asarray(linear.weight).tobytes())
                n += 1
    vlm = getattr(model, "vlm", None)
    tower = getattr(vlm, "vision_tower", None) if vlm is not None else None
    if tower is not None and hasattr(tower, "named_modules"):
        for _, mod in tower.named_modules():
            linear = getattr(mod, "linear", mod)
            if isinstance(linear, nn.QuantizedLinear):
                digest.update(np.asarray(linear.weight).tobytes())
                n += 1
    for layer in _maybe_expert_layers(model):
        inner = decoder_layer_inner(layer)
        for _, mod in inner.named_modules():
            linear = getattr(mod, "linear", mod)
            if isinstance(linear, nn.QuantizedLinear):
                digest.update(np.asarray(linear.weight).tobytes())
                n += 1
    if n < 1:
        raise RuntimeError("packed_weight_fingerprint found no QuantizedLinear")
    return f"{n}:{digest.hexdigest()}"


def freeze_vision_features(model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    """Encode pixels once, stop-grad, cache. Language-only LoRA; vision off the tape.

    Incompatible with vision LoRA: adapters would get zero gradient.
    """
    if has_vision_lora(model):
        raise RuntimeError(
            "freeze_vision_features stop-grads encode; incompatible with vision LoRA"
        )
    if batch.get("cached_image_features") is not None:
        return batch
    pixels = batch.get("pixel_values")
    if pixels is None:
        pixels = batch.get("pixel_values_videos")
    if pixels is None:
        return batch
    vlm = getattr(model, "vlm", None)
    if vlm is None or getattr(vlm, "vision_tower", None) is None:
        raise RuntimeError("PAI LoRA batch has pixels but model.vlm.vision_tower is missing")
    grid = batch.get("image_grid_thw")
    if grid is None:
        grid = batch.get("video_grid_thw")
    if grid is None:
        raise RuntimeError("PAI LoRA batch has pixels but no image_grid_thw")
    dtype = vlm.vision_tower.patch_embed.proj.weight.dtype
    hidden, deepstack = vlm.vision_tower(
        mx.array(pixels).astype(dtype),
        mx.array(grid, dtype=mx.int32),
    )
    if deepstack is None:
        raise RuntimeError("vision_tower returned no deepstack_visual_embeds")
    hidden = mx.stop_gradient(hidden)
    deepstack = [mx.stop_gradient(d) for d in deepstack]
    mx.eval(hidden, *deepstack)
    out = dict(batch)
    out["cached_image_features"] = hidden
    out["cached_deepstack_visual_embeds"] = deepstack
    return out


def sft_lora_update(
    model: Any,
    batch: dict[str, Any],
    optimizer: Any,
    stage: str = "stage1",
) -> TrainUpdateOutput:
    """One Stage-1 (or other) train step + Adam update on LoRA only."""
    t_all = time.perf_counter()
    assert_only_lora_trainable(model)
    encode_cache_ms = 0.0
    if has_vision_lora(model):
        if (
            batch.get("cached_image_features") is not None
            or batch.get("cached_deepstack_visual_embeds") is not None
        ):
            raise RuntimeError(
                "vision LoRA cannot use cached/stop-grad image features; "
                "encode must stay on the tape"
            )
    else:
        already = batch.get("cached_image_features") is not None
        t_enc = time.perf_counter()
        batch = freeze_vision_features(model, batch)
        if not already:
            encode_cache_ms = (time.perf_counter() - t_enc) * 1000.0

    def loss_fn(m: Any) -> mx.array:
        return sft_train_step(m, batch, stage=stage, materialize=False).loss

    loss, fwd_bwd_ms, adam_ms = run_value_and_grad_update(model, loss_fn, optimizer)
    n_expert = 0 if stage == "stage1" else 1
    return TrainUpdateOutput(
        loss=loss,
        times=TrainStepTimes(
            encode_cache_ms=encode_cache_ms,
            fwd_bwd_ms=fwd_bwd_ms,
            adam_ms=adam_ms,
            total_ms=(time.perf_counter() - t_all) * 1000.0,
            n_vlm_forwards=1,
            n_expert_forwards=n_expert,
        ),
    )
