"""all4: affine 4-bit on the full VLM and the diffusion expert.

Action-in / action-out stay dense bf16. Leaves whose last dim is not
divisible by 64 (vision ``linear_fc2`` at 4304) stay dense and are
listed — that is not a silent fallback. Conv3D has no ``to_quantized``.

Do not pass ``AlpamayoR1MLX`` into the VLM or expert walkers.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.nn as nn

from mlx_port.models.quantize_lm import (
    QUANT_BITS,
    QUANT_GROUP_SIZE,
    QUANT_MODE,
    QUANT_SPEC,
)
from mlx_port.stage_timers import set_quantized

ALL4_DIR_ENV = "ALPAMAYO_ALL4_DIR"
DEFAULT_ALL4_DIRNAME = "mlx_all4"
ALL4_VLM_WEIGHTS_NAME = "vlm.safetensors"
ALL4_EXPERT_WEIGHTS_NAME = "expert.safetensors"
ALL4_CONFIG_NAME = "config.json"

_VLM_DONE = "_alpamayo_vlm_all4"
_EXPERT_DONE = "_alpamayo_expert_all4"

ACTION_KEEP_SUBSTR = ("action_in", "action_out", "action_in_proj", "action_out_proj")


def last_dim_packable(module: Any, group_size: int = QUANT_GROUP_SIZE) -> bool:
    """True when the quantized axis (weight last dim) is divisible by group size."""
    if module is None or not hasattr(module, "weight"):
        return False
    last = int(module.weight.shape[-1])
    return last > 0 and last % int(group_size) == 0


def all4_predicate(path: str, module: Any) -> bool | dict:
    """Pack Linear / Embedding when last dim is ÷64. Action projections stay dense."""
    if module is None:
        return False
    if not hasattr(module, "to_quantized"):
        return False
    pl = str(path or "").lower()
    if any(s in pl for s in ACTION_KEEP_SUBSTR):
        return False
    if not last_dim_packable(module):
        return False
    return {"bits": QUANT_BITS, "group_size": QUANT_GROUP_SIZE, "mode": QUANT_MODE}


def _refuse_full_alpamayo(root: Any, what: str) -> None:
    if root is None:
        raise ValueError(f"{what} requires a module")
    if hasattr(root, "vlm") and hasattr(root, "expert"):
        raise ValueError(f"{what}: pass vlm or expert, not AlpamayoR1MLX")


def _refuse_if_not_vlm(vlm: Any) -> None:
    _refuse_full_alpamayo(vlm, "quantize_vlm_all4")
    if not hasattr(vlm, "vision_tower") or not hasattr(vlm, "language_model"):
        raise ValueError("quantize_vlm_all4: pass the full VLM (vision_tower + language_model)")


def _refuse_if_not_expert(expert: Any) -> None:
    _refuse_full_alpamayo(expert, "quantize_expert_all4")
    if hasattr(expert, "vision_tower"):
        raise ValueError("quantize_expert_all4: pass the expert, not the VLM")
    if hasattr(expert, "action_in_proj") or hasattr(expert, "action_out_proj"):
        raise ValueError("quantize_expert_all4: pass the expert, not AlpamayoR1MLX")
    if not hasattr(expert, "language_model"):
        raise ValueError("quantize_expert_all4: pass AlpamayoExpert (must have language_model)")


def _leaf_summary(root: Any) -> dict[str, Any]:
    q_lin: list[str] = []
    q_emb: list[str] = []
    dense: list[str] = []
    unpacked: list[str] = []
    for path, mod in root.named_modules():
        name = path or "."
        if isinstance(mod, nn.QuantizedLinear):
            q_lin.append(name)
        elif isinstance(mod, nn.QuantizedEmbedding):
            q_emb.append(name)
        elif isinstance(mod, (nn.Linear, nn.Embedding)):
            dense.append(name)
            if hasattr(mod, "to_quantized") and not last_dim_packable(mod):
                unpacked.append(name)
    return {
        "n_quantized_linear": len(q_lin),
        "n_quantized_embedding": len(q_emb),
        "n_dense_linear_or_embed": len(dense),
        "quantized_linear_paths": q_lin,
        "quantized_embedding_paths": q_emb,
        "dense_paths": dense,
        "unpacked_last_dim_paths": unpacked,
        "bits": QUANT_BITS,
        "group_size": QUANT_GROUP_SIZE,
        "mode": QUANT_MODE,
        "spec": QUANT_SPEC,
    }


def _audit_after_quantize(root: Any, summary: dict[str, Any]) -> None:
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError("all4: no Linear was quantized")
    for path, mod in root.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Embedding)):
            continue
        pl = str(path or "").lower()
        if any(s in pl for s in ACTION_KEEP_SUBSTR):
            continue
        if last_dim_packable(mod):
            raise RuntimeError(
                f"all4 left a packable {type(mod).__name__} dense: {path or '.'}"
            )


def _get_child(parent: Any, name: str) -> Any:
    if name.isdigit():
        idx = int(name)
        if isinstance(parent, (list, tuple)):
            return parent[idx]
        if hasattr(parent, "layers"):
            return parent.layers[idx]
        if hasattr(parent, "blocks"):
            return parent.blocks[idx]
        raise TypeError(f"cannot index {type(parent).__name__} with {name}")
    return getattr(parent, name)


def _set_child(parent: Any, name: str, value: Any) -> None:
    if name.isdigit():
        idx = int(name)
        if isinstance(parent, list):
            parent[idx] = value
            return
        if hasattr(parent, "layers"):
            parent.layers[idx] = value
            return
        if hasattr(parent, "blocks"):
            parent.blocks[idx] = value
            return
        raise TypeError(f"cannot assign index {name} on {type(parent).__name__}")
    setattr(parent, name, value)


def install_empty_quantized_modules(root: Any, predicate) -> int:
    """Replace packable Linear / Embedding with empty quantized leaves."""
    n = 0
    for path, mod in list(root.named_modules()):
        spec = predicate(path, mod)
        if spec is False:
            continue
        if isinstance(mod, nn.Linear):
            in_dims = int(mod.weight.shape[1])
            out_dims = int(mod.weight.shape[0])
            q = nn.QuantizedLinear(
                in_dims,
                out_dims,
                bias="bias" in mod,
                group_size=int(spec["group_size"]),
                bits=int(spec["bits"]),
                mode=str(spec["mode"]),
            )
        elif isinstance(mod, nn.Embedding):
            q = nn.QuantizedEmbedding(
                int(mod.weight.shape[0]),
                int(mod.weight.shape[1]),
                group_size=int(spec["group_size"]),
                bits=int(spec["bits"]),
                mode=str(spec["mode"]),
            )
        else:
            continue
        parts = path.split(".")
        parent = root
        for part in parts[:-1]:
            parent = _get_child(parent, part)
        _set_child(parent, parts[-1], q)
        n += 1
    if n < 1:
        raise RuntimeError("install_empty_quantized_modules: no leaf was replaced")
    return n


def resolve_all4_dir(alpamayo_path: str, all4_path: str | None = None) -> str:
    if all4_path:
        return os.path.abspath(all4_path)
    env = os.environ.get(ALL4_DIR_ENV, "").strip()
    if env:
        return os.path.abspath(env)
    if not alpamayo_path:
        raise ValueError("resolve_all4_dir requires alpamayo_path or all4_path")
    return os.path.join(os.path.abspath(alpamayo_path), DEFAULT_ALL4_DIRNAME)


def all4_checkpoint_ready(dest_dir: str) -> bool:
    """True if vlm + expert + config exist. Incomplete trio raises."""
    if not dest_dir:
        raise ValueError("all4_checkpoint_ready requires dest_dir")
    vlm = os.path.join(dest_dir, ALL4_VLM_WEIGHTS_NAME)
    expert = os.path.join(dest_dir, ALL4_EXPERT_WEIGHTS_NAME)
    config = os.path.join(dest_dir, ALL4_CONFIG_NAME)
    have = (os.path.isfile(vlm), os.path.isfile(expert), os.path.isfile(config))
    if all(have):
        return True
    if any(have):
        raise FileNotFoundError(
            f"incomplete all4 checkpoint in {dest_dir}: "
            f"vlm={have[0]} expert={have[1]} config={have[2]}. "
            "Delete the leftovers or finish the save."
        )
    return False


def _read_all4_config(dest_dir: str) -> dict[str, Any]:
    path = os.path.join(dest_dir, ALL4_CONFIG_NAME)
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is not a JSON object")
    for key, expected in (
        ("spec", QUANT_SPEC),
        ("bits", QUANT_BITS),
        ("group_size", QUANT_GROUP_SIZE),
        ("mode", QUANT_MODE),
    ):
        if cfg.get(key) != expected:
            raise ValueError(
                f"{path} {key}={cfg.get(key)!r} does not match {expected!r}"
            )
    return cfg


def _write_all4_config(dest_dir: str, payload: dict[str, Any]) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, ALL4_CONFIG_NAME)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def quantize_vlm_all4(vlm: Any) -> dict[str, Any]:
    """In-place affine-4 of packable VLM Linears / embeddings."""
    _refuse_if_not_vlm(vlm)
    if getattr(vlm, _VLM_DONE, False):
        summary = _leaf_summary(vlm)
        print(
            f"[QUANT] VLM already {QUANT_SPEC}: "
            f"{summary['n_quantized_linear']} QuantizedLinear, "
            f"{summary['n_quantized_embedding']} QuantizedEmbedding"
        )
        return summary
    nn.quantize(
        vlm,
        group_size=QUANT_GROUP_SIZE,
        bits=QUANT_BITS,
        mode=QUANT_MODE,
        class_predicate=all4_predicate,
    )
    setattr(vlm, _VLM_DONE, True)
    summary = _leaf_summary(vlm)
    _audit_after_quantize(vlm, summary)
    set_quantized("lm", QUANT_SPEC)
    set_quantized("vision", QUANT_SPEC)
    print(
        f"[QUANT] VLM {QUANT_SPEC}: {summary['n_quantized_linear']} QuantizedLinear, "
        f"{summary['n_quantized_embedding']} QuantizedEmbedding, "
        f"{len(summary['unpacked_last_dim_paths'])} dense last-dim skips "
        f"({', '.join(summary['unpacked_last_dim_paths']) or 'none'})"
    )
    return summary


def quantize_expert_all4(expert: Any) -> dict[str, Any]:
    """In-place affine-4 of packable expert Linears."""
    _refuse_if_not_expert(expert)
    if getattr(expert, _EXPERT_DONE, False):
        summary = _leaf_summary(expert)
        print(
            f"[QUANT] expert already {QUANT_SPEC}: "
            f"{summary['n_quantized_linear']} QuantizedLinear"
        )
        return summary
    nn.quantize(
        expert,
        group_size=QUANT_GROUP_SIZE,
        bits=QUANT_BITS,
        mode=QUANT_MODE,
        class_predicate=all4_predicate,
    )
    setattr(expert, _EXPERT_DONE, True)
    summary = _leaf_summary(expert)
    _audit_after_quantize(expert, summary)
    set_quantized("expert", QUANT_SPEC)
    print(
        f"[QUANT] expert {QUANT_SPEC}: {summary['n_quantized_linear']} QuantizedLinear"
    )
    return summary


def save_vlm_all4(vlm: Any, dest_dir: str) -> dict[str, Any]:
    _refuse_if_not_vlm(vlm)
    if dest_dir is None or not str(dest_dir).strip():
        raise ValueError("save_vlm_all4 requires dest_dir")
    summary = _leaf_summary(vlm)
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError("save_vlm_all4: VLM is dense; quantize first")
    os.makedirs(dest_dir, exist_ok=True)
    weights_path = os.path.join(dest_dir, ALL4_VLM_WEIGHTS_NAME)
    vlm.save_weights(weights_path)
    nbytes = os.path.getsize(weights_path)
    print(
        f"[QUANT] saved {QUANT_SPEC} VLM → {weights_path} "
        f"({nbytes / (1024 ** 3):.2f} GiB)"
    )
    return {**summary, "dest_dir": dest_dir, "weights_path": weights_path, "bytes": nbytes}


def save_expert_all4(expert: Any, dest_dir: str, vlm_summary: dict[str, Any]) -> dict[str, Any]:
    _refuse_if_not_expert(expert)
    if dest_dir is None or not str(dest_dir).strip():
        raise ValueError("save_expert_all4 requires dest_dir")
    summary = _leaf_summary(expert)
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError("save_expert_all4: expert is dense; quantize first")
    os.makedirs(dest_dir, exist_ok=True)
    weights_path = os.path.join(dest_dir, ALL4_EXPERT_WEIGHTS_NAME)
    expert.save_weights(weights_path)
    nbytes = os.path.getsize(weights_path)
    vlm_bytes = int(vlm_summary.get("bytes") or 0)
    cfg = {
        "spec": QUANT_SPEC,
        "bits": QUANT_BITS,
        "group_size": QUANT_GROUP_SIZE,
        "mode": QUANT_MODE,
        "vlm_weights": ALL4_VLM_WEIGHTS_NAME,
        "expert_weights": ALL4_EXPERT_WEIGHTS_NAME,
        "n_vlm_quantized_linear": vlm_summary.get("n_quantized_linear"),
        "n_vlm_quantized_embedding": vlm_summary.get("n_quantized_embedding"),
        "n_expert_quantized_linear": summary["n_quantized_linear"],
        "vlm_unpacked_last_dim_paths": vlm_summary.get("unpacked_last_dim_paths") or [],
        "vlm_bytes": vlm_bytes,
        "expert_bytes": nbytes,
    }
    _write_all4_config(dest_dir, cfg)
    print(
        f"[QUANT] saved {QUANT_SPEC} expert → {weights_path} "
        f"({nbytes / (1024 ** 3):.2f} GiB)"
    )
    return {**summary, "dest_dir": dest_dir, "weights_path": weights_path, "bytes": nbytes}


def load_vlm_all4(vlm: Any, dest_dir: str) -> dict[str, Any]:
    _refuse_if_not_vlm(vlm)
    if not all4_checkpoint_ready(dest_dir):
        raise FileNotFoundError(f"all4 checkpoint not found in {dest_dir}")
    cfg = _read_all4_config(dest_dir)
    if _leaf_summary(vlm)["n_quantized_linear"] < 1:
        install_empty_quantized_modules(vlm, all4_predicate)
    weights_path = os.path.join(dest_dir, ALL4_VLM_WEIGHTS_NAME)
    vlm.load_weights(weights_path, strict=True)
    setattr(vlm, _VLM_DONE, True)
    summary = _leaf_summary(vlm)
    expected = cfg.get("n_vlm_quantized_linear")
    if expected is not None and summary["n_quantized_linear"] != expected:
        raise RuntimeError(
            f"load_vlm_all4: expected {expected} QuantizedLinear, "
            f"got {summary['n_quantized_linear']}"
        )
    set_quantized("lm", QUANT_SPEC)
    set_quantized("vision", QUANT_SPEC)
    print(
        f"[QUANT] loaded {QUANT_SPEC} VLM from {weights_path}: "
        f"{summary['n_quantized_linear']} QuantizedLinear, "
        f"{summary['n_quantized_embedding']} QuantizedEmbedding"
    )
    return {**summary, "dest_dir": dest_dir, "source": "disk"}


def load_expert_all4(expert: Any, dest_dir: str) -> dict[str, Any]:
    _refuse_if_not_expert(expert)
    if not all4_checkpoint_ready(dest_dir):
        raise FileNotFoundError(f"all4 checkpoint not found in {dest_dir}")
    cfg = _read_all4_config(dest_dir)
    if _leaf_summary(expert)["n_quantized_linear"] < 1:
        install_empty_quantized_modules(expert, all4_predicate)
    weights_path = os.path.join(dest_dir, ALL4_EXPERT_WEIGHTS_NAME)
    expert.load_weights(weights_path, strict=True)
    setattr(expert, _EXPERT_DONE, True)
    summary = _leaf_summary(expert)
    expected = cfg.get("n_expert_quantized_linear")
    if expected is not None and summary["n_quantized_linear"] != expected:
        raise RuntimeError(
            f"load_expert_all4: expected {expected} QuantizedLinear, "
            f"got {summary['n_quantized_linear']}"
        )
    set_quantized("expert", QUANT_SPEC)
    print(
        f"[QUANT] loaded {QUANT_SPEC} expert from {weights_path}: "
        f"{summary['n_quantized_linear']} QuantizedLinear"
    )
    return {**summary, "dest_dir": dest_dir, "source": "disk"}


def apply_vlm_all4(vlm: Any, dest_dir: str) -> dict[str, Any]:
    """Load packed VLM from disk, or live-pack (does not write config yet)."""
    if all4_checkpoint_ready(dest_dir):
        return load_vlm_all4(vlm, dest_dir)
    summary = quantize_vlm_all4(vlm)
    saved = save_vlm_all4(vlm, dest_dir)
    return {**summary, **saved, "source": "live-pack"}


def apply_expert_all4(expert: Any, dest_dir: str, vlm_summary: dict[str, Any]) -> dict[str, Any]:
    """Load packed expert from disk, or live-pack and finish config.json.

    After a same-process VLM live-pack, only ``vlm.safetensors`` exists.
    Do not call ``all4_checkpoint_ready`` here — that trio check would raise.
    """
    vlm_w = os.path.join(dest_dir, ALL4_VLM_WEIGHTS_NAME)
    exp_w = os.path.join(dest_dir, ALL4_EXPERT_WEIGHTS_NAME)
    cfg_w = os.path.join(dest_dir, ALL4_CONFIG_NAME)
    if os.path.isfile(vlm_w) and os.path.isfile(exp_w) and os.path.isfile(cfg_w):
        return load_expert_all4(expert, dest_dir)
    if not os.path.isfile(vlm_w):
        raise RuntimeError(
            "apply_expert_all4: missing vlm.safetensors; apply_vlm_all4 first"
        )
    if os.path.isfile(exp_w) or os.path.isfile(cfg_w):
        raise FileNotFoundError(
            f"incomplete all4 expert/config in {dest_dir}: "
            f"expert={os.path.isfile(exp_w)} config={os.path.isfile(cfg_w)}"
        )
    summary = quantize_expert_all4(expert)
    saved = save_expert_all4(expert, dest_dir, vlm_summary)
    return {**summary, **saved, "source": "live-pack"}
