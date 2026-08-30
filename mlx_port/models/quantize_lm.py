"""T3.1 Recipe A: affine 4-bit on the language tower only.

Vision, expert, ``lm_head``, and embeddings stay dense bf16. Do not pass the
full ``AlpamayoR1MLX`` or the VLM (with ``vision_tower``) into
``quantize_language_tower`` — raise instead of walking them.

Kept load paths: dense bf16 (default), T3.1 (``lm4``), and all4 (VLM +
expert; see ``quantize_all.py``). ``ALPAMAYO_QUANT`` may be unset,
``none``, ``lm4``, or ``all4``. Other values raise.
"""

from __future__ import annotations

import json
import os
from typing import Any

import mlx.nn as nn

from mlx_port.stage_timers import set_quantized

QUANT_MODE_ENV = "ALPAMAYO_QUANT"
LM4_DIR_ENV = "ALPAMAYO_LM4_DIR"
DEFAULT_LM4_DIRNAME = "mlx_lm4"
LM4_WEIGHTS_NAME = "language_model.safetensors"
LM4_CONFIG_NAME = "config.json"

QUANT_BITS = 4
QUANT_GROUP_SIZE = 64
QUANT_MODE = "affine"
QUANT_SPEC = f"affine-{QUANT_BITS}-gs{QUANT_GROUP_SIZE}"

# Path substrings that must stay bf16 (not 8-bit). Matches T3.1 KEEP_DENSE.
KEEP_DENSE_SUBSTR = (
    "vision",
    "visual",
    "vit",
    "action",
    "expert",
    "lm_head",
    "embed",
    "embed_tokens",
)

_ATTR_DONE = "_alpamayo_lm_quantized"


def mark_language_tower_dense() -> None:
    """Record the signed P2f path: decoder stays dense bf16."""
    set_quantized("lm", "bf16")
    set_quantized("vision", "bf16")
    set_quantized("expert", "bf16")
    print("[QUANT] language tower dense bf16")


def resolve_quant_mode(
    quantize_lm: bool = False, quantize_all: bool = False
) -> str:
    """Return ``none``, ``lm4``, or ``all4``. Env overrides kwargs.

    ``ALPAMAYO_QUANT=none|lm4|all4``. Any other value raises.
    ``quantize_lm`` and ``quantize_all`` are exclusive when env is unset.
    """
    env = os.environ.get(QUANT_MODE_ENV, "").strip().lower()
    if env:
        if env == "none":
            return "none"
        if env in ("lm4", "all4"):
            return env
        raise ValueError(
            f"{QUANT_MODE_ENV}={env!r} is not supported. "
            "Use unset / none (bf16), lm4 (T3.1), or all4 (VLM+expert)."
        )
    if quantize_lm and quantize_all:
        raise ValueError("quantize_lm and quantize_all are exclusive")
    if quantize_all:
        return "all4"
    if quantize_lm:
        return "lm4"
    return "none"


def lm_quant_enabled(quantize_lm: bool) -> bool:
    """T3.1 on/off. Unset follows the kwarg (default False = bf16)."""
    return resolve_quant_mode(quantize_lm=quantize_lm) == "lm4"


def keep_dense(path: str) -> bool:
    if path is None:
        raise ValueError("keep_dense requires a module path")
    pl = str(path).lower()
    return any(s in pl for s in KEEP_DENSE_SUBSTR)


def language_tower_predicate(path: str, module: Any) -> bool | dict:
    """4-bit affine gs64, or False to leave the module dense."""
    if module is None:
        return False
    if not hasattr(module, "to_quantized"):
        return False
    if keep_dense(path):
        return False
    return {"bits": QUANT_BITS, "group_size": QUANT_GROUP_SIZE, "mode": QUANT_MODE}


def _refuse_if_not_language_tower(language_model: Any) -> None:
    if language_model is None:
        raise ValueError("quantize_language_tower requires a language model")
    if hasattr(language_model, "vlm") and hasattr(language_model, "expert"):
        raise ValueError(
            "quantize_language_tower: pass vlm.language_model, not AlpamayoR1MLX"
        )
    if hasattr(language_model, "vision_tower"):
        raise ValueError(
            "quantize_language_tower: pass language_model, not the full VLM"
        )
    if not hasattr(language_model, "lm_head"):
        raise ValueError(
            "quantize_language_tower: pass vlm.language_model (must have lm_head)"
        )


def _leaf_summary(language_model: Any) -> dict[str, Any]:
    quantized: list[str] = []
    dense_kept: list[str] = []
    for path, mod in language_model.named_modules():
        if isinstance(mod, nn.QuantizedLinear):
            quantized.append(path or ".")
        elif isinstance(mod, (nn.Linear, nn.Embedding)):
            dense_kept.append(path or ".")
    return {
        "n_quantized_linear": len(quantized),
        "n_dense_linear_or_embed": len(dense_kept),
        "quantized_paths": quantized,
        "dense_paths": dense_kept,
        "bits": QUANT_BITS,
        "group_size": QUANT_GROUP_SIZE,
        "mode": QUANT_MODE,
        "spec": QUANT_SPEC,
    }


def quantize_language_tower(language_model: Any) -> dict[str, Any]:
    """In-place affine 4-bit PTQ of decoder Linears. Embeddings and lm_head stay bf16."""
    _refuse_if_not_language_tower(language_model)
    if getattr(language_model, _ATTR_DONE, False):
        summary = _leaf_summary(language_model)
        print(
            f"[QUANT] language tower already {QUANT_SPEC}: "
            f"{summary['n_quantized_linear']} QuantizedLinear"
        )
        return summary

    nn.quantize(
        language_model,
        group_size=QUANT_GROUP_SIZE,
        bits=QUANT_BITS,
        mode=QUANT_MODE,
        class_predicate=language_tower_predicate,
    )
    setattr(language_model, _ATTR_DONE, True)
    summary = _leaf_summary(language_model)
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError(
            "quantize_language_tower: no Linear was quantized "
            "(predicate kept everything dense, or the tower has no Linears)"
        )
    for path in summary["dense_paths"]:
        pl = path.lower()
        if not keep_dense(pl):
            raise RuntimeError(
                f"quantize_language_tower left a non-KEEP_DENSE Linear dense: {path}"
            )
    set_quantized("lm", QUANT_SPEC)
    set_quantized("vision", "bf16")
    set_quantized("expert", "bf16")
    print(
        f"[QUANT] language tower {QUANT_SPEC}: "
        f"{summary['n_quantized_linear']} QuantizedLinear, "
        f"kept dense {summary['n_dense_linear_or_embed']} "
        f"({', '.join(summary['dense_paths']) or 'none'})"
    )
    return summary


def resolve_lm4_dir(alpamayo_path: str, lm4_path: str | None = None) -> str:
    """Packed T3.1 language-tower dir. Kwarg, then env, then ``{alpamayo}/mlx_lm4``."""
    if lm4_path:
        return os.path.abspath(lm4_path)
    env = os.environ.get(LM4_DIR_ENV, "").strip()
    if env:
        return os.path.abspath(env)
    if not alpamayo_path:
        raise ValueError("resolve_lm4_dir requires alpamayo_path or lm4_path")
    return os.path.join(os.path.abspath(alpamayo_path), DEFAULT_LM4_DIRNAME)


def lm4_checkpoint_ready(dest_dir: str) -> bool:
    """True if both files exist. Incomplete pair raises — do not live-pack."""
    if not dest_dir:
        raise ValueError("lm4_checkpoint_ready requires dest_dir")
    weights = os.path.join(dest_dir, LM4_WEIGHTS_NAME)
    config = os.path.join(dest_dir, LM4_CONFIG_NAME)
    have_w = os.path.isfile(weights)
    have_c = os.path.isfile(config)
    if have_w and have_c:
        return True
    if have_w or have_c:
        raise FileNotFoundError(
            f"incomplete T3.1 checkpoint in {dest_dir}: "
            f"weights={have_w} config={have_c}. "
            "Delete the leftover file or finish the save."
        )
    return False


def _get_child(parent: Any, name: str) -> Any:
    if name.isdigit():
        idx = int(name)
        if isinstance(parent, (list, tuple)):
            return parent[idx]
        if hasattr(parent, "layers"):
            return parent.layers[idx]
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
        raise TypeError(f"cannot assign index {name} on {type(parent).__name__}")
    setattr(parent, name, value)


def install_empty_quantized_linears(language_model: Any) -> int:
    """Replace decoder Linears with empty QuantizedLinear (no live pack)."""
    _refuse_if_not_language_tower(language_model)
    n = 0
    for path, mod in list(language_model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        spec = language_tower_predicate(path, mod)
        if spec is False:
            continue
        in_dims = int(mod.weight.shape[1])
        out_dims = int(mod.weight.shape[0])
        has_bias = "bias" in mod
        q = nn.QuantizedLinear(
            in_dims,
            out_dims,
            bias=has_bias,
            group_size=int(spec["group_size"]),
            bits=int(spec["bits"]),
            mode=str(spec["mode"]),
        )
        parts = path.split(".")
        parent = language_model
        for part in parts[:-1]:
            parent = _get_child(parent, part)
        _set_child(parent, parts[-1], q)
        n += 1
    if n < 1:
        raise RuntimeError(
            "install_empty_quantized_linears: no decoder Linear was replaced"
        )
    return n


def _read_lm4_config(dest_dir: str) -> dict[str, Any]:
    path = os.path.join(dest_dir, LM4_CONFIG_NAME)
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


def save_language_tower(language_model: Any, dest_dir: str) -> dict[str, Any]:
    """Write packed language-tower weights + config. Raises if the tower is dense."""
    _refuse_if_not_language_tower(language_model)
    if dest_dir is None or not str(dest_dir).strip():
        raise ValueError("save_language_tower requires dest_dir")
    summary = _leaf_summary(language_model)
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError(
            "save_language_tower: language tower is dense; quantize first"
        )
    os.makedirs(dest_dir, exist_ok=True)
    weights_path = os.path.join(dest_dir, LM4_WEIGHTS_NAME)
    language_model.save_weights(weights_path)
    nbytes = os.path.getsize(weights_path)
    cfg = {
        "spec": QUANT_SPEC,
        "bits": QUANT_BITS,
        "group_size": QUANT_GROUP_SIZE,
        "mode": QUANT_MODE,
        "n_quantized_linear": summary["n_quantized_linear"],
        "dense_paths": summary["dense_paths"],
        "weights": LM4_WEIGHTS_NAME,
        "bytes": nbytes,
    }
    config_path = os.path.join(dest_dir, LM4_CONFIG_NAME)
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(
        f"[QUANT] saved {QUANT_SPEC} language tower → {weights_path} "
        f"({nbytes / (1024 ** 3):.2f} GiB, "
        f"{summary['n_quantized_linear']} QuantizedLinear)"
    )
    return {**summary, "dest_dir": dest_dir, "weights_path": weights_path, "bytes": nbytes}


def load_language_tower(language_model: Any, dest_dir: str) -> dict[str, Any]:
    """Install QuantizedLinear leaves and load packed weights. No live PTQ pack."""
    _refuse_if_not_language_tower(language_model)
    if not lm4_checkpoint_ready(dest_dir):
        raise FileNotFoundError(f"T3.1 checkpoint not found in {dest_dir}")
    cfg = _read_lm4_config(dest_dir)
    summary_before = _leaf_summary(language_model)
    if summary_before["n_quantized_linear"] < 1:
        install_empty_quantized_linears(language_model)
    weights_path = os.path.join(dest_dir, LM4_WEIGHTS_NAME)
    language_model.load_weights(weights_path, strict=True)
    setattr(language_model, _ATTR_DONE, True)
    summary = _leaf_summary(language_model)
    expected = cfg.get("n_quantized_linear")
    if expected is not None and summary["n_quantized_linear"] != expected:
        raise RuntimeError(
            f"load_language_tower: expected {expected} QuantizedLinear, "
            f"got {summary['n_quantized_linear']}"
        )
    if summary["n_quantized_linear"] < 1:
        raise RuntimeError("load_language_tower: no QuantizedLinear after load")
    set_quantized("lm", QUANT_SPEC)
    set_quantized("vision", "bf16")
    set_quantized("expert", "bf16")
    print(
        f"[QUANT] loaded {QUANT_SPEC} language tower from {weights_path}: "
        f"{summary['n_quantized_linear']} QuantizedLinear"
    )
    return {**summary, "dest_dir": dest_dir, "source": "disk"}


def apply_language_tower_quant(language_model: Any, dest_dir: str) -> dict[str, Any]:
    """Load packed T3.1 from disk, or live-pack and save if the dir is empty."""
    if lm4_checkpoint_ready(dest_dir):
        return load_language_tower(language_model, dest_dir)
    summary = quantize_language_tower(language_model)
    saved = save_language_tower(language_model, dest_dir)
    return {**summary, **saved, "source": "live-pack"}
