"""Compile Qwen3-VL decoder layers for the prefill (seq_len > 1) path.

``layer(...)`` looks up ``__call__`` on the class, so an instance assignment
is ignored. Each layer is replaced with ``CompiledPrefillLayer``, whose class
``__call__`` is what ``Qwen3VLModel`` actually invokes.

``mx.compile`` cannot take a ``KVCache`` as an argument. The cache is captured
in the closure and the function is recaptured when the cache object or
sequence length changes. Decode (seq_len == 1) stays eager.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn

from mlx_port.stage_timers import set_compiled

# First successful graph capture (for e2e / tests). Not a silent fallback.
_prefill_captures = 0


def prefill_compile_captures() -> int:
    return _prefill_captures


def reset_prefill_compile_captures() -> None:
    global _prefill_captures
    _prefill_captures = 0


class CompiledPrefillLayer(nn.Module):
    """nn.Module wrapper so ``layer(...)`` hits this class ``__call__``."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        if inner is None:
            raise ValueError("CompiledPrefillLayer requires an inner decoder layer")
        self.inner = inner
        self._alpamayo_prefill_compiled = True
        self._fns: dict[tuple, Callable] = {}

    def __call__(self, x, mask=None, cache=None, position_ids=None):
        seq = int(x.shape[1])
        if seq <= 1 or cache is None:
            return self.inner(x, mask, cache, position_ids)

        if isinstance(mask, str) or mask is None:
            mask_key: Any = mask
        else:
            mask_key = ("array", tuple(mask.shape))
        key = (id(cache), seq, mask_key)
        fn = self._fns.get(key)
        if fn is None:
            inner = self.inner
            if isinstance(mask, str) or mask is None:

                def _fn(h, pos):
                    out = inner(h, mask, cache, pos)
                    if cache.keys is None or cache.values is None:
                        raise RuntimeError(
                            "compiled prefill did not write KV cache keys/values"
                        )
                    return out, cache.keys, cache.values

                fn = mx.compile(_fn)
            else:

                def _fn(h, pos, m):
                    out = inner(h, m, cache, pos)
                    if cache.keys is None or cache.values is None:
                        raise RuntimeError(
                            "compiled prefill did not write KV cache keys/values"
                        )
                    return out, cache.keys, cache.values

                fn = mx.compile(_fn)
            self._fns[key] = fn
            global _prefill_captures
            _prefill_captures += 1
            if _prefill_captures == 1:
                print(
                    f"[COMPILE] first prefill graph capture seq_len={seq} "
                    f"(later layers recapture per cache/seq)"
                )

        if isinstance(mask, str) or mask is None:
            out, keys, values = fn(x, position_ids)
        else:
            out, keys, values = fn(x, position_ids, mask)
        # Compile captures KV as outputs. Write them back so eager decode can eval.
        cache.keys = keys
        cache.values = values
        return out


def wrap_decoder_layer_prefill(layer: Any) -> CompiledPrefillLayer:
    """Return a class-``__call__`` wrapper. Idempotent."""
    if isinstance(layer, CompiledPrefillLayer):
        return layer
    if layer is None:
        raise ValueError("wrap_decoder_layer_prefill requires a decoder layer")
    return CompiledPrefillLayer(layer)


def install_compiled_prefill(language_model: Any) -> int:
    """Replace ``language_model.model.layers`` entries with compiled wrappers.

    Returns the number of layers wrapped (0 if already installed).
    """
    if language_model is None or not hasattr(language_model, "model"):
        raise ValueError("install_compiled_prefill requires a language model with .model")
    layers = getattr(language_model.model, "layers", None)
    if layers is None:
        raise ValueError("language_model.model.layers is missing")

    n = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, CompiledPrefillLayer):
            continue
        layers[i] = wrap_decoder_layer_prefill(layer)
        n += 1
    set_compiled("prefill", True)
    if n:
        print(f"[COMPILE] installed CompiledPrefillLayer on {n} decoder layers")
    return n
