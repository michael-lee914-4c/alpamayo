"""``MLPEncoder`` rejects ``num_enc_layers < 1``.

Weight-free. Complementary to open #29 (Fourier logspace ``[sin,cos]*√2``)
and #30 (action-in ``timesteps[..., -1]``). A zero-layer trunk would skip
the RMSNorm+Linear stack and emit the raw Fourier concat into the expert.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_PROJ = _SRC / "alpamayo_r1" / "models" / "action_in_proj.py"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_mlp_encoder():
    _ensure_src_on_path()
    pytest.importorskip("torch")
    try:
        from alpamayo_r1.models.action_in_proj import MLPEncoder

        return MLPEncoder
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_action_in_proj_mlp", _PROJ)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_PROJ}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            pytest.skip(f"NVIDIA action_in_proj unavailable: {exc}")
        return module.MLPEncoder


def test_source_requires_at_least_one_enc_layer():
    src = _PROJ.read_text()
    assert "assert 1 <= num_enc_layers, f\"{num_enc_layers=} must be >= 1\"" in src
    assert "nn.Linear(num_input_feats, hidden_size)" in src
    assert "RMSNorm(hidden_size, eps=1e-5)" in src


def test_zero_and_negative_layers_raise():
    MLPEncoder = _load_mlp_encoder()
    with pytest.raises(AssertionError, match="num_enc_layers"):
        MLPEncoder(num_input_feats=4, num_enc_layers=0, hidden_size=8, outdim=4)
    with pytest.raises(AssertionError, match="num_enc_layers"):
        MLPEncoder(num_input_feats=4, num_enc_layers=-1, hidden_size=8, outdim=4)


def test_one_layer_maps_to_outdim():
    torch = pytest.importorskip("torch")
    MLPEncoder = _load_mlp_encoder()
    enc = MLPEncoder(num_input_feats=3, num_enc_layers=1, hidden_size=8, outdim=5)
    out = enc(torch.zeros(2, 3))
    assert out.shape == (2, 5)
