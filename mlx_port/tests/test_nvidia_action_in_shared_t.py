"""Action-in encodes ``timesteps[..., -1]`` once and repeats it over T.

Weight-free. Complementary to open #30 (``timesteps[..., -1]`` is the
source) and #29 (Fourier logspace). A per-waypoint ``timesteps`` tensor
still collapses to the last scalar; repeating that vector is what the
expert sees. Indexing ``[..., 0]`` or skipping the repeat would
condition CFM on the wrong t.
"""

from __future__ import annotations

import importlib.util
import math
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


def _load_fourier():
    _ensure_src_on_path()
    pytest.importorskip("torch")
    try:
        from alpamayo_r1.models.action_in_proj import FourierEncoderV2

        return FourierEncoderV2
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_action_in_shared_t", _PROJ)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_PROJ}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            pytest.skip(f"NVIDIA action_in_proj unavailable: {exc}")
        return module.FourierEncoderV2


def test_source_takes_last_timestep_and_repeats():
    src = _PROJ.read_text()
    assert "timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])" in src
    assert "timestep_feats = timestep_feats.repeat(1, T, 1)" in src
    assert "action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)" in src


def test_fm_rank_matches_action_and_repeats_across_t():
    torch = pytest.importorskip("torch")
    FourierEncoderV2 = _load_fourier()
    enc = FourierEncoderV2(dim=4, max_freq=10.0)
    # FlowMatching unsqueezes t until it matches action rank (B, N, 2) → (B, 1, 1).
    timesteps = torch.tensor([[[0.9]]], dtype=torch.float32)
    T = 5
    feats = enc(timesteps[..., -1])
    assert feats.shape == (1, 1, 4)
    repeated = feats.repeat(1, T, 1)
    assert repeated.shape == (1, T, 4)
    torch.testing.assert_close(repeated[:, 0], repeated[:, -1])
    other = enc(torch.tensor([[[0.1]]], dtype=torch.float32)[..., -1])
    assert not torch.allclose(repeated[:, 0], other[:, 0])
    # Fourier is [sin, cos] * √2, so t=0 is zeros then √2.
    zero = enc(torch.zeros(1, 1))
    half = 2
    torch.testing.assert_close(zero[..., :half], torch.zeros(1, 1, half))
    torch.testing.assert_close(zero[..., half:], torch.full((1, 1, half), math.sqrt(2.0)))
