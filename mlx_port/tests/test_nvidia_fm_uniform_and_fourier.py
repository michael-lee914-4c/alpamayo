"""FM uniform sampler (no 0.999 cap) and FourierEncoderV2 log-freqs.

Weight-free. Complementary to open #25 (interp/MSE/Euler) and #28 (beta
``t=0.999*(1-u)``) — those never locked the uniform branch, the invalid
sampler/int-method guards, or the log-spaced ``[sin, cos] * √2`` features
that feed every expert action token.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_flow_matching():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.diffusion.flow_matching import FlowMatching
    except Exception as exc:
        pytest.skip(f"NVIDIA FlowMatching unavailable: {exc}")
    return FlowMatching


def _load_fourier():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.models.action_in_proj import FourierEncoderV2
    except Exception as exc:
        pytest.skip(f"NVIDIA FourierEncoderV2 unavailable: {exc}")
    return FourierEncoderV2


def test_uniform_sampler_is_unscaled_unit_interval(torch, monkeypatch):
    FlowMatching = _load_flow_matching()
    fm = FlowMatching(x_dims=(2,), train_timestep_sampler="uniform")

    def _dirac(shape, device=None):
        return torch.full(shape, 0.2, device=device)

    monkeypatch.setattr(torch, "rand", _dirac)
    data = fm.construct_training_data(torch.ones(4, 2))
    t = data["timesteps"].reshape(4)
    assert torch.allclose(t, torch.full((4,), 0.2))
    # Uniform must not apply the beta 0.999 flip/cap.
    assert float(t.max()) == pytest.approx(0.2)
    assert not hasattr(fm, "beta_scale_constant")


def test_invalid_sampler_and_int_method_raise(torch):
    FlowMatching = _load_flow_matching()
    fm = FlowMatching(x_dims=(2,), train_timestep_sampler="uniform")
    fm.train_timestep_sampler = "nope"
    try:
        fm.construct_training_data(torch.ones(2, 2))
    except ValueError as exc:
        assert "time sampler" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown time sampler")

    fm.train_timestep_sampler = "uniform"
    try:
        fm.sample(batch_size=1, step_fn=lambda *, x, t: x, int_method="rk4")
    except ValueError as exc:
        assert "integration method" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-euler integrator")


def test_fourier_encoder_v2_is_logspace_sin_cos_sqrt2(torch):
    FourierEncoderV2 = _load_fourier()
    enc = FourierEncoderV2(dim=8, max_freq=100.0)
    assert enc.freqs.shape == (1, 4)
    assert float(enc.freqs[0, 0].item()) == pytest.approx(1.0)
    assert float(enc.freqs[0, -1].item()) == pytest.approx(100.0)
    # logspace(0, 2, 4) is geometric: 1, 100^(1/3), 100^(2/3), 100
    ratio = enc.freqs[0, 1:] / enc.freqs[0, :-1]
    assert torch.allclose(ratio, ratio[0].expand_as(ratio), atol=1e-5)

    out = enc(torch.zeros(3))
    assert out.shape == (3, 8)
    assert torch.allclose(out[:, :4], torch.zeros(3, 4), atol=1e-6)
    assert torch.allclose(out[:, 4:], torch.full((3, 4), math.sqrt(2.0)), atol=1e-5)
