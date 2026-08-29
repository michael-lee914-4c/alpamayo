"""Shape, key, and PyTorch numerical checks for PerWaypointActionInProjV2."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_port.models.action_in_proj_mlx import PerWaypointActionInProjV2
from mlx_port.models.alpamayo_r1_mlx import ActionOutProj, ActionSpace


CHECKPOINT_ACTION_IN_KEYS = {
    "encoder.trunk.0.bias",
    "encoder.trunk.0.weight",
    "encoder.trunk.2.weight",
    "encoder.trunk.3.bias",
    "encoder.trunk.3.weight",
    "encoder.trunk.5.weight",
    "encoder.trunk.6.bias",
    "encoder.trunk.6.weight",
    "norm.bias",
    "norm.weight",
    "sinus.0.freqs",
    "sinus.1.freqs",
    "timestep_fourier_encoder.freqs",
}


def _alpamayo_proj(**kwargs) -> PerWaypointActionInProjV2:
    cfg = dict(
        in_dims=(64, 2),
        out_dim=2048,
        num_enc_layers=2,
        hidden_size=512,
        max_freq=100.0,
        num_fourier_feats=20,
    )
    cfg.update(kwargs)
    return PerWaypointActionInProjV2(**cfg)


def test_action_space_dims_follow_n_waypoints():
    assert ActionSpace(n_waypoints=8).get_action_space_dims() == (8, 2)
    assert ActionSpace(n_waypoints=64).get_action_space_dims() == (64, 2)


def test_action_in_proj_parameter_names_match_checkpoint():
    proj = _alpamayo_proj()
    names = {k for k, _ in tree_flatten(proj.parameters())}
    missing = CHECKPOINT_ACTION_IN_KEYS - names
    extra = names - CHECKPOINT_ACTION_IN_KEYS
    assert not missing, f"MLX ActionInProj missing checkpoint keys: {sorted(missing)}"
    assert not extra, f"MLX ActionInProj extra keys vs checkpoint: {sorted(extra)}"


def test_action_in_proj_forward_shape():
    proj = _alpamayo_proj()
    x = mx.random.normal((2, 64, 2))
    t = mx.full((2, 1, 1), 0.3)
    out = proj(x, t)
    assert out.shape == (2, 64, 2048)
    mx.eval(out)


def test_action_out_proj_checkpoint_layout():
    layer = ActionOutProj(2048, 2)
    names = {k for k, _ in tree_flatten(layer.parameters())}
    assert names == {"weight", "bias"}
    assert layer.weight.shape == (2, 2048)


def test_action_in_proj_matches_pytorch():
    torch = pytest.importorskip("torch")
    from alpamayo_r1.models.action_in_proj import (
        PerWaypointActionInProjV2 as PTProj,
    )

    mx.random.seed(0)
    torch.manual_seed(0)
    mlx_proj = _alpamayo_proj()
    pt_proj = PTProj(
        in_dims=[64, 2],
        out_dim=2048,
        num_enc_layers=2,
        hidden_size=512,
        max_freq=100.0,
        num_fourier_feats=20,
    )
    pt_sd = pt_proj.state_dict()
    remapped = []
    for k, v in tree_flatten(mlx_proj.parameters()):
        if k not in pt_sd:
            pytest.fail(f"MLX key {k} not in PyTorch state_dict")
        remapped.append((k, mx.array(pt_sd[k].detach().cpu().float().numpy())))
    mlx_proj.load_weights(remapped, strict=True)

    x_np = np.random.RandomState(1).randn(2, 64, 2).astype(np.float32)
    t_np = np.full((2, 1, 1), 0.3, dtype=np.float32)
    pt_out = pt_proj(torch.from_numpy(x_np), torch.from_numpy(t_np)).detach().cpu().numpy()
    mlx_out = np.array(mlx_proj(mx.array(x_np), mx.array(t_np)))
    np.testing.assert_allclose(mlx_out, pt_out, rtol=1e-4, atol=1e-4)
