"""MLX port of NVIDIA ``PerWaypointActionInProjV2`` (action_in_proj.py)."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class FourierEncoderV2(nn.Module):
    """Log-spaced sine/cosine features. Matches NVIDIA ``FourierEncoderV2``."""

    def __init__(self, dim: int, max_freq: float = 100.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Fourier dim must be even, got {dim}")
        half = dim // 2
        freqs = np.logspace(0.0, math.log10(max_freq), num=half, dtype=np.float32)
        self.out_dim = dim
        # Checkpoint key: *.freqs (NVIDIA register_buffer).
        self.freqs = mx.array(freqs[None, :])  # (1, half)

    def __call__(self, x: mx.array) -> mx.array:
        arg = x[..., None] * self.freqs * (2.0 * math.pi)
        return mx.concatenate([mx.sin(arg), mx.cos(arg)], axis=-1) * math.sqrt(2.0)


class _SiLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class MLPEncoder(nn.Module):
    """NVIDIA ``MLPEncoder`` trunk layout (same Sequential indices)."""

    def __init__(self, num_input_feats: int, num_enc_layers: int, hidden_size: int, outdim: int):
        super().__init__()
        if num_enc_layers < 1:
            raise ValueError(f"num_enc_layers must be >= 1, got {num_enc_layers}")
        layers: list[nn.Module] = [nn.Linear(num_input_feats, hidden_size), _SiLU()]
        for layeri in range(num_enc_layers):
            if layeri < num_enc_layers - 1:
                layers.extend(
                    [
                        nn.RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, hidden_size),
                        _SiLU(),
                    ]
                )
            else:
                layers.extend(
                    [
                        nn.RMSNorm(hidden_size, eps=1e-5),
                        nn.Linear(hidden_size, outdim),
                    ]
                )
        self.trunk = layers

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.trunk:
            x = layer(x)
        return x


class PerWaypointActionInProjV2(nn.Module):
    """Project (B, T, C_action) + timestep → (B, T, hidden) for the expert."""

    def __init__(
        self,
        in_dims: tuple[int, ...] | list[int],
        out_dim: int,
        num_enc_layers: int = 4,
        hidden_size: int = 1024,
        max_freq: float = 100.0,
        num_fourier_feats: int = 20,
    ):
        super().__init__()
        self.in_dims = tuple(in_dims)
        self.out_dim = int(out_dim)
        action_dim = int(self.in_dims[-1])
        self.sinus = [
            FourierEncoderV2(dim=num_fourier_feats, max_freq=max_freq)
            for _ in range(action_dim)
        ]
        self.timestep_fourier_encoder = FourierEncoderV2(
            dim=num_fourier_feats, max_freq=max_freq
        )
        num_input_feats = action_dim * num_fourier_feats + num_fourier_feats
        self.encoder = MLPEncoder(
            num_input_feats=num_input_feats,
            num_enc_layers=num_enc_layers,
            hidden_size=hidden_size,
            outdim=out_dim,
        )
        self.norm = nn.LayerNorm(out_dim)

    def __call__(self, x: mx.array, timesteps: mx.array) -> mx.array:
        b, t, _ = x.shape
        x = x.astype(mx.float32)
        timesteps = timesteps.astype(mx.float32)
        action_feats = mx.concatenate(
            [enc(x[:, :, i]) for i, enc in enumerate(self.sinus)],
            axis=-1,
        )
        # NVIDIA: encoder(timesteps[..., -1]).repeat(1, T, 1)
        timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
        if timestep_feats.ndim == 2:
            timestep_feats = timestep_feats[:, None, :]
        if timestep_feats.shape[1] == 1 and t != 1:
            timestep_feats = mx.broadcast_to(timestep_feats, (b, t, timestep_feats.shape[-1]))
        feats = mx.concatenate([action_feats, timestep_feats], axis=-1)
        flat = feats.reshape((b * t, -1))
        return self.norm(self.encoder(flat)).reshape((b, t, -1))
