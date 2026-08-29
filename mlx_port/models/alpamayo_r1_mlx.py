# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AlpamayoR1 model implementation for MLX.

Complete custom from_pretrained that mirrors the full NVIDIA AlpamayoR1.__init__ structure
using MLX-native modules.
"""

import json
import os
from glob import glob
from typing import Any

import mlx.core as mx

from mlx_port.profiling import is_profiling_enabled, StepProfiler
import mlx.nn as nn
from safetensors import safe_open

from mlx_port.vlm_loader import load_vlm_with_alpamayo_tokens


class AlpamayoPatchEmbed(nn.Module):
    """
    Custom PatchEmbed for Alpamayo fine-tuned vision weights.

    The Alpamayo Conv3D weight was trained expecting channels-first input
    (N, C, T, H, W). mlx_vlm's default PatchEmbed forces channels-last via
    .moveaxis(1, 4). This class bypasses that and calls mx.conv3d directly
    on the channels-first layout produced by our preprocessing.
    """

    def __init__(
        self,
        proj: nn.Conv3d,
        in_channels: int | None = None,
        temporal_patch_size: int | None = None,
        patch_size: int | None = None,
    ):
        super().__init__()
        self.proj = proj
        # Derive dimensions from the upstream PatchEmbed / Conv3d when available.
        # This removes hard-coded magic numbers and respects the configuration
        # that was used to create the original (and now loaded) weight.
        w_shape = proj.weight.shape  # observed MLX layout: (out, kT, kH, kW, in) or (out, in, kT, kH, kW)
        self.in_channels = in_channels or (w_shape[4] if len(w_shape) == 5 and w_shape[4] == 3 else w_shape[1])
        self.temporal_patch_size = temporal_patch_size or (w_shape[1] if len(w_shape) == 5 else w_shape[2])
        self.patch_size = patch_size or (w_shape[2] if len(w_shape) == 5 else w_shape[3])
        self.hidden_size = w_shape[0]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # hidden_states is expected to arrive as (N_patches, C, T, H, W) channels-first
        # from our corrected preprocessing. Dimensions are taken from the upstream
        # PatchEmbed configuration so we never hard-code 3/2/16.
        hidden_states = hidden_states.reshape(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states)  # direct channels-first conv
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        return hidden_states
from mlx_port.models.token_utils_mlx import fuse_traj_tokens as fuse_traj_tokens_util
from mlx_port.models.action_in_proj_mlx import PerWaypointActionInProjV2
from mlx_port.models.action_space_utils_mlx import (
    solve_xs_eq_y,
    dxy_theta_to_v,
    dxy_theta_to_v_without_v0,
    theta_smooth,
    unwrap_angle,
)


# =============================================================================
# Diffusion Expert
# =============================================================================

class ExpertDecoderLayer(nn.Module):
    """Simplified Qwen-style decoder layer for the diffusion expert."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.self_attn = nn.MultiHeadAttention(dims=hidden_size, num_heads=num_heads, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

    def __call__(self, x, mask=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ExpertDecoder(nn.Module):
    """Stack of decoder layers for the diffusion expert (expert_cfg)."""

    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, intermediate_size: int):
        super().__init__()
        self.layers = [
            ExpertDecoderLayer(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


# =============================================================================
# Action Projection Modules
# =============================================================================

# Real checkpoint module. Old stub was Sequential Linear(in_dims=5) and did
# not match PerWaypointActionInProjV2 keys (Fourier + MLPEncoder + LayerNorm).
ActionInProj = PerWaypointActionInProjV2


class ActionOutProj(nn.Linear):
    """NVIDIA ``action_out_proj`` is ``torch.nn.Linear(expert_hidden, action_dim)``.

    Checkpoint keys are ``action_out_proj.weight`` / ``.bias`` (no ``.linear``).
    """


# =============================================================================
# Diffusion and Action Space (Row 7 – MLX ports)
# =============================================================================

class FlowMatching(nn.Module):
    """MLX port of FlowMatching (Euler integration for flow matching).

    This implements the exact sampling loop from NVIDIA's FlowMatching._euler:
        x = randn(...)
        for i in range(num_steps):
            t = time_steps[i]
            v = step_fn(x=x, t=t)
            x = x + dt * v
    """

    def __init__(
        self,
        x_dims: Any = None,
        int_method: str = "euler",
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.x_dims = [x_dims] if isinstance(x_dims, int) else list(x_dims) if x_dims else []
        self.int_method = int_method
        self.num_inference_steps = num_inference_steps

    def sample(
        self,
        batch_size: int,
        step_fn: Any,
        return_all_steps: bool = False,
        inference_step: int | None = None,
        **kwargs: Any,
    ) -> mx.array:
        """Run Euler integration using the provided step_fn.

        Args:
            batch_size: Number of trajectories to sample.
            step_fn: Callable with signature step_fn(x: mx.array, t: mx.array) -> mx.array
            return_all_steps: If True, return all intermediate states.
            inference_step: Override for number of integration steps.

        Returns:
            mx.array of shape [batch_size, *x_dims] (final sample) or
            tuple (all_steps, time_steps) if return_all_steps=True.
        """
        if self.int_method != "euler":
            raise ValueError(f"Only 'euler' integration is supported, got {self.int_method}")

        n_steps = inference_step or self.num_inference_steps
        x_dims = self.x_dims or [5]  # fallback if not provided

        # Initialize noise
        shape = (batch_size, *x_dims)
        x = mx.random.normal(shape=shape)

        time_steps = mx.linspace(0.0, 1.0, n_steps + 1)

        # Row 12 – profiling for diffusion sampling
        diffusion_profiler = StepProfiler(
            enabled=is_profiling_enabled(),
            name="Diffusion"
        )

        if return_all_steps:
            all_steps = [x]

        for i in range(n_steps):
            diffusion_profiler.step_start(i)
            dt = time_steps[i + 1] - time_steps[i]
            # Broadcast dt and t to match x shape
            t_start = mx.full(shape, float(time_steps[i]))
            v = step_fn(x=x, t=t_start)
            x = x + dt * v
            mx.eval(x)
            if i % 3 == 0:          # every 3 diffusion steps
                mx.clear_cache()
            diffusion_profiler.step_end()
            if return_all_steps:
                all_steps.append(x)

        diffusion_profiler.summary()

        if return_all_steps:
            return mx.stack(all_steps, axis=1), time_steps
        return x


class ActionSpace:
    """MLX port of ActionSpace (stores unicycle params + conversion).

    For Row 7 we only need get_action_space_dims() and a stub for action_to_traj.
    Full implementation will be added when we need real trajectory conversion.
    """

    def __init__(
        self,
        accel_mean: float = 0.0,
        accel_std: float = 1.0,
        curvature_mean: float = 0.0,
        curvature_std: float = 1.0,
        dt: float = 0.1,
        n_waypoints: int = 8,
        theta_lambda: float = 1e-6,
        theta_ridge: float = 1e-8,
        v_lambda: float = 1e-6,
        v_ridge: float = 1e-4,
        **_unused,
    ):
        self.accel_mean = accel_mean
        self.accel_std = accel_std
        self.curvature_mean = curvature_mean
        self.curvature_std = curvature_std
        self.dt = dt
        self.n_waypoints = n_waypoints
        # Same defaults as UnicycleAccelCurvatureActionSpace / config.json.
        self.theta_lambda = theta_lambda
        self.theta_ridge = theta_ridge
        self.v_lambda = v_lambda
        self.v_ridge = v_ridge

    def get_action_space_dims(self) -> tuple[int, ...]:
        """Return the shape expected by the diffusion expert.

        In the NVIDIA implementation this is typically (n_future_tokens, 2)
        for the UnicycleAccelCurvatureActionSpace (accel, curvature).
        """
        return (self.n_waypoints, 2)

    def estimate_t0_states(
        self,
        traj_history_xyz: mx.array,
        traj_history_rot: mx.array,
    ) -> dict[str, mx.array]:
        """NVIDIA ``estimate_t0_states``: last-history speed via ``without_v0``."""
        xyz = mx.array(traj_history_xyz)
        rot = mx.array(traj_history_rot)
        dxy = xyz[..., 1:, :2] - xyz[..., :-1, :2]
        yaw = unwrap_angle(mx.arctan2(rot[..., 1, 0], rot[..., 0, 0]))
        v = dxy_theta_to_v_without_v0(
            dxy,
            yaw,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )
        return {"v": v[..., -1]}

    def action_to_traj(
        self,
        action: mx.array,
        traj_history_xyz: mx.array,
        traj_history_rot: mx.array,
        dt: float = 0.1,
        t0_states: dict | None = None,
    ) -> tuple[mx.array, mx.array]:
        """MLX port of UnicycleAccelCurvatureActionSpace.action_to_traj.

        Converts normalized actions (accel, curvature) into future xyz/rot
        trajectories using a discrete unicycle kinematic model.

        This matches the core integration logic from the NVIDIA implementation.
        """
        # action: (..., T, 2)
        accel = action[..., 0]
        kappa = action[..., 1]

        # Denormalize (using stored stats; defaults are identity)
        accel = accel * self.accel_std + self.accel_mean
        kappa = kappa * self.curvature_std + self.curvature_mean

        if t0_states is None or "v" not in t0_states:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        v0 = mx.array(t0_states["v"])
        v0 = mx.broadcast_to(v0, accel.shape[:-1])

        # Simple Euler integration (velocity, theta, position)
        dt_2 = 0.5 * dt**2
        velocity = mx.concatenate(
            [
                v0[..., None],
                v0[..., None] + mx.cumsum(accel * dt, axis=-1),
            ],
            axis=-1,
        )

        theta = mx.concatenate(
            [
                mx.zeros_like(v0)[..., None],
                mx.cumsum(kappa * velocity[..., :-1] * dt, axis=-1)
                + mx.cumsum(kappa * accel * dt_2, axis=-1),
            ],
            axis=-1,
        )

        half_dt = 0.5 * dt
        x = mx.cumsum(
            velocity[..., :-1] * mx.cos(theta[..., :-1]) * half_dt
            + velocity[..., 1:] * mx.cos(theta[..., 1:]) * half_dt,
            axis=-1,
        )
        y = mx.cumsum(
            velocity[..., :-1] * mx.sin(theta[..., :-1]) * half_dt
            + velocity[..., 1:] * mx.sin(theta[..., 1:]) * half_dt,
            axis=-1,
        )

        # Build future xyz (z = last history z)
        batch_shape = traj_history_xyz.shape[:-2]
        n_future = x.shape[-1]
        traj_future_xyz = mx.zeros((*batch_shape, n_future, 3))
        traj_future_xyz[..., 0] = x
        traj_future_xyz[..., 1] = y
        # z from last history
        last_z = traj_history_xyz[..., -1:, 2]
        traj_future_xyz[..., 2] = mx.broadcast_to(last_z, traj_future_xyz[..., 2].shape)

        # Simple rotation matrix from theta (2D yaw → 3D rotation)
        c = mx.cos(theta[..., 1:])
        s = mx.sin(theta[..., 1:])
        # 3x3 rotation matrices (yaw only)
        rot = mx.stack(
            [
                mx.stack([c, -s, mx.zeros_like(c)], axis=-1),
                mx.stack([s, c, mx.zeros_like(s)], axis=-1),
                mx.stack(
                    [mx.zeros_like(c), mx.zeros_like(c), mx.ones_like(c)], axis=-1
                ),
            ],
            axis=-2,
        )
        traj_future_rot = rot

        return traj_future_xyz, traj_future_rot

    def traj_to_action(
        self,
        traj_history_xyz: mx.array,
        traj_history_rot: mx.array,
        traj_future_xyz: mx.array,
        traj_future_rot: mx.array,
        t0_states: dict | None = None,
        output_all_states: bool = False,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Real MLX implementation of traj_to_action using the numerical solvers
        from action_space_utils_mlx.py.

        This version aims for numerical correctness by using:
        - theta_smooth (for heading extraction + smoothing)
        - dxy_theta_to_v (for velocity recovery)
        - solve_xs_eq_y (for regularized acceleration recovery)

        The implementation follows the same mathematical structure as
        NVIDIA's UnicycleAccelCurvatureActionSpace.traj_to_action.
        """
        # Ensure inputs are MLX arrays (dataset loader may return numpy arrays)
        traj_history_xyz = mx.array(traj_history_xyz)
        traj_history_rot = mx.array(traj_history_rot)
        traj_future_xyz = mx.array(traj_future_xyz)
        traj_future_rot = mx.array(traj_future_rot)

        if t0_states is None or "v" not in t0_states:
            t0_states = self.estimate_t0_states(traj_history_xyz, traj_history_rot)
        v0 = t0_states["v"]

        # 1. Build full path and differences
        full_xy = mx.concatenate(
            [traj_history_xyz[..., -1:, :2], traj_future_xyz[..., :2]], axis=-2
        )
        dxy = full_xy[..., 1:, :] - full_xy[..., :-1, :]

        # 2. Heading smoothing. NVIDIA theta_smooth returns (..., T+1) with
        # theta[..., 0] pinned to 0 via solve_single_constraint.
        theta = theta_smooth(
            traj_future_rot,
            dt=self.dt,
            theta_lambda=self.theta_lambda,
            theta_ridge=self.theta_ridge,
        )

        # 3. Velocity recovery (v has length n_future + 1)
        # Pass the full theta (length N+1) so dxy_theta_to_v can use mid-point headings
        v = dxy_theta_to_v(
            dxy,
            theta,
            v0,
            dt=self.dt,
            v_lambda=self.v_lambda,
            v_ridge=self.v_ridge,
        )

        # 4. Acceleration via real regularized solver (solve_xs_eq_y)
        # y must be acceleration (dv/dt), not velocity difference, so the data term
        # in solve_xs_eq_y recovers the correct physical units.
        y = (v[..., 1:] - v[..., :-1]) / self.dt
        s = mx.ones_like(y)
        accel = solve_xs_eq_y(
            s=s,
            y=y,
            w_smooth2=1e-6,  # minimal smoothing – keeps 1e-6 fidelity for constant-accel case
            lam=1e-4,
            ridge=1e-4,
            dt=self.dt,
        )

        # 5. Curvature recovery (now lengths match: theta has len n+1, v has len n+1)
        n = accel.shape[-1]
        s = v[..., :n] * self.dt
        dtheta = theta[..., 1 : n + 1] - theta[..., :n]
        kappa = dtheta / (s + 1e-8)

        # 6. Normalize to match training distribution
        accel = (accel - self.accel_mean) / self.accel_std
        kappa = (kappa - self.curvature_mean) / self.curvature_std

        # Ensure final action has exactly the expected shape (B, n_waypoints, 2)
        expected_n = self.n_waypoints
        if accel.shape[-1] != expected_n:
            # Pad or truncate to expected length
            if accel.shape[-1] < expected_n:
                pad = mx.zeros((*accel.shape[:-1], expected_n - accel.shape[-1]))
                accel = mx.concatenate([accel, pad], axis=-1)
                kappa = mx.concatenate([kappa, pad], axis=-1)
            else:
                accel = accel[..., :expected_n]
                kappa = kappa[..., :expected_n]

        action = mx.stack([accel, kappa], axis=-1)

        if output_all_states:
            states = mx.stack([v[..., :-1], accel, theta[..., :-1]], axis=-1)
            return action, states
        return action


# =============================================================================
# Main Model
# =============================================================================

class AlpamayoR1MLX(nn.Module):
    """MLX-native Alpamayo-R1 model (complete row 5 implementation)."""

    def __init__(
        self,
        vlm: nn.Module,
        expert: nn.Module | None = None,
        action_in_proj: nn.Module | None = None,
        action_out_proj: nn.Module | None = None,
        diffusion: Any = None,
        action_space: Any = None,
        # Token fusion attributes (for Row 6)
        hist_traj_tokenizer: Any = None,
        hist_token_start_idx: int = 0,
        traj_token_ids: dict[str, int] | None = None,
    ):
        super().__init__()
        self.vlm = vlm
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj
        self.diffusion = diffusion
        self.action_space = action_space

        # Token fusion (Row 6)
        self.hist_traj_tokenizer = hist_traj_tokenizer
        self.hist_token_start_idx = hist_token_start_idx
        self.traj_token_ids = traj_token_ids or {}

    @classmethod
    def from_pretrained(
        cls,
        alpamayo_path: str,
        vlm_model_path: str | None = None,
        load_expert: bool = True,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "AlpamayoR1MLX":
        """Custom from_pretrained mirroring NVIDIA AlpamayoR1.__init__.

        Args:
            alpamayo_path: Path to the Alpamayo-R1-10B checkpoint (for expert weights).
            vlm_model_path: Path to the base Qwen3-VL-8B-Instruct checkpoint.
                            Defaults to the local copy used throughout the port.
        """
        from mlx_port.vlm_loader import LOCAL_QWEN_PROCESSOR_PATH

        vlm_path = vlm_model_path or LOCAL_QWEN_PROCESSOR_PATH

        alpamayo_cfg_path = os.path.join(alpamayo_path, "config.json")
        if not os.path.isfile(alpamayo_cfg_path):
            raise FileNotFoundError(
                f"Alpamayo checkpoint config.json not found at {alpamayo_cfg_path}."
            )
        with open(alpamayo_cfg_path) as f:
            alpamayo_cfg = json.load(f)

        # NVIDIA _build_processor adds all traj_vocab_size <iN> tokens first,
        # then SPECIAL_TOKENS. Doing this in two phases (768, then specials,
        # then the rest) maps <|prompt_start|> onto checkpoint row <i768>.
        declared_traj_vocab_size = alpamayo_cfg.get("traj_vocab_size", 4000)

        # 1. Load VLM + Alpamayo tokens from the base Qwen checkpoint
        vlm, processor = load_vlm_with_alpamayo_tokens(
            vlm_path, traj_vocab_size=declared_traj_vocab_size
        )

        # 1.5 Load the fine-tuned VLM weights from the Alpamayo checkpoint safetensors.
        # This replaces the base-Qwen weights with Alpamayo's learned CoC / driving weights
        # (including proper embeddings for <|cot_start|>, <|traj_future_start|>, and <iN> tokens).
        cls._load_vlm_weights(vlm, alpamayo_path, dtype=dtype)

        # ------------------------------------------------------------------
        # Replace the vision tower's PatchEmbed with an Alpamayo-aware version.
        # The Alpamayo fine-tuned Conv3D weight expects channels-first input.
        # mlx_vlm's default PatchEmbed unconditionally does .moveaxis(1,4) to
        # channels-last, which breaks the loaded weight. We swap in a minimal
        # custom module that performs a direct channels-first conv3d.
        # ------------------------------------------------------------------
        original_pe = vlm.vision_tower.patch_embed
        vlm.vision_tower.patch_embed = AlpamayoPatchEmbed(
            original_pe.proj,
            in_channels=original_pe.in_channels,
            temporal_patch_size=original_pe.temporal_patch_size,
            patch_size=original_pe.patch_size,
        )

        # ------------------------------------------------------------------
        # Replace base instances with Alpamayo subclass instances via the
        # explicit from_existing API (proper instantiation entry point).
        # ------------------------------------------------------------------
        from mlx_port.models.alpamayo_qwen3vl import AlpamayoModel, AlpamayoLanguageModel

        vlm.language_model = AlpamayoLanguageModel.from_existing(vlm.language_model)
        vlm = AlpamayoModel.from_existing(vlm)

        print("[AlpamayoR1MLX] Instantiated AlpamayoLanguageModel and AlpamayoModel via from_existing")

        # Note: mlx-vlm Model objects do not expose .astype(); we rely on the
        # checkpoint being in the correct dtype (bfloat16). The final model.astype(dtype)
        # below will cast the expert/action modules we own.

        # 2. Expert is mlx_vlm Qwen3-VL attention (mRoPE + explicit position_ids / mask).
        # Do not use mlx_lm Qwen3 here — it RoPEs at cache.offset and ignores NVIDIA's ids.
        expert = None
        if load_expert:
            from mlx_port.models.expert_mlx import (
                AlpamayoExpert,
                text_config_from_vlm_and_overrides,
            )

            if not hasattr(vlm, "config") or not hasattr(vlm.config, "text_config"):
                raise RuntimeError(
                    "Loaded VLM does not expose config.text_config. Cannot derive expert architecture."
                )

            expert_overrides = alpamayo_cfg.get("expert_cfg", {})
            if not expert_overrides:
                raise ValueError(
                    "Alpamayo config.json does not contain 'expert_cfg'. "
                    "Cannot determine expert architecture."
                )

            expert_text = text_config_from_vlm_and_overrides(
                vlm.config.text_config, expert_overrides
            )
            expert = AlpamayoExpert(expert_text)

        # 3. Create action projections using dimensions from the Alpamayo checkpoint config
        # (expert_cfg.hidden_size is the authoritative source, matching the expert we just built)
        expert_cfg = alpamayo_cfg.get("expert_cfg", {})
        expert_hidden_size = expert_cfg.get("hidden_size", 2048)

        # 4. Diffusion and action space. Checkpoint Unicycle uses n_waypoints=64
        # so x_dims is (64, 2). Do not pop n_waypoints — that kept diffusion at (8, 2).
        as_cfg = dict(alpamayo_cfg.get("action_space_cfg") or {})
        as_cfg.pop("_target_", None)
        action_space = ActionSpace(**as_cfg)
        x_dims = action_space.get_action_space_dims()
        diffusion = FlowMatching(x_dims=x_dims, num_inference_steps=10)

        proj_cfg = dict(alpamayo_cfg.get("action_in_proj_cfg") or {})
        proj_cfg.pop("_target_", None)
        action_in_proj = ActionInProj(
            in_dims=x_dims,
            out_dim=expert_hidden_size,
            **proj_cfg,
        )
        action_out_proj = ActionOutProj(expert_hidden_size, x_dims[-1])

        # 4.5 History tokenizer (Row 6). NVIDIA hist_traj_tokenizer_cfg is
        # DeltaTrajectoryTokenizer: 16 steps × xyz = 48 tokens. Discrete action
        # bins are the *future* tokenizer, not history fusion.
        from mlx_port.models.trajectory_tokenizer_mlx import DeltaTrajectoryTokenizerMLX

        hist_cfg = alpamayo_cfg.get("hist_traj_tokenizer_cfg") or {}
        hist_kwargs = {k: v for k, v in hist_cfg.items() if k != "_target_"}
        hist_traj_tokenizer = DeltaTrajectoryTokenizerMLX(**hist_kwargs)

        model = cls(
            vlm=vlm,
            expert=expert,
            action_in_proj=action_in_proj,
            action_out_proj=action_out_proj,
            diffusion=diffusion,
            action_space=action_space,
        )
        # Expose tokenizer and processor for convenience (mirrors NVIDIA AlpamayoR1)
        model.tokenizer = processor.tokenizer
        model.processor = processor  # Store the full wrapper for mlx_vlm.generate/stream_generate

        # 5. Load all weights
        if load_expert:
            cls._load_expert_and_action_weights(model, alpamayo_path, dtype=dtype)

        # 6. Attach token fusion attributes from the processor (Row 6)
        tokenizer = processor.tokenizer
        declared_traj_start = alpamayo_cfg.get("traj_token_start_idx", 151669)

        if tokenizer.traj_token_start_idx != declared_traj_start:
            raise RuntimeError(
                f"Tokenizer <i0> id is {tokenizer.traj_token_start_idx}, "
                f"but Alpamayo config traj_token_start_idx is {declared_traj_start}. "
                "Token add-order does not match the checkpoint embedding layout."
            )
        cfg_traj_ids = alpamayo_cfg.get("traj_token_ids", {})
        for name, expected_id in cfg_traj_ids.items():
            actual_id = tokenizer.traj_token_ids.get(name)
            if actual_id != expected_id:
                raise RuntimeError(
                    f"Tokenizer traj_token_ids[{name!r}] is {actual_id}, "
                    f"but Alpamayo config has {expected_id}."
                )

        tokenizer.hist_traj_tokenizer = hist_traj_tokenizer
        tokenizer.hist_token_start_idx = tokenizer.traj_token_start_idx

        model.hist_traj_tokenizer = hist_traj_tokenizer
        model.hist_token_start_idx = tokenizer.traj_token_start_idx
        model.traj_token_ids = dict(tokenizer.traj_token_ids)
        model.traj_token_start_idx = tokenizer.traj_token_start_idx
        model.traj_vocab_size = declared_traj_vocab_size
        model.tokens_per_future_traj = alpamayo_cfg.get("tokens_per_future_traj", 32)
        model.tokens_per_history_traj = alpamayo_cfg.get("tokens_per_history_traj", 48)

        # Contiguous after the NVIDIA add order: <i0> .. <i{N-1}> then specials.
        traj_token_id_list = [
            tokenizer.convert_tokens_to_ids(f"<i{i}>")
            for i in range(declared_traj_vocab_size)
        ]
        model.traj_token_id_list = traj_token_id_list

        print(
            f"[AlpamayoR1MLX] tokenizer layout: vocab={len(tokenizer)} "
            f"i0={tokenizer.traj_token_start_idx} "
            f"i{declared_traj_vocab_size - 1}={tokenizer.traj_token_end_idx} "
            f"cot_start={tokenizer.convert_tokens_to_ids('<|cot_start|>')} "
            f"traj_future_start={tokenizer.traj_token_ids.get('future_start')} "
            f"hist_tok={type(hist_traj_tokenizer).__name__}"
        )

        # Final dtype enforcement for the whole model (Row 10)
        # Note: mlx.nn.Module.astype is not available on custom top-level modules
        # that contain non-mlx-vlm submodules. The VLM and expert weights are already
        # loaded in the requested dtype, so we skip the top-level cast here.
        # If needed later, we can implement a recursive _apply_dtype helper.

        return model

    @staticmethod
    def _load_vlm_weights(vlm: Any, alpamayo_path: str, dtype: mx.Dtype = mx.bfloat16) -> None:
        """Load all vlm.* weights from the Alpamayo checkpoint into the given mlx_vlm model.

        This achieves parity with the NVIDIA implementation where the fine-tuned
        Alpamayo VLM weights (language_model + vision_tower) are loaded directly
        from the Alpamayo safetensors via the "vlm." prefix.

        Remapping rules:
          vlm.model.language_model.*  -> language_model.model.*
          vlm.model.visual.*          -> vision_tower.*
        """
        import torch
        from safetensors import safe_open
        from glob import glob

        shard_files = sorted(glob(os.path.join(alpamayo_path, "model-*.safetensors")))
        if not shard_files:
            print("[AlpamayoR1MLX] No safetensors shards found; skipping VLM weight load")
            return

        total_vlm_keys = 0
        loaded = 0

        def _get_nested(obj, parts):
            """Navigate through attributes and list indices."""
            current = obj
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    current = current[idx] if isinstance(current, (list, tuple)) else getattr(current, f"layers")[idx] if hasattr(current, "layers") else current
                else:
                    current = getattr(current, part)
            return current

        def _assign(arr: mx.array, dotted: str) -> bool:
            """Navigate the module tree (supporting list indices) and assign the array."""
            parts = dotted.split(".")
            try:
                parent_parts = parts[:-1]
                last = parts[-1]
                parent = vlm
                for p in parent_parts:
                    if p.isdigit():
                        idx = int(p)
                        if isinstance(parent, (list, tuple)):
                            parent = parent[idx]
                        elif hasattr(parent, "layers"):
                            parent = parent.layers[idx]
                        elif hasattr(parent, "blocks"):
                            parent = parent.blocks[idx]
                        else:
                            parent = parent[idx]
                    else:
                        parent = getattr(parent, p)
                setattr(parent, last, arr)
                return True
            except (AttributeError, IndexError, TypeError) as e:
                return False

        for shard in shard_files:
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if not key.startswith("vlm."):
                        continue
                    total_vlm_keys += 1
                    suffix = key[4:]

                    new_key = None
                    if suffix.startswith("model.language_model."):
                        inner = suffix[len("model.language_model."):]
                        new_key = f"language_model.model.{inner}"
                    elif suffix.startswith("model.visual."):
                        inner = suffix[len("model.visual."):]
                        new_key = f"vision_tower.{inner}"
                    elif suffix.startswith("lm_head"):
                        # Language modeling head lives under language_model in mlx_vlm
                        inner = suffix[len("lm_head"):]  # e.g. ".weight"
                        new_key = f"language_model.lm_head{inner}"
                    else:
                        continue

                    t = f.get_tensor(key)
                    if t.dtype == torch.bfloat16:
                        t = t.to(torch.float32)
                    arr = mx.array(t.cpu().numpy())
                    if dtype == mx.bfloat16:
                        arr = arr.astype(mx.bfloat16)
                    elif arr.dtype != dtype:
                        arr = arr.astype(dtype)

                    if _assign(arr, new_key):
                        loaded += 1

        print(f"[AlpamayoR1MLX] Loaded {loaded}/{total_vlm_keys} VLM weights from Alpamayo checkpoint (fine-tuned CoC + vision head)")

    @staticmethod
    def _load_expert_and_action_weights(
        model: "AlpamayoR1MLX",
        alpamayo_path: str,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> None:
        """Load expert + action projection weights from safetensors.

        Uses torch as an intermediate loader because the safetensors files contain
        bfloat16 weights and `safe_open(framework="mlx")` currently raises
        "TypeError: data type 'bfloat16' not understood" on this system.
        Torch CPU understands bfloat16 and can hand the arrays to MLX.
        """
        import torch  # torch is available in the dev venv; used only for weight loading

        shard_files = sorted(glob(os.path.join(alpamayo_path, "model-*.safetensors")))
        if not shard_files:
            raise FileNotFoundError(
                f"No safetensors shards (model-*.safetensors) found in {alpamayo_path}. "
                "AlpamayoR1MLX.from_pretrained requires the full Alpamayo-R1-10B checkpoint."
            )

        state_dict = {}
        for shard in shard_files:
            # torch.load on safetensors via the safetensors.torch helper or plain torch
            # For .safetensors we use safe_open with framework="pt" (torch) which works reliably.
            with safe_open(shard, framework="pt") as f:
                for key in f.keys():
                    if key.startswith(("expert.", "action_in_proj.", "action_out_proj.")):
                        t = f.get_tensor(key)  # torch.Tensor (possibly bfloat16)
                        # Torch CPU builds often cannot .numpy() bfloat16 directly.
                        # Upcast to float32 first (safe, lossless for values), then convert.
                        if t.dtype == torch.bfloat16:
                            t = t.to(torch.float32)
                        arr = t.cpu().numpy()
                        mx_arr = mx.array(arr)
                        if dtype == mx.bfloat16:
                            # Preserve the original checkpoint precision
                            mx_arr = mx_arr.astype(mx.bfloat16)
                        elif mx_arr.dtype != dtype:
                            mx_arr = mx_arr.astype(dtype)
                        state_dict[key] = mx_arr

        # Remap keys for load_weights
        # The expert is a qwen3_vl wrapper, so its parameters live under
        # expert.language_model.model.layers.* (see tree_flatten output).
        # NVIDIA stores them under expert.* prefix in the checkpoint.
        remapped = []
        for k, v in state_dict.items():
            if k.startswith("expert."):
                # expert.layers.0.xxx -> expert.language_model.model.layers.0.xxx
                new_k = "expert.language_model.model." + k[len("expert."):]
                remapped.append((new_k, v))
            elif k.startswith("action_in_proj."):
                new_k = "action_in_proj." + k[len("action_in_proj."):]
                remapped.append((new_k, v))
            elif k.startswith("action_out_proj."):
                new_k = "action_out_proj." + k[len("action_out_proj."):]
                remapped.append((new_k, v))

        model.load_weights(remapped, strict=False)
        print(f"[AlpamayoR1MLX] Loaded {len(remapped)} weights from Alpamayo-R1-10B checkpoint")

    def fuse_traj_tokens(self, input_ids: mx.array, traj_data: dict[str, Any] | None = None) -> mx.array:
        """Fuse history trajectory tokens into the input sequence.

        This is the Row 6 integration point. It calls the MLX-native
        fusion utility using the tokenizer attributes attached during
        from_pretrained.
        """
        if self.hist_traj_tokenizer is None or not self.traj_token_ids:
            raise RuntimeError(
                "hist_traj_tokenizer or traj_token_ids is not set on the model. "
                "This should have been attached in from_pretrained(). "
                "Trajectory token fusion requires a properly initialized Alpamayo tokenizer."
            )

        return fuse_traj_tokens_util(
            input_ids=input_ids,
            traj_data=traj_data,
            hist_traj_tokenizer=self.hist_traj_tokenizer,
            hist_token_start_idx=self.hist_token_start_idx,
            traj_token_ids=self.traj_token_ids,
        )