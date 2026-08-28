"""MLX port of NVIDIA's DiscreteTrajectoryTokenizer and DeltaTrajectoryTokenizer.

These are used to convert ego history/future trajectories into discrete tokens
for the VLM prompt and to decode actions back to trajectories.
"""

from typing import Any, Tuple

import mlx.core as mx
import numpy as np

from mlx_port.models.alpamayo_r1_mlx import ActionSpace


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


class DeltaTrajectoryTokenizerMLX:
    """MLX port of ``alpamayo_r1.models.delta_tokenizer.DeltaTrajectoryTokenizer``.

    History fusion uses this tokenizer: 16 waypoints × 3 xyz deltas = 48 tokens
    (``predict_yaw=False``, the Alpamayo-R1-10B default).
    """

    def __init__(
        self,
        ego_xyz_min: tuple[float, float, float] = (-4, -4, -10),
        ego_xyz_max: tuple[float, float, float] = (4, 4, 10),
        ego_yaw_min: float = float(-np.pi),
        ego_yaw_max: float = float(np.pi),
        num_bins: int = 1000,
        predict_yaw: bool = False,
    ):
        self.ego_xyz_min = np.asarray(ego_xyz_min, dtype=np.float64)
        self.ego_xyz_max = np.asarray(ego_xyz_max, dtype=np.float64)
        self.ego_yaw_min = float(ego_yaw_min)
        self.ego_yaw_max = float(ego_yaw_max)
        self.num_bins = int(num_bins)
        self._predict_yaw = bool(predict_yaw)

    @property
    def vocab_size(self) -> int:
        return self.num_bins

    def encode(
        self,
        hist_xyz: Any,
        hist_rot: Any,
        fut_xyz: Any,
        fut_rot: Any,
        hist_tstamp: Any = None,
        fut_tstamp: Any = None,
    ) -> mx.array:
        """Encode xyz (and optional yaw) deltas. ``hist_*`` is unused, matching NVIDIA."""
        del hist_xyz, hist_rot, hist_tstamp, fut_tstamp
        xyz = _to_numpy(fut_xyz).astype(np.float64, copy=False)
        # F.pad(fut_xyz, [0, 0, 1, 0, 0, 0]) — one zero waypoint prepended on T.
        xyz = np.pad(xyz, ((0, 0), (1, 0), (0, 0)))
        xyz = xyz[:, 1:] - xyz[:, :-1]
        xyz = (xyz - self.ego_xyz_min) / (self.ego_xyz_max - self.ego_xyz_min)
        xyz = np.clip(np.rint(xyz * (self.num_bins - 1)), 0, self.num_bins - 1).astype(np.int64)
        if not self._predict_yaw:
            return mx.array(xyz.reshape(xyz.shape[0], -1))

        fut_rot_np = _to_numpy(fut_rot).astype(np.float64, copy=False)
        yaw = np.arctan2(fut_rot_np[..., 0, 1], fut_rot_np[..., 0, 0])
        yaw_padded = np.pad(yaw, ((0, 0), (1, 0)))
        delta_yaw = yaw_padded[:, 1:] - yaw_padded[:, :-1]
        delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))
        delta_yaw = (delta_yaw - self.ego_yaw_min) / (self.ego_yaw_max - self.ego_yaw_min)
        delta_yaw = np.clip(
            np.rint(delta_yaw * (self.num_bins - 1)), 0, self.num_bins - 1
        ).astype(np.int64)
        xyzw = np.concatenate([xyz, delta_yaw[..., None]], axis=-1)
        return mx.array(xyzw.reshape(xyzw.shape[0], -1))

    def decode(
        self,
        hist_xyz: Any,
        hist_rot: Any,
        tokens: Any,
        hist_tstamp: Any = None,
    ) -> Tuple[mx.array, mx.array, Any]:
        del hist_tstamp
        m = 4 if self._predict_yaw else 3
        tokens_np = _to_numpy(tokens)
        b = tokens_np.shape[0]
        xyzw = tokens_np.reshape(b, -1, m).astype(np.float64)
        xyz = xyzw[..., :3] / (self.num_bins - 1)
        xyz = xyz * (self.ego_xyz_max - self.ego_xyz_min) + self.ego_xyz_min
        fut_xyz = np.cumsum(xyz, axis=1)
        if not self._predict_yaw:
            fut_rot = _yaw_rotation_matrices(fut_xyz)
            return mx.array(fut_xyz), mx.array(fut_rot), None
        yaw = xyzw[..., 3] / (self.num_bins - 1)
        yaw = yaw * (self.ego_yaw_max - self.ego_yaw_min) + self.ego_yaw_min
        yaw = np.cumsum(yaw, axis=1)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        zeros = np.zeros_like(cos_yaw)
        ones = np.ones_like(cos_yaw)
        fut_rot = np.stack(
            [
                np.stack([cos_yaw, -sin_yaw, zeros], axis=-1),
                np.stack([sin_yaw, cos_yaw, zeros], axis=-1),
                np.stack([zeros, zeros, ones], axis=-1),
            ],
            axis=-2,
        )
        return mx.array(fut_xyz), mx.array(fut_rot), None


def _yaw_rotation_matrices(trajectory: np.ndarray, window_size: int = 10, poly_order: int = 3):
    """Port of NVIDIA ``get_yaw_rotation_matrices`` (decode-only)."""
    B, N = trajectory.shape[:2]
    out = np.zeros((B, N, 3, 3), dtype=np.float64)
    for b in range(B):
        traj_batch = trajectory[b]
        for i in range(N):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(N, start_idx + window_size)
            if end_idx - start_idx < window_size:
                start_idx = max(0, end_idx - window_size)
            window_points = traj_batch[start_idx:end_idx]
            t = np.arange(len(window_points))
            x_coeffs = np.polyfit(t, window_points[:, 0], poly_order)
            y_coeffs = np.polyfit(t, window_points[:, 1], poly_order)
            center_t = min(i - start_idx, window_size - 1)
            dx = np.polyval(np.polyder(x_coeffs), center_t)
            dy = np.polyval(np.polyder(y_coeffs), center_t)
            yaw = np.arctan2(dy, dx)
            c, s = np.cos(yaw), np.sin(yaw)
            out[b, i] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return out


class DiscreteTrajectoryTokenizerMLX:
    """MLX port of DiscreteTrajectoryTokenizer.

    This tokenizer uses an ActionSpace (UnicycleAccelCurvature) to convert
    trajectories to actions, then discretizes the actions into tokens.
    """

    def __init__(
        self,
        action_space: ActionSpace | None = None,
        dims_min: list[float] | None = None,
        dims_max: list[float] | None = None,
        num_bins: int = 1000,
    ):
        self.action_space = action_space or ActionSpace()
        # Default normalization bounds for accel/curvature (typical values)
        self.dims_min = mx.array(dims_min or [-5.0, -1.0])
        self.dims_max = mx.array(dims_max or [5.0, 1.0])
        self.num_bins = num_bins

    def encode(
        self,
        hist_xyz: mx.array,
        hist_rot: mx.array,
        fut_xyz: mx.array,
        fut_rot: mx.array,
        hist_tstamp: mx.array | None = None,
        fut_tstamp: mx.array | None = None,
    ) -> mx.array:
        """Encode trajectories into discrete action tokens.

        For history tokenization we treat the history as "future" for the
        tokenizer (as done in NVIDIA's tokenize_history_trajectory).
        """
        # Flatten batch
        B = hist_xyz.shape[0]
        hist_xyz_flat = hist_xyz.reshape(B, -1, 3)
        hist_rot_flat = hist_rot.reshape(B, -1, 3, 3)
        fut_xyz_flat = fut_xyz.reshape(B, -1, 3)
        fut_rot_flat = fut_rot.reshape(B, -1, 3, 3)

        # Real implementation: convert history trajectory to actions via traj_to_action
        # Always provide t0_states with zero v0 for history tokenization (no ego velocity in prompt)
        B = hist_xyz_flat.shape[0]
        t0_states = {"v": mx.zeros((B,), dtype=mx.float32)}
        action = self.action_space.traj_to_action(
            hist_xyz_flat, hist_rot_flat, fut_xyz_flat, fut_rot_flat, t0_states=t0_states
        )
        # Discretize (simple linear quantization into num_bins)
        action = mx.clip(action, self.dims_min, self.dims_max)
        scale = self.dims_max - self.dims_min
        tokens = ((action - self.dims_min) / scale * (self.num_bins - 1)).astype(mx.int32)
        # Flatten last two dims (accel + kappa per waypoint)
        tokens = tokens.reshape(B, -1)
        return tokens

    def decode(
        self,
        hist_xyz: mx.array,
        hist_rot: mx.array,
        tokens: mx.array,
        hist_tstamp: mx.array | None = None,
    ) -> Tuple[mx.array, mx.array, Any]:
        """Decode tokens back to future trajectories."""
        # This would use action_space.action_to_traj after denormalization.
        # For now we delegate to the ActionSpace we already implemented.
        action = tokens.reshape(-1, *self.action_space.get_action_space_dims())
        fut_xyz, fut_rot = self.action_space.action_to_traj(
            action, hist_xyz, hist_rot
        )
        return fut_xyz, fut_rot, None