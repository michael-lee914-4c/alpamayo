"""RL reshape is camera-major flatten + ``repeat_interleave`` + unsqueeze 1.

Weight-free. Complementary to open #16 (preprocess twice around reshape)
and #15 (``squeeze(0)`` keeps n_traj). ``reshape_tensors_for_rl`` turns
``(N_cam, N_frame, C, H, W)`` into ``(N_cam * N_frame, 1, C, H, W)``
and repeats each camera id ``N_frame`` times. ``tile`` (frame-major) or
skipping the singleton dim would scramble Qwen's camera-major images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_PAI = _REPO / "src" / "alpamayo_r1" / "data" / "pai.py"


def test_source_repeat_interleave_and_unsqueeze():
    src = _PAI.read_text()
    body = src.split("if self.reshape_tensors_for_rl:")[1]
    assert "image_frames.reshape(n_cam * n_frame, *image_frames.shape[2:]).unsqueeze(" in body
    assert "camera_indices.repeat_interleave(n_frame)" in body
    assert "camera_indices.repeat(" not in body
    assert "camera_indices.tile(" not in body


def test_repeat_interleave_is_camera_major():
    cams = np.array([0, 6])
    n_frame = 2
    ids = np.repeat(cams, n_frame)
    assert ids.tolist() == [0, 0, 6, 6]
    tiled = np.tile(cams, n_frame)
    assert tiled.tolist() == [0, 6, 0, 6]
