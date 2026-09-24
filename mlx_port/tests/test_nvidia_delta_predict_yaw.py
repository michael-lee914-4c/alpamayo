"""``DeltaTrajectoryTokenizer(predict_yaw=True)`` uses ``R[0,1]``, not so3 yaw.

Weight-free. Complementary to open #10 (default-branch first-step pad /
``-so3`` yaw) and the MLX encode-parity suite. The yaw branch reads
``atan2(R[..., 0, 1], R[..., 0, 0])`` (``-sin`` for a standard yaw
matrix) then decodes with ``cumsum`` + ``[[cos,-sin],[sin,cos]]``.
Switching encode to ``so3_to_yaw`` (``R[1,0]``) without flipping decode
mirrors every heading.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DELTA = _REPO / "src" / "alpamayo_r1" / "models" / "delta_tokenizer.py"


def test_source_predict_yaw_uses_r01_and_cumsum():
    src = _DELTA.read_text()
    assert "yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])" in src
    assert "yaw = torch.cumsum(yaw, dim=1)" in src
    assert "torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1)" in src
    # Official heading helper is the opposite pair; this branch must not use it.
    assert "fut_rot[..., 1, 0]" not in src
    assert "so3_to_yaw" not in src


def test_encode_then_decode_recovers_negated_standard_yaw():
    torch = pytest.importorskip("torch")
    from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer

    yaw = torch.tensor([[0.4, 0.8]], dtype=torch.float32)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    zeros = torch.zeros_like(cos_y)
    ones = torch.ones_like(cos_y)
    # Standard +Z yaw: [[cos,-sin,0],[sin,cos,0],[0,0,1]].
    fut_rot = torch.stack(
        [
            torch.stack([cos_y, -sin_y, zeros], dim=-1),
            torch.stack([sin_y, cos_y, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    fut_xyz = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32)
    hist_xyz = torch.zeros(1, 1, 3)
    hist_rot = torch.eye(3).view(1, 1, 3, 3)

    tok = DeltaTrajectoryTokenizer(predict_yaw=True, num_bins=1000)
    tokens = tok.encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
    assert tokens.shape == (1, 8)

    # Encode stores atan2(-sin, cos) = -yaw, then delta from a padded 0.
    encoded_yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])
    torch.testing.assert_close(encoded_yaw, -yaw, atol=1e-6, rtol=0)

    dec_xyz, dec_rot, dec_t = tok.decode(hist_xyz, hist_rot, tokens)
    assert dec_t is None
    # XY is quantized over [-4, 4]; Z over [-10, 10] so a true-zero
    # delta can accumulate ~0.02 m over two steps.
    torch.testing.assert_close(dec_xyz[..., :2], fut_xyz[..., :2], atol=2e-2, rtol=0)
    # Decode rebuilds from the negated heading, so R becomes R^T.
    torch.testing.assert_close(dec_rot, fut_rot.transpose(-1, -2), atol=5e-2, rtol=0)
