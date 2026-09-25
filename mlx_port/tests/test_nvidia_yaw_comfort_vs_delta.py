"""Comfort yaw is ``R[1,0]``; delta ``predict_yaw`` encode is ``R[0,1]``.

Weight-free. Complementary to open #41 (comfort uses the first SO(3) column)
and #44 (delta encode ``atan2(R[0,1], R[0,0])``, decode is Rᵀ of a standard
yaw). On a standard yaw matrix ``[[c,-s],[s,c]]`` those two extracts are
negatives of each other. Swapping them without flipping the paired path
mirrors every heading used by GRPO comfort or Stage-1 discrete tokens.
"""

from __future__ import annotations

from pathlib import Path

import math

import pytest

_REPO = Path(__file__).resolve().parents[2]
_COMFORT = _REPO / "finetune" / "rl" / "rewards" / "comfort_reward.py"
_DELTA = _REPO / "src" / "alpamayo_r1" / "models" / "delta_tokenizer.py"
_ROT = _REPO / "src" / "alpamayo_r1" / "geometry" / "rotation.py"


def test_source_comfort_uses_first_column_so3():
    src = _COMFORT.read_text()
    assert "ego_h = torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])" in src


def test_source_delta_encode_uses_r01():
    src = _DELTA.read_text()
    assert "yaw = torch.atan2(fut_rot[..., 0, 1], fut_rot[..., 0, 0])" in src


def test_source_so3_to_yaw_matches_comfort():
    src = _ROT.read_text()
    assert "cos_th_sin_phi = rot_mat[..., 1, 0]" in src
    assert "cos_th_cos_phi = rot_mat[..., 0, 0]" in src
    assert "return torch.atan2(cos_th_sin_phi, cos_th_cos_phi)" in src


def test_standard_yaw_matrix_extracts_are_negatives():
    torch = pytest.importorskip("torch")
    yaw = math.pi / 6
    c, s = math.cos(yaw), math.sin(yaw)
    rot = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    comfort = torch.atan2(rot[..., 1, 0], rot[..., 0, 0])
    delta = torch.atan2(rot[..., 0, 1], rot[..., 0, 0])
    torch.testing.assert_close(comfort, torch.tensor(yaw), atol=1e-6, rtol=0)
    torch.testing.assert_close(delta, torch.tensor(-yaw), atol=1e-6, rtol=0)
    torch.testing.assert_close(comfort, -delta, atol=1e-6, rtol=0)
