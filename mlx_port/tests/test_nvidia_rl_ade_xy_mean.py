"""RL ``calculate_ade`` is XY-only and means over every remaining dim.

Weight-free. Complementary to open #24/#34 (official ``compute_ade``
horizon raise / ``only_xy``) and #25 (ADE 3.0 hard gate). The reward
helper asserts exact shape equality, drops Z, then ``distances.mean()``
across batch *and* time. Averaging only over T, or including Z, changes
the GRPO scalar that gates comfort.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_REWARD = _REPO / "finetune" / "rl" / "rewards" / "traj_reward.py"


def _calculate_ade(pred_trajectory, gt_trajectory):
    torch = pytest.importorskip("torch")
    assert pred_trajectory.shape == gt_trajectory.shape, (
        f"Shape mismatch: pred {pred_trajectory.shape} vs gt {gt_trajectory.shape}"
    )
    pred_xy = pred_trajectory[..., :2]
    gt_xy = gt_trajectory[..., :2]
    distances = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return float(distances.mean())


def test_source_asserts_shape_and_means_xy():
    src = _REWARD.read_text()
    assert "assert pred_trajectory.shape == gt_trajectory.shape" in src
    assert "pred_xy = pred_trajectory[..., :2]" in src
    assert "return float(distances.mean())" in src
    assert "horizon" not in src


def test_z_is_ignored_and_mean_is_global():
    torch = pytest.importorskip("torch")
    pred = torch.tensor(
        [
            [[0.0, 0.0, 9.0], [2.0, 0.0, 9.0]],
            [[0.0, 0.0, -9.0], [0.0, 4.0, -9.0]],
        ],
        dtype=torch.float32,
    )
    gt = torch.zeros_like(pred)
    # Per-step XY distances: 0, 2, 0, 4 → mean 1.5. Z must not enter.
    assert _calculate_ade(pred, gt) == pytest.approx(1.5)


def test_shape_mismatch_raises():
    torch = pytest.importorskip("torch")
    pred = torch.zeros(2, 4, 3)
    gt = torch.zeros(4, 3)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        _calculate_ade(pred, gt)
