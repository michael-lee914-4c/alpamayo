"""RL decode + reward helpers. Weight-free; needs torch (macos-26 mlx-ci)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_FINETUNE = _REPO / "finetune"


def _load_rl_module(mod_name: str, rel: str):
    path = _FINETUNE / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_extract_traj_tokens_last_start_clamp_and_empty(torch):
    try:
        from alpamayo_r1.models.token_utils import extract_traj_tokens
    except Exception as exc:
        pytest.skip(f"NVIDIA token_utils unavailable: {exc}")

    specials = {"traj_future_start": 7, "traj_future_end": 8}
    offset = 100
    vocab = 10

    # Last start wins; extras after expected_len are dropped.
    two_starts = torch.tensor(
        [[0, 7, 101, 102, 7, 103, 104, 105, 106, 8]],
        dtype=torch.long,
    )
    got = extract_traj_tokens(two_starts, specials, 3, offset, vocab)
    assert got.shape == (1, 3)
    assert got.tolist() == [[3, 4, 5]]

    # Missing markers → zeros of the expected future length (RL empty-span fallback).
    empty = extract_traj_tokens(torch.zeros((2, 4), dtype=torch.long), specials, 5, offset, vocab)
    assert empty.shape == (2, 5)
    assert int(empty.sum().item()) == 0

    # Out-of-vocab bins clamp; a short span pads the tail with zeros.
    short_invalid = torch.tensor([[7, 109, 199, 8, 0]], dtype=torch.long)
    clamped = extract_traj_tokens(short_invalid, specials, 4, offset, vocab)
    assert clamped.tolist() == [[9, 9, 0, 0]]


def test_calculate_ade_is_xy_only_and_rejects_shape_mismatch(torch):
    traj = _load_rl_module("rl_traj_reward", "rl/rewards/traj_reward.py")
    pred = torch.tensor([[[0.0, 0.0, 9.0], [1.0, 0.0, -4.0]]], dtype=torch.float32)
    gt = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 3.0]]], dtype=torch.float32)
    assert traj.calculate_ade(pred, gt) == 0.0
    shifted = gt.clone()
    shifted[..., 0] += 2.0
    assert abs(traj.calculate_ade(pred, shifted) - 2.0) < 1e-6
    try:
        traj.calculate_ade(pred, gt[:, :1])
    except AssertionError as exc:
        assert "Shape mismatch" in str(exc)
    else:
        raise AssertionError("expected AssertionError when pred/gt shapes differ")


def test_comfort_yaw_wrap_and_straight_line_bounds(torch):
    comfort = _load_rl_module("rl_comfort_reward", "rl/rewards/comfort_reward.py")
    # +pi → -pi is a small wrap, not a 2π jump.
    yaw = torch.tensor([3.0, -3.0], dtype=torch.float32)
    rate = comfort._diff_yaw(yaw)
    wrap = ((-3.0 - 3.0) + 2.0 * float(np.pi)) * comfort.PLANNING_FREQ
    assert abs(float(rate[0].item()) - wrap) < 1e-5
    assert abs(float(rate[1].item()) - wrap) < 1e-5

    # Exclusive bounds: sitting on MAX_LON_ACCEL is not comfortable.
    on_bound = comfort._within_bound(
        torch.tensor([[comfort.MAX_LON_ACCEL]], dtype=torch.float32),
        comfort.MIN_LON_ACCEL,
        comfort.MAX_LON_ACCEL,
    )
    assert float(on_bound.item()) == 0.0

    t = 8
    xyz = torch.zeros((1, 1, 1, t, 3), dtype=torch.float32)
    xyz[0, 0, 0, :, 0] = torch.arange(t, dtype=torch.float32) * 0.5
    rot = torch.eye(3, dtype=torch.float32).expand(1, 1, 1, t, 3, 3).clone()
    scores = comfort.compute_comfort(xyz, rot)
    assert set(scores) == set(comfort.COMFORT_METRIC_CONFIG_DICT)
    for name, val in scores.items():
        assert val.shape == (1,)
        assert float(val.item()) == 1.0


def test_reward_cfg_requires_both_weights():
    agg = _load_rl_module("rl_aggregated_reward", "rl/rewards/aggregated_reward.py")
    try:
        agg._get_reward_cfg(None)
    except ValueError as exc:
        assert "custom.alpamayo.reward" in str(exc)
    else:
        raise AssertionError("expected ValueError when reward config is missing")
    try:
        agg._get_reward_cfg(SimpleNamespace(custom={"alpamayo": {"reward": {"traj_l2_weight": 1.0}}}))
    except ValueError as exc:
        assert "comfort_weight" in str(exc)
    else:
        raise AssertionError("expected ValueError when comfort_weight is missing")
    got = agg._get_reward_cfg(
        SimpleNamespace(
            custom={"alpamayo": {"reward": {"traj_l2_weight": 2, "comfort_weight": 0.5}}}
        )
    )
    assert got == {"traj_l2_weight": 2.0, "comfort_weight": 0.5}
