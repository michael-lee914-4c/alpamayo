"""RL aggregated-reward ADE gate and comfort exclusive bounds.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #22/#24,
which locked yaw wrap, required reward keys, and empty-span decode — not
the 3.0m ADE hard gate or a violent-traj comfort fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def _stub_cosmos_logger() -> None:
    if "cosmos_rl.utils.logging" in sys.modules:
        return
    cosmos = ModuleType("cosmos_rl")
    utils = ModuleType("cosmos_rl.utils")
    logging = ModuleType("cosmos_rl.utils.logging")
    logging.logger = SimpleNamespace(debug=lambda *a, **k: None)
    sys.modules.setdefault("cosmos_rl", cosmos)
    sys.modules.setdefault("cosmos_rl.utils", utils)
    sys.modules["cosmos_rl.utils.logging"] = logging


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_within_bound_is_exclusive_and_comfort_fails_violent_traj(torch):
    comfort = _load_rl_module("rl_comfort_reward_gate", "rl/rewards/comfort_reward.py")
    interior = comfort._within_bound(
        torch.tensor([[0.0]], dtype=torch.float32),
        comfort.MIN_LON_ACCEL,
        comfort.MAX_LON_ACCEL,
    )
    assert float(interior.item()) == 1.0
    on_min = comfort._within_bound(
        torch.tensor([[comfort.MIN_LON_ACCEL]], dtype=torch.float32),
        comfort.MIN_LON_ACCEL,
        comfort.MAX_LON_ACCEL,
    )
    assert float(on_min.item()) == 0.0

    t = 8
    xyz = torch.zeros((1, 1, 1, t, 3), dtype=torch.float32)
    # Quadratic x(t) → large lon accel; speed increases so mag-jerk is also out of bound.
    xyz[0, 0, 0, :, 0] = torch.arange(t, dtype=torch.float32) ** 2
    rot = torch.eye(3, dtype=torch.float32).expand(1, 1, 1, t, 3, 3).clone()
    scores = comfort.compute_comfort(xyz, rot)
    assert float(scores["comfort/lon_accel"].item()) == 0.0
    assert float(scores["comfort/jerk"].item()) == 0.0
    assert float(scores["comfort/lon_jerk"].item()) == 0.0


def test_aggregated_reward_hard_gates_ade_at_3m(torch):
    _stub_cosmos_logger()
    agg = _load_rl_module("rl_aggregated_reward_gate", "rl/rewards/aggregated_reward.py")
    comfort = _load_rl_module("rl_comfort_reward_gate", "rl/rewards/comfort_reward.py")

    t = 4
    gt = torch.zeros((1, t, 3), dtype=torch.float32)
    hist = torch.zeros((1, 2, 3), dtype=torch.float32)
    hist_rot = torch.eye(3).expand(1, 2, 3, 3).clone()
    pred_rot = torch.eye(3).expand(1, t, 3, 3).clone()

    cfg = SimpleNamespace(
        custom={"alpamayo": {"reward": {"traj_l2_weight": 1.0, "comfort_weight": 0.25}}}
    )

    def _decode(to_be_evaluated, *_a, **_k):
        offset = float(to_be_evaluated)
        pred = gt.clone()
        pred[..., 0] += offset
        return pred, pred_rot

    # compute_reward imports these inside the function.
    sys.modules.setdefault("rl", ModuleType("rl"))
    sys.modules.setdefault("rl.rewards", ModuleType("rl.rewards"))
    sys.modules.setdefault("rl.utils", ModuleType("rl.utils"))
    sys.modules["rl.rewards.comfort_reward"] = comfort
    sys.modules["rl.rewards.traj_reward"] = _load_rl_module(
        "rl_traj_reward_gate", "rl/rewards/traj_reward.py"
    )
    decode_mod = ModuleType("rl.utils.trajectory_decode")
    decode_mod.decode_rollout_trajectory = _decode
    sys.modules["rl.utils.trajectory_decode"] = decode_mod

    ref = {
        "ego_future_xyz": gt,
        "ego_history_xyz": hist,
        "ego_history_rot": hist_rot,
    }
    # ADE == 3.0 is not < 3.0 → hard penalty -1, comfort ignored.
    gated, gated_dict = agg.compute_reward(
        "3.0",
        ref,
        tokenizer=None,
        traj_tokenizer=None,
        config=cfg,
        model_config=None,
    )
    assert gated == -1.0
    assert abs(gated_dict["traj_L2"] - 3.0) < 1e-5

    # ADE == 1.5 with a still comfort-passing straight translate.
    # Straight +x shift is constant velocity 0 relative to itself? Positions are
    # constant-in-time offset, so velocity is 0 → comfort metrics are 1.0,
    # comfort_score = 1 - 1 = 0.
    below, below_dict = agg.compute_reward(
        "1.5",
        ref,
        tokenizer=None,
        traj_tokenizer=None,
        config=cfg,
        model_config=None,
    )
    assert abs(below_dict["traj_L2"] - 1.5) < 1e-5
    expected = -1.0 * (1.5 / 3.0) + 0.25 * below_dict["comfort_reward"]
    assert abs(below - expected) < 1e-5
    assert below > -1.0
