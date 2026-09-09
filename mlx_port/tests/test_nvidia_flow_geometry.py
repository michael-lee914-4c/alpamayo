"""NVIDIA CFM training algebra and geometry helpers.

Weight-free. Needs torch (macos-26 mlx-ci). Complementary to open #17–#24
official ADE / MLX CFM graph tests — this locks the CUDA interpolation,
MSE target, Euler step, and yaw wrap used by action/reward.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_fns(rel: str, names: tuple[str, ...], extra: dict | None = None):
    path = _SRC / rel
    tree = ast.parse(path.read_text())
    keep = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and getattr(node, "name", None) in names
    ]
    missing = set(names) - {node.name for node in keep}
    if missing:
        raise AssertionError(f"{path} is missing {sorted(missing)}")
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"__name__": "extracted", **(extra or {})}
    exec(compile(mod, str(path), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_flow(torch):
    _ensure_src_on_path()
    try:
        from alpamayo_r1.diffusion.flow_matching import FlowMatching
    except Exception as exc:
        pytest.skip(f"NVIDIA FlowMatching unavailable: {exc}")
    return FlowMatching


def test_flow_matching_training_data_is_linear_interpolation(torch):
    FlowMatching = _load_flow(torch)
    fm = FlowMatching(x_dims=(2,), train_timestep_sampler="uniform")
    x = torch.tensor([[2.0, -4.0], [0.0, 1.0]])
    data = fm.construct_training_data(x)
    t = data["timesteps"]
    while t.ndim < x.ndim:
        t = t.unsqueeze(-1)
    expected = t * data["x"] + (1.0 - t) * data["noise"]
    assert torch.allclose(data["noisy_x"], expected, atol=1e-6)
    assert data["x"].shape == x.shape
    assert data["noise"].shape == x.shape
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1.0


def test_flow_matching_loss_is_mse_of_x_minus_noise(torch):
    FlowMatching = _load_flow(torch)
    fm = FlowMatching(x_dims=(2,), train_timestep_sampler="uniform")
    x = torch.tensor([[1.0, 2.0]])
    noise = torch.tensor([[0.0, 1.0]])
    pred = torch.tensor([[1.0, 1.0]])  # equals x - noise
    loss = fm.compute_loss_from_pred({"x": x, "noise": noise}, pred)
    assert abs(float(loss.item()) - 0.0) < 1e-6
    wrong = fm.compute_loss_from_pred({"x": x, "noise": noise}, torch.zeros_like(pred))
    # target is [1, 1], pred is [0, 0] → MSE 1.
    assert abs(float(wrong.item()) - 1.0) < 1e-6


def test_flow_matching_euler_integrates_constant_velocity(torch):
    FlowMatching = _load_flow(torch)
    fm = FlowMatching(x_dims=(2,), int_method="euler", num_inference_steps=4)
    torch.manual_seed(0)

    def step_fn(*, x, t):
        del x, t
        return torch.ones((1, 2))

    out = fm.sample(batch_size=1, step_fn=step_fn, device=torch.device("cpu"))
    # x0 + 1.0 * v, v=1 → each dim increases by 1 regardless of x0.
    # We cannot know x0, so compare two-step vs identity field via return_all_steps.
    all_steps, times = fm.sample(
        batch_size=1, step_fn=step_fn, device=torch.device("cpu"), return_all_steps=True
    )
    assert times.numel() == 5
    assert all_steps.shape == (1, 5, 2)
    # Constant v=1 over t in [0,1] adds exactly 1.0.
    delta = all_steps[:, -1] - all_steps[:, 0]
    assert torch.allclose(delta, torch.ones_like(delta), atol=1e-5)
    assert out.shape == (1, 2)

    try:
        FlowMatching(x_dims=(2,), int_method="rk4").sample(1, step_fn=step_fn)
    except ValueError as exc:
        assert "rk4" in str(exc)
    else:
        raise AssertionError("expected ValueError for a non-euler integrator")

    try:
        FlowMatching(x_dims=(2,), train_timestep_sampler="cosine").construct_training_data(
            torch.zeros((1, 2))
        )
    except ValueError as exc:
        assert "cosine" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown time sampler")


def test_angle_wrap_and_so3_yaw_are_left_closed(torch):
    _ensure_src_on_path()
    fns = _load_fns(
        "alpamayo_r1/geometry/rotation.py",
        ("angle_wrap", "so3_to_yaw_np", "so3_to_yaw_torch", "rotation_matrix", "transform_coords_2d_np"),
        extra={
            "np": np,
            "torch": torch,
            "Optional": object,
            "TypeVar": lambda *a, **k: object,
            "Union": object,
            "TensorOrNDArray": object,
            "R": None,
        },
    )
    wrapped = fns["angle_wrap"](np.array([math.pi, -math.pi, 3 * math.pi, -3 * math.pi]))
    # Contract is [-pi, pi): +pi wraps to -pi; -pi stays -pi.
    assert abs(float(wrapped[0]) + math.pi) < 1e-6
    assert abs(float(wrapped[1]) + math.pi) < 1e-6
    assert abs(float(wrapped[2]) + math.pi) < 1e-6
    assert abs(float(wrapped[3]) + math.pi) < 1e-6

    eye = np.eye(3, dtype=np.float32)
    assert abs(float(fns["so3_to_yaw_np"](eye))) < 1e-6
    yaw90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    assert abs(float(fns["so3_to_yaw_np"](yaw90)) - math.pi / 2) < 1e-5
    assert abs(float(fns["so3_to_yaw_torch"](torch.from_numpy(yaw90)).item()) - math.pi / 2) < 1e-5

    rot = fns["rotation_matrix"](math.pi / 2)
    got = fns["transform_coords_2d_np"](np.array([1.0, 0.0]), angle=math.pi / 2)
    assert np.allclose(got, np.array([0.0, 1.0]), atol=1e-6)
    assert np.allclose(rot @ np.array([1.0, 0.0]), np.array([0.0, 1.0]), atol=1e-6)


def test_grouped_corner_distance_picks_best_of_k(torch):
    _ensure_src_on_path()
    try:
        from alpamayo_r1.metrics.distance_metrics import compute_grouped_corner_distance
    except Exception as exc:
        pytest.skip(f"NVIDIA corner distance unavailable: {exc}")

    # Identity boxes: one matching sample and one translated by 2m in X.
    gt_xyz = torch.zeros(1, 2, 3)
    gt_rot = torch.eye(3).expand(1, 2, 3, 3).clone()
    pred_xyz = torch.zeros(1, 1, 2, 2, 3)
    pred_xyz[0, 0, 1, :, 0] = 2.0
    pred_rot = torch.eye(3).expand(1, 1, 2, 2, 3, 3).clone()
    dims = torch.tensor([2.0, 2.0, 2.0])
    out = compute_grouped_corner_distance(
        pred_xyz, pred_rot, gt_xyz, gt_rot, dims, disable_summary=True
    )
    assert set(out) == {"corner_distance"}
    # Best-of-K is the identity sample → 0.
    assert abs(float(out["corner_distance"].item()) - 0.0) < 1e-5
