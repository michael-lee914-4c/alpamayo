"""NVIDIA delta decode: polyfit yaw when predict_yaw is off.

Weight-free. Complementary to open #26 (origin-pad / clamp / −so3 encode
sign) — that suite never asserted the heading matrices produced by
``get_yaw_rotation_matrices``, which is the only rotation source on the
no-yaw decode path used by RL comfort/ADE.
"""

from __future__ import annotations

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


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_yaw():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.models.delta_tokenizer import get_yaw_rotation_matrices
    except Exception as exc:
        pytest.skip(f"NVIDIA delta yaw helper unavailable: {exc}")
    return get_yaw_rotation_matrices


def _yaw_from_R(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def test_polyfit_yaw_is_zero_on_plus_x_line():
    get_yaw_rotation_matrices = _load_yaw()
    # Straight +X: dx>0, dy=0 → yaw ≈ 0 and R maps +X onto +X.
    traj = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    R = get_yaw_rotation_matrices(traj)
    assert R.shape == (1, 4, 3, 3)
    assert np.isfinite(R).all()
    yaws = _yaw_from_R(R[0])
    assert np.max(np.abs(yaws)) < 1e-6
    x_hat = R[0] @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(x_hat, np.tile([1.0, 0.0, 0.0], (4, 1)), atol=1e-6)


def test_polyfit_yaw_turns_toward_plus_y():
    get_yaw_rotation_matrices = _load_yaw()
    # 12 steps +X then 12 steps +Y. Default window=10 so the last heading
    # is fit on a +Y-only neighborhood (a short L overshoots ~1.83 rad).
    xs = np.concatenate([np.linspace(0.0, 11.0, 12), np.full(12, 11.0)])
    ys = np.concatenate([np.zeros(12), np.linspace(1.0, 12.0, 12)])
    traj = np.stack([xs, ys, np.zeros(24)], axis=-1)[None]
    R = get_yaw_rotation_matrices(traj)
    yaws = _yaw_from_R(R[0])
    assert yaws[2] == pytest.approx(0.0, abs=0.15)
    assert yaws[-1] == pytest.approx(np.pi / 2, abs=0.15)
    y_hat = R[0, -1] @ np.array([1.0, 0.0, 0.0])
    assert y_hat[1] > y_hat[0]


def test_polyfit_yaw_edge_window_is_finite_and_batched():
    get_yaw_rotation_matrices = _load_yaw()
    plus_x = np.stack(
        [
            np.linspace(0.0, 3.0, 6),
            np.zeros(6),
            np.zeros(6),
        ],
        axis=-1,
    )
    plus_y = np.stack(
        [
            np.zeros(6),
            np.linspace(0.0, 3.0, 6),
            np.zeros(6),
        ],
        axis=-1,
    )
    R = get_yaw_rotation_matrices(np.stack([plus_x, plus_y], axis=0))
    assert R.shape == (2, 6, 3, 3)
    assert np.isfinite(R).all()
    yaws = _yaw_from_R(R)
    assert np.max(np.abs(yaws[0])) < 1e-6
    assert np.max(np.abs(yaws[1] - np.pi / 2)) < 1e-6
