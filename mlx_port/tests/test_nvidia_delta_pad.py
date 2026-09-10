"""NVIDIA delta tokenizer: origin-pad first step, clamp, and yaw sign.

Weight-free. Complementary to the existing MLX/NVIDIA encode-id parity
suite — this locks the pad-to-origin first waypoint, out-of-range clamp,
and ``atan2(R[0,1], R[0,0]) = −so3_yaw`` convention.
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


def _load_delta():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer
    except Exception as exc:
        pytest.skip(f"NVIDIA delta tokenizer unavailable: {exc}")
    return DeltaTrajectoryTokenizer


def _bin(value: float, lo: float = -4.0, hi: float = 4.0, n: int = 1000) -> int:
    return int(round((value - lo) / (hi - lo) * (n - 1)))


def test_delta_encode_first_step_is_absolute_from_origin(torch):
    DeltaTrajectoryTokenizer = _load_delta()
    tok = DeltaTrajectoryTokenizer(num_bins=1000)
    # Pad prepends a zero waypoint, so tokens are (p0-0, p1-p0, p2-p1).
    fut = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    hist = torch.zeros(1, 1, 3)
    hrot = torch.eye(3).expand(1, 1, 3, 3).clone()
    frot = torch.eye(3).expand(1, 3, 3, 3).clone()
    ids = tok.encode(hist, hrot, fut, frot)
    assert ids.shape == (1, 9)
    # Δ = (0,0,0), (1,0,0), (2,0,0) → bins 500 / 624 / 749 on X, 500 on Y/Z.
    assert ids[0].tolist() == [
        _bin(0),
        _bin(0),
        500,
        _bin(1),
        _bin(0),
        500,
        _bin(2),
        _bin(0),
        500,
    ]


def test_delta_encode_clamps_out_of_range_xyz(torch):
    DeltaTrajectoryTokenizer = _load_delta()
    tok = DeltaTrajectoryTokenizer(num_bins=1000)
    hist = torch.zeros(1, 1, 3)
    hrot = torch.eye(3).expand(1, 1, 3, 3).clone()
    fut = torch.tensor([[[100.0, -100.0, 0.0]]])
    frot = torch.eye(3).expand(1, 1, 3, 3).clone()
    ids = tok.encode(hist, hrot, fut, frot)
    assert ids[0].tolist() == [999, 0, 500]


def test_delta_yaw_is_negative_of_so3_yaw_and_wraps(torch):
    DeltaTrajectoryTokenizer = _load_delta()
    tok = DeltaTrajectoryTokenizer(num_bins=1000, predict_yaw=True)
    yaw = np.pi / 2
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    # Standard Rz(yaw): so3_to_yaw = +π/2; delta tokenizer reads R[0,1],R[0,0] → −π/2.
    rot = torch.tensor([[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)
    extracted = float(torch.atan2(rot[..., 0, 1], rot[..., 0, 0]).item())
    so3 = float(torch.atan2(rot[..., 1, 0], rot[..., 0, 0]).item())
    assert extracted == pytest.approx(-np.pi / 2, abs=1e-5)
    assert so3 == pytest.approx(np.pi / 2, abs=1e-5)
    assert extracted == pytest.approx(-so3, abs=1e-5)

    hist = torch.zeros(1, 1, 3)
    hrot = torch.eye(3).expand(1, 1, 3, 3).clone()
    fut = torch.zeros(1, 1, 3)
    ids = tok.encode(hist, hrot, fut, rot.unsqueeze(0) if rot.ndim == 2 else rot.view(1, 1, 3, 3))
    # tokens = (xyz Δ, Δyaw); first Δyaw is extracted_yaw - 0, wrapped.
    # (−π/2 − (−π)) / (2π) * 999 → 0.25 * 999 = 249.75 → 250
    yaw_bin = int(ids[0, 3].item())
    expected = int(round(((-np.pi / 2) - (-np.pi)) / (2 * np.pi) * 999))
    assert yaw_bin == expected


def test_delta_decode_is_cumsum_of_dequantized_deltas(torch):
    DeltaTrajectoryTokenizer = _load_delta()
    tok = DeltaTrajectoryTokenizer(num_bins=1000)
    # Hand-picked bins: mid / +1m-ish / +2m-ish on X.
    tokens = torch.tensor([[500, 500, 500, 624, 500, 500, 749, 500, 500]], dtype=torch.long)
    hist = torch.zeros(1, 1, 3)
    hrot = torch.eye(3).expand(1, 1, 3, 3).clone()
    xyz, rot, ts = tok.decode(hist, hrot, tokens)
    assert ts is None
    assert xyz.shape == (1, 3, 3)
    # Dequant of bin b is lo + (b/999)*(hi-lo).
    d0 = -4.0 + 500 / 999 * 8.0
    d1 = -4.0 + 624 / 999 * 8.0
    d2 = -4.0 + 749 / 999 * 8.0
    expect = torch.tensor([d0, d0 + d1, d0 + d1 + d2], dtype=xyz.dtype)
    assert torch.allclose(xyz[0, :, 0], expect, atol=1e-5)
    assert rot.shape == (1, 3, 3, 3)
