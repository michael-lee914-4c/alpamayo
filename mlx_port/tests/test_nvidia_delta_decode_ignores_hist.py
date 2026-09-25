"""Delta ``predict_yaw=False`` decode ignores history and polyfits heading.

Weight-free. Complementary to open #44 (``predict_yaw=True`` uses ``R[0,1]``)
and #26 (encode pads the first delta from the origin). Decode ``cumsum``s
deltas from zero — it does not add the last history waypoint — then builds
``fut_rot`` from ``get_yaw_rotation_matrices(fut_xyz)``. A heading taken from
``hist_rot`` or a decode that starts at ``hist_xyz[:, -1]`` would rotate every
discrete Stage-1 rollout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DELTA = _REPO / "src" / "alpamayo_r1" / "models" / "delta_tokenizer.py"


def test_source_decode_cumsum_then_polyfit_when_yaw_off():
    src = _DELTA.read_text()
    assert "fut_xyz = torch.cumsum(xyz, dim=1)" in src
    assert "if not self._predict_yaw:" in src
    assert "fut_rot = get_yaw_rotation_matrices(xyz_cpu)" in src
    assert "return fut_xyz, fut_rot, None" in src
    # hist tensors are accepted but unused for the no-yaw path
    assert "del hist_tstamp" in src


def test_decode_starts_at_origin_not_last_history():
    torch = pytest.importorskip("torch")
    from alpamayo_r1.models.delta_tokenizer import DeltaTrajectoryTokenizer

    tok = DeltaTrajectoryTokenizer(num_bins=11, predict_yaw=False)
    # Bin 10 → x = 4 m/step; bin 5 → y/z = 0. Eight steps keep the polyfit stable.
    step = torch.tensor([10, 5, 5], dtype=torch.long)
    tokens = step.repeat(8).unsqueeze(0)
    hist_xyz = torch.tensor([[[10.0, 20.0, 30.0]]])
    hist_rot = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    fut_xyz, fut_rot, fut_t = tok.decode(hist_xyz, hist_rot, tokens)
    assert fut_t is None
    # First waypoint is one +x delta from the origin, not from (10, 20, 30).
    torch.testing.assert_close(fut_xyz[0, 0], torch.tensor([4.0, 0.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(fut_xyz[0, -1], torch.tensor([32.0, 0.0, 0.0]), atol=1e-5, rtol=0)
    assert float(fut_xyz[0, 0, 0]) != 14.0
    # Heading comes from the +x polyfit, not the 90° history yaw.
    torch.testing.assert_close(fut_rot[0, 0], torch.eye(3), atol=1e-4, rtol=0)
