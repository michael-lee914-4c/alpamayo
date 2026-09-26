"""Stage-2 infer repeats the last traj *group*, not the last timestep.

Weight-free. Complementary to open #30 (last-hist group layout) and #45
(``traj_to_action`` cats last-hist). ``ego_history_xyz[:, -1]`` on a
``[B, n_traj, T, 3]`` tensor is ``[B, T, 3]``. Indexing ``[:, :, -1]``
would feed ``action_to_traj`` a single waypoint per group and break
origin / Z-copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_MODEL = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"


def test_source_indexes_last_group_not_last_step():
    src = _MODEL.read_text()
    snippet = src.split("hist_xyz_rep = einops.repeat(")[1].split("pred_xyz, pred_rot =")[0]
    assert "ego_history_xyz[:, -1]" in snippet
    assert "ego_history_rot[:, -1]" in snippet
    assert '"b ... -> (b n) ..."' in snippet
    assert "ego_history_xyz[:, :, -1]" not in snippet
    assert "ego_history_xyz[..., -1, :]" not in snippet


def test_last_group_keeps_full_horizon():
    # [B=1, n_traj=2, T=4, 3]: group 0 is zeros, group 1 is arange.
    hist = np.zeros((1, 2, 4, 3), dtype=np.float32)
    hist[0, 1] = np.arange(12, dtype=np.float32).reshape(4, 3)
    last_group = hist[:, -1]
    last_step = hist[:, :, -1]
    assert last_group.shape == (1, 4, 3)
    assert last_step.shape == (1, 2, 3)
    np.testing.assert_array_equal(last_group[0], hist[0, 1])
    assert not np.array_equal(last_group.reshape(-1)[:3], last_step.reshape(-1)[:3])
