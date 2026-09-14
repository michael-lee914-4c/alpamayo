"""NVIDIA infer sample layout: last-hist group, ns/nj, num_return_sequences.

Weight-free. Complementary to open #21 (MLX last-of-n_traj squeeze) and
#27 (eval last-group) — the CUDA infer path repeats ``ego_history[:, -1]``,
rearranges ``(b ns nj)`` with ``ns=num_traj_sets`` / ``nj=num_traj_samples``,
and sets ``num_return_sequences = num_traj_samples`` (not ``* num_traj_sets``).
A first-group or ns/nj swap would score the wrong sample grid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_ALPAMAYO = _SRC / "alpamayo_r1" / "models" / "alpamayo_r1.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_source_num_return_sequences_is_samples_not_sets_times_samples():
    src = _ALPAMAYO.read_text()
    assert "generation_config.num_return_sequences = num_traj_samples" in src
    assert "num_return_sequences = num_traj_samples * num_traj_sets" not in src
    assert "n_samples_total = num_traj_samples * num_traj_sets" in src
    assert 'ns=num_traj_sets, nj=num_traj_samples' in src
    assert "ego_history_xyz[:, -1]" in src
    assert "ego_history_rot[:, -1]" in src


def test_hist_repeat_uses_last_traj_group_not_first(torch):
    einops = pytest.importorskip("einops")
    # B=2, n_traj_group=3. First group is 0s, last group is 9s.
    hist = torch.zeros(2, 3, 4, 3)
    hist[:, -1, :, 0] = 9.0
    hist[:, 0, :, 0] = -1.0
    n_samples_total = 6  # 2 sets * 3 samples
    rep = einops.repeat(hist[:, -1], "b ... -> (b n) ...", n=n_samples_total)
    assert rep.shape == (12, 4, 3)
    assert torch.allclose(rep[..., 0], torch.full((12, 4), 9.0))
    # A first-group swap would copy -1, not 9.
    first = einops.repeat(hist[:, 0], "b ... -> (b n) ...", n=n_samples_total)
    assert not torch.allclose(rep, first)


def test_rearrange_is_sets_then_samples_not_swapped(torch):
    einops = pytest.importorskip("einops")
    num_traj_sets, num_traj_samples = 2, 3
    b = 1
    # Flat index i maps to (set=i//3, sample=i%3) under (b ns nj).
    flat = torch.arange(b * num_traj_sets * num_traj_samples, dtype=torch.float32).view(
        b * num_traj_sets * num_traj_samples, 1, 1
    )
    xyz = einops.rearrange(
        flat, "(b ns nj) ... -> b ns nj ...", ns=num_traj_sets, nj=num_traj_samples
    )
    assert xyz.shape == (1, 2, 3, 1, 1)
    assert xyz[0, 0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert xyz[0, 1, :, 0, 0].tolist() == [3.0, 4.0, 5.0]
    swapped = einops.rearrange(
        flat, "(b ns nj) ... -> b ns nj ...", ns=num_traj_samples, nj=num_traj_sets
    )
    assert swapped.shape == (1, 3, 2, 1, 1)
    assert swapped[0, 0, :, 0, 0].tolist() != xyz[0, 0, :, 0, 0].tolist()


def test_infer_attention_mask_neg_inf_from_offset_to_prefill_tail(torch):
    # Cache length 8, n_diff=2, offset=3 → mask columns [3, 6) i.e. padding
    # after the crop, not the diffusion tokens at the end.
    n_diff = 2
    cache_len = 8
    offset = 3
    mask = torch.zeros((1, 1, n_diff, cache_len + n_diff), dtype=torch.float32)
    mask[0, :, :, offset:-n_diff] = torch.finfo(mask.dtype).min
    # Columns 0..2 stay 0 (kept prefix); 3..5 are -inf; 8..9 (diff tokens) stay 0.
    assert torch.all(mask[0, 0, 0, :offset] == 0)
    assert torch.all(mask[0, 0, 0, offset:-n_diff] < 0)
    assert torch.all(mask[0, 0, 0, -n_diff:] == 0)
    src = _ALPAMAYO.read_text()
    assert "offset[i] : -n_diffusion_tokens" in src
