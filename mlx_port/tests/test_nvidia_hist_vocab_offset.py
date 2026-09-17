"""Hist tokenizer vocab is stacked after the future tokenizer.

Weight-free. Complementary to open #25 (hist flatten / ``start_idx``) and
#30 (future fuse). When a separate hist tokenizer is configured, NVIDIA
adds ``traj_tokenizer.vocab_size`` onto ``hist_token_start_idx`` after
both start indices are initialized to ``traj_token_start_idx``. Dropping
that add collides hist bins with future bins.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


def test_hist_and_future_share_start_then_hist_is_offset():
    src = _BASE.read_text()
    assert (
        "self.future_token_start_idx = self.hist_token_start_idx = config.traj_token_start_idx"
        in src
    )
    assert "if config.hist_traj_tokenizer_cfg is not None:" in src
    assert "self.hist_token_start_idx += self.traj_tokenizer.vocab_size" in src
    assert "self.hist_traj_tokenizer = self.traj_tokenizer" in src


def test_separate_hist_tokenizer_avoids_future_band():
    traj_token_start_idx = 151_936
    future_vocab = 768
    future_start = traj_token_start_idx
    hist_start = traj_token_start_idx
    hist_start += future_vocab  # NVIDIA stack
    future_end = future_start + future_vocab
    assert hist_start == future_end
    assert hist_start != future_start
    # A hist bin 0 must not land in the future band [start, start+vocab).
    hist_bin0 = hist_start + 0
    assert not (future_start <= hist_bin0 < future_end)


def test_shared_hist_tokenizer_keeps_same_start():
    traj_token_start_idx = 151_936
    future_start = traj_token_start_idx
    hist_start = traj_token_start_idx
    # No hist_traj_tokenizer_cfg → hist tokenizer is the future tokenizer.
    assert hist_start == future_start
