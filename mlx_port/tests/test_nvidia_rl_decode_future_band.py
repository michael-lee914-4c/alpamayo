"""RL decode subtracts ``traj_token_start_idx`` (the future band).

Weight-free. Complementary to open #34 (hist tokenizer
``start_idx += future.vocab_size``) and #32 (``rsplit`` last-start). History
IDs live in ``[traj_token_start_idx + V, ...)``; future IDs stay in
``[traj_token_start_idx, traj_token_start_idx + V)``. Decode feeds
``model_config.traj_token_start_idx`` into ``extract_traj_tokens``. Using the
hist offset would clamp every bin to 0 and zero the GRPO ADE / comfort.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEC = _REPO / "finetune" / "rl" / "utils" / "trajectory_decode.py"
_STATE = _REPO / "finetune" / "rl" / "state.py"
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


def test_source_decode_uses_traj_token_start_idx():
    src = _DEC.read_text()
    assert "token_id_offset = model_config.traj_token_start_idx" in src
    assert "expected_len = model_config.tokens_per_future_traj" in src
    assert "extract_traj_tokens(" in src
    assert "token_id_offset," in src
    # Must not subtract the hist-shifted index.
    assert "hist_token_start_idx" not in src
    assert "future_token_start_idx" not in src


def test_source_hist_band_is_shifted_by_future_vocab():
    state = _STATE.read_text()
    assert "self.hist_token_start_idx = self.future_token_start_idx = self.config.traj_token_start_idx" in state
    assert "self.hist_token_start_idx += self.traj_tokenizer.vocab_size" in state
    base = _BASE.read_text()
    assert "self.future_token_start_idx = self.hist_token_start_idx = config.traj_token_start_idx" in base
    assert "self.hist_token_start_idx += self.traj_tokenizer.vocab_size" in base


def test_future_band_offset_recovers_bins_hist_offset_does_not():
    start = 1000
    vocab = 10
    hist_start = start + vocab
    raw_ids = [start + 3, start + 7]
    future_bins = [i - start for i in raw_ids]
    hist_bins = [max(0, min(vocab - 1, i - hist_start)) for i in raw_ids]
    assert future_bins == [3, 7]
    # Hist offset underflows and clamps to 0.
    assert hist_bins == [0, 0]
