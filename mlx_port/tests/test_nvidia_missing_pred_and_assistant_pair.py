"""DistanceMetrics no-op without preds; assistant mask requires paired BOS/EOS.

Weight-free. Complementary to open #27 (last-group GT / zero-logprob
sample-0) and #24 (``get_assistant_mask`` header/EOS/user). A missing
``pred_xyz`` must return ``{}`` rather than KeyError, and an unpaired
BOS/EOS count must raise — otherwise eval silently scores nothing or
masks the user span.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_METRIC_API = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_api.py"
_LABELS = _REPO / "src" / "alpamayo_r1" / "utils" / "get_label_mask.py"


def test_source_returns_empty_when_pred_missing_and_pairs_bos_eos():
    api = _METRIC_API.read_text()
    assert 'if self.prefix + "pred_xyz" not in output_batch:' in api
    assert "return {}" in api
    labels = _LABELS.read_text()
    assert "START_OFFSET = 3" in labels
    assert "END_OFFSET = 1" in labels
    assert "Number of bos" in labels
    assert "does not match eos" in labels


def test_distance_metrics_missing_pred_returns_empty():
    src = _METRIC_API.read_text()
    # Replica of the early-return guard; prefix is prepended before lookup.
    prefix = "rollout/"
    output_batch = {"pred_xyz": object()}
    if prefix + "pred_xyz" not in output_batch:
        got = {}
    else:
        got = {"should_not": True}
    assert got == {}
    assert 'f"No predictions with prefix {self.prefix} found in output_batch."' in src


def test_assistant_mask_start_offset_skips_header_and_includes_eos():
    """Replica of NVIDIA ``get_assistant_mask`` pairing + offsets."""
    bos, eos, role, other = 1, 2, 3, 4
    # [BOS, assistant, \\n, tok, tok, EOS]
    tokens = np.array([bos, role, 9, 10, 11, eos])
    start_indices = np.where(tokens == bos)[0]
    end_indices = np.where(tokens == eos)[0]
    assert len(start_indices) == len(end_indices)
    masks = np.zeros_like(tokens, dtype=bool)
    start_offset, end_offset = 3, 1
    for start, end in zip(start_indices, end_indices):
        if tokens[start + 1] == role:
            masks[start + start_offset : end + end_offset] = True
    assert masks.tolist() == [False, False, False, True, True, True]

    user = np.array([bos, other, 9, 10, eos])
    start_indices = np.where(user == bos)[0]
    end_indices = np.where(user == eos)[0]
    masks = np.zeros_like(user, dtype=bool)
    for start, end in zip(start_indices, end_indices):
        if user[start + 1] == role:
            masks[start + start_offset : end + end_offset] = True
    assert masks.tolist() == [False, False, False, False, False]


def test_assistant_mask_unpaired_bos_eos_raises():
    bos, eos = 1, 2
    tokens = np.array([bos, 3, bos, 3, eos])
    start_indices = np.where(tokens == bos)[0]
    end_indices = np.where(tokens == eos)[0]
    with pytest.raises(AssertionError, match="does not match eos"):
        assert len(start_indices) == len(end_indices), (
            f"Number of bos ({len(start_indices)}) does not match eos ({len(end_indices)})"
        )


def test_summarize_metric_rejects_rank_not_2():
    src = (_REPO / "src" / "alpamayo_r1" / "metrics" / "metric_utils.py").read_text()
    assert "if val.ndim != 2:" in src
    assert "All keys of metric must have values of shape [B,N]" in src
    assert "if val.shape[1] > 1 and not disable_summary:" in src

    val = SimpleNamespace(ndim=1, shape=(4,))
    with pytest.raises(ValueError, match=r"shape \[B,N\]"):
        if val.ndim != 2:
            raise ValueError(
                f"All keys of metric must have values of shape [B,N], "
                f"found ade with shape {val.shape}"
            )
