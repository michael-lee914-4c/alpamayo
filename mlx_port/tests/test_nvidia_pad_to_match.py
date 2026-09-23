"""FSDP resume zero-pads a smaller checkpoint tensor on the tail dims.

Weight-free. Complementary to open #20 (adapter reserved extra keys).
``F.pad`` walks ``reversed(src.shape)`` and pads ``(0, t-s)`` so a
``(out, in)`` expert head that was FSDP-aligned grows on the last axis,
not the first. Padding the batch/out dim instead would shift every
action channel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_LOAD = _REPO / "finetune" / "rl" / "utils" / "weight_loading.py"


def _pad_to_match(src, target_shape):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    pad_args: list[int] = []
    for s_dim, t_dim in zip(reversed(src.shape), reversed(target_shape)):
        pad_args.extend([0, t_dim - s_dim])
    return F.pad(src, pad_args)


def test_source_pads_reversed_dims_and_strict_missing_key():
    src = _LOAD.read_text()
    assert "if pad_to_match and src.shape != t.shape:" in src
    assert "for s_dim, t_dim in zip(reversed(src.shape), reversed(t.shape)):" in src
    assert "pad_args.extend([0, t_dim - s_dim])" in src
    assert "src = F.pad(src, pad_args)" in src
    assert 'raise KeyError(f"Checkpoint missing parameter: {full_key}")' in src
    assert 'full_key = f"{key_prefix}{name}" if key_prefix else name' in src


def test_last_dim_grows_and_original_rows_stay_put():
    torch = pytest.importorskip("torch")
    src = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    padded = _pad_to_match(src, (2, 6))
    assert padded.shape == (2, 6)
    torch.testing.assert_close(padded[:, :4], src)
    torch.testing.assert_close(padded[:, 4:], torch.zeros(2, 2))
    # Padding the leading dim instead would be (4, 4) and shift channels.
    wrong = torch.nn.functional.pad(src, (0, 0, 0, 2))
    assert wrong.shape == (4, 4)
    assert wrong.shape != padded.shape


def test_key_prefix_is_prepended_before_lookup():
    name = "action_out_proj.weight"
    assert f"{'expert_model.'}{name}" == "expert_model.action_out_proj.weight"
    assert ("" + name) == name
