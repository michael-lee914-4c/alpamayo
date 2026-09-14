"""NVIDIA expert RoPE: train uses KV length; infer uses first start+1.

Weight-free. Complementary to open #28 (SFT crop is *last*
``<|traj_future_start|>`` + 1) — Stage-2 *train* then adds
``rope_deltas + kv_seq_len``, while infer uses the *first* start
(``argmax``) + 1. Unifying those two would shift expert positions on
left-pad / multi-start batches and look like a CFM regression.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_train_position_ids(torch):
    einops = pytest.importorskip("einops")
    path = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"
    tree = ast.parse(path.read_text())
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainableAlpamayoR1":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_process_position_ids_qwen2_5_vl":
                    method = item
                    break
    if method is None:
        pytest.skip("train _process_position_ids_qwen2_5_vl missing")
    # Bind as a free function: drop ``self``.
    method.args.args = method.args.args[1:]
    mod = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns = {"__name__": "extracted", "torch": torch, "einops": einops, "Any": object}
    exec(compile(mod, str(path), "exec"), ns)
    return ns["_process_position_ids_qwen2_5_vl"]


def _infer_offset_and_position_ids(torch, sequences, future_start_id, rope_deltas, n_diff):
    """Mirror ``AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout``."""
    einops = pytest.importorskip("einops")
    b_star, seq_len = sequences.shape
    device = sequences.device
    start_mask = sequences == future_start_id
    has_start = start_mask.any(dim=1)
    first_start = start_mask.int().argmax(dim=1)
    last_token = torch.full((b_star,), seq_len - 1, device=device)
    valid = torch.where(has_start, first_start, last_token)
    offset = valid + 1
    position_ids = torch.arange(n_diff, device=device)
    position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star).clone()
    delta = rope_deltas + offset[:, None]
    position_ids = position_ids + delta.to(position_ids.device)
    return offset, position_ids


def test_train_position_ids_are_3row_plus_rope_and_kv_len(torch):
    fn = _load_train_position_ids(torch)
    rope = torch.tensor([[[4]], [[4]], [[4]]], dtype=torch.long)  # (3, B, 1)
    kv = SimpleNamespace(get_seq_length=lambda: 10)
    vlm_outputs = SimpleNamespace(rope_deltas=rope, past_key_values=kv)
    ids = fn(vlm_outputs, batch_size=1, num_expert_tokens=5, device=torch.device("cpu"))
    assert ids.shape == (3, 1, 5)
    # arange(5) + 4 + 10 → [14, 15, 16, 17, 18] on every RoPE row.
    expect = torch.arange(5) + 14
    assert torch.equal(ids[0, 0], expect)
    assert torch.equal(ids[1, 0], expect)
    assert torch.equal(ids[2, 0], expect)


def test_infer_offset_is_first_start_plus_one_not_last(torch):
    # Two starts: last-start crop (#28) would pick index 7; infer argmax picks 2.
    sequences = torch.tensor([[0, 0, 99, 1, 1, 1, 1, 99, 2, 2]], dtype=torch.long)
    rope = torch.tensor([[[3]], [[3]], [[3]]], dtype=torch.long)
    offset, ids = _infer_offset_and_position_ids(
        torch, sequences, future_start_id=99, rope_deltas=rope, n_diff=4
    )
    assert offset.tolist() == [3]  # first start 2 + 1
    assert ids.shape == (3, 1, 4)
    expect = torch.arange(4) + 3 + 3
    assert torch.equal(ids[0, 0], expect)

    src = (_SRC / "alpamayo_r1" / "models" / "alpamayo_r1.py").read_text()
    assert "traj_future_start_mask.int().argmax(dim=1)" in src
    assert "offset = valid_token_pos_id + 1" in src
    # Train crop (last nonzero + 1) must stay distinct from this first-start path.
    train_src = (_REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py").read_text()
    assert "last_traj_future_start_idx[-1, 1] + 1" in train_src


def test_infer_missing_start_uses_last_token_plus_one(torch):
    sequences = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    rope = torch.zeros(3, 1, 1, dtype=torch.long)
    offset, ids = _infer_offset_and_position_ids(
        torch, sequences, future_start_id=99, rope_deltas=rope, n_diff=2
    )
    assert offset.tolist() == [4]  # last index 3 + 1
    assert torch.equal(ids[0, 0], torch.tensor([4, 5]))
