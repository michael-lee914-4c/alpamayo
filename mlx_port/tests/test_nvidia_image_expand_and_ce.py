"""NVIDIA image-token expansion and next-token CE guards.

Weight-free. Complementary to open #30 (uint8 /255 + flatten) and
#20/#24 (two-mean CE / label masks) — Qwen preprocess expands each
``image_token`` to ``grid.prod() // merge_size**2`` placeholders,
collate ORs assistant EOS onto the train mask and zeros every label
in generation mode, and ``_compute_next_token_loss`` returns 0 on an
empty shift-mask and writes ``-inf`` for ``~token_mask``. Dropping
``merge_size**2`` would 4× the vision prefix.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_QWEN = _REPO / "src" / "alpamayo_r1" / "processor" / "qwen_processor.py"
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_IGNORE = -100


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_next_token_loss(torch):
    tree = ast.parse(_SFT_BASE.read_text())
    method = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TrainableReasoningVLA":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_compute_next_token_loss":
                    method = item
                    break
    if method is None:
        pytest.skip("_compute_next_token_loss missing")
    method.args.args = method.args.args[1:]  # drop self
    # Strip the dynamo decorator if present.
    method.decorator_list = []
    mod = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, object] = {
        "__name__": "extracted",
        "torch": torch,
        "IGNORE_INDEX": _IGNORE,
        "ModelOutput": object,
    }
    exec(compile(mod, str(_SFT_BASE), "exec"), ns)
    return ns["_compute_next_token_loss"]


def test_image_token_expand_is_grid_prod_over_merge_sq():
    src = _QWEN.read_text()
    assert "merge_length = processor.image_processor.merge_size**2" in src
    assert '"<|placeholder|>" * (image_inputs["image_grid_thw"][index].prod() // merge_length)' in src
    assert "assert index == len(image_inputs[\"image_grid_thw\"])" in src

    merge_size = 2
    merge_length = merge_size**2
    grid_prod = 1 * 4 * 4
    n_tokens = grid_prod // merge_length
    assert n_tokens == 4
    # Forgetting **2 leaves 16 vision tokens per 2×2 merged patch.
    assert grid_prod // merge_size == 8
    assert n_tokens != grid_prod // merge_size


def test_collate_or_assistant_eos_and_gen_mode_zeros_labels():
    src = _QWEN.read_text()
    assert "batched_data[\"labels_mask\"] |= eos_mask_assistant" in src
    assert "role=\"assistant\"" in src
    assert "batched_data[\"labels_mask\"] = torch.zeros_like(" in src
    assert "if not generation_mode:" in src


def test_next_token_loss_empty_mask_is_zero_and_token_mask_neg_inf(torch):
    fn = _load_next_token_loss(torch)
    logits = torch.zeros(1, 4, 5)
    labels = torch.tensor([[1, 2, 3, 4]])
    empty = torch.zeros_like(labels, dtype=torch.bool)
    loss = fn(SimpleNamespace(logits=logits), labels, labels_mask=empty)
    assert float(loss.item()) == 0.0

    # token_mask drops vocab id 0: the only valid class is 1, so CE is 0
    # when the kept logit is a one-hot at the label.
    logits = torch.full((1, 3, 3), -20.0)
    logits[0, 0, 1] = 10.0  # position 0 predicts label at position 1
    labels = torch.tensor([[9, 1, 1]])
    mask = torch.tensor([[False, True, False]])
    token_mask = torch.tensor([False, True, False])
    # Clone so the in-place -inf write cannot leak into later asserts.
    loss = fn(
        SimpleNamespace(logits=logits.clone()),
        labels,
        labels_mask=mask,
        token_mask=token_mask,
    )
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-5)
    # A missing token_mask would still be ~0 here because id 1 already
    # dominates; lock the write itself on a copy.
    scores = torch.zeros(1, 2, 4)
    token_mask = torch.tensor([True, False, True, False])
    scores[..., ~token_mask] = torch.finfo(scores.dtype).min
    assert float(scores[0, 0, 1].item()) < 0
    assert float(scores[0, 0, 0].item()) == 0.0
    src = _SFT_BASE.read_text()
    assert "shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min" in src
    assert "if labels_mask[:, 1:].sum() == 0:" in src
