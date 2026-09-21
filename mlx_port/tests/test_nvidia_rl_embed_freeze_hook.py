"""RL ``freeze_base_model_except_embeddings`` zeros base-vocab grads.

Weight-free. Complementary to open #38 (Stage-2 ``cotrain_vlm=False``
freezes the whole VLM). The RL hook trains only the *new* embed rows;
without it GRPO would overwrite the pretrained Qwen table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_RL_BASE = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "base_model.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_source_hook_zeros_original_vocab():
    src = _RL_BASE.read_text()
    assert "def freeze_base_model_except_embeddings(self) -> None:" in src
    assert "for param in self.parameters():" in src
    assert "param.requires_grad = False" in src
    assert "self.vlm.language_model.embed_tokens.weight.requires_grad = True" in src
    assert "grad[: self.original_vocab_size] = 0" in src
    assert "self.vlm.language_model.embed_tokens.weight.register_hook(reset_grad)" in src


def test_embed_hook_keeps_only_new_token_grads(torch):
    original_vocab_size = 4
    weight = torch.nn.Parameter(torch.ones(6, 3))
    weight.requires_grad = True

    def reset_grad(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.clone()
        grad[:original_vocab_size] = 0
        return grad

    weight.register_hook(reset_grad)
    (weight * torch.ones_like(weight)).sum().backward()
    assert weight.grad is not None
    assert torch.all(weight.grad[:original_vocab_size] == 0)
    assert torch.all(weight.grad[original_vocab_size:] == 1)

    # A missing slice would train the whole table.
    raw = torch.ones_like(weight)
    assert not torch.equal(weight.grad, raw)
