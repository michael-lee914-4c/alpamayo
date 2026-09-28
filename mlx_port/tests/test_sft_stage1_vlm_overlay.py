"""Stage-1 VLM overlay must actually land on the Stage-2 module.

Official Stage-2 (`docs/FINETUNE_SFT.md`) loads Alpamayo-R1 via
``TrainableAlpamayoR1.from_pretrained`` then overlays a Stage-1 Trainer
directory. Two bugs made that overlay a no-op:

1. ``__init__`` passed the inner Qwen VLM, whose keys have no ``vlm.``
   prefix, while Trainer shards are ``vlm.*``. ``strict=False`` applied
   nothing.
2. HuggingFace ``from_pretrained`` copies the release checkpoint *after*
   ``__init__``, so even a successful overlay would be overwritten.

Weight-free. The align helper has no torch dependency; the loader test
uses a tiny safetensors shard when torch is available.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finetune.sft.models.vlm_keys import (
    align_vlm_state_dict_keys,
    apply_aligned_vlm_state_dict,
)

_REPO = Path(__file__).resolve().parents[2]
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"
_SFT_AR1 = _REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py"


def test_align_keeps_vlm_prefix_for_wrapper_destination():
    src = {"vlm.model.w": 1, "vlm.lm_head.weight": 2}
    dest = {"vlm.model.w", "vlm.lm_head.weight", "expert.w"}
    assert align_vlm_state_dict_keys(src, dest) == src


def test_align_strips_vlm_prefix_for_inner_qwen_destination():
    src = {"vlm.model.w": 1, "vlm.lm_head.weight": 2}
    dest = {"model.w", "lm_head.weight"}
    assert align_vlm_state_dict_keys(src, dest) == {
        "model.w": 1,
        "lm_head.weight": 2,
    }


def test_align_does_not_strip_when_source_has_no_prefix():
    src = {"model.w": 1}
    dest = {"model.w"}
    assert align_vlm_state_dict_keys(src, dest) == src


def test_from_pretrained_overlays_stage1_after_base_load():
    src = _SFT_AR1.read_text(encoding="utf-8")
    pop_at = src.index('kwargs.pop("stage1_vlm_checkpoint_path"')
    super_at = src.index("super().from_pretrained")
    overlay_at = src.index("load_alpamayo1_vlm(stage1_vlm_checkpoint_path, model)")
    assert pop_at < super_at < overlay_at
    # Direct construction must overlay the wrapper, not the inner VLM.
    assert "load_alpamayo1_vlm(stage1_vlm_checkpoint_path, self)" in src
    assert "load_alpamayo1_vlm(stage1_vlm_checkpoint_path, self.vlm)" not in src


class _FakeModule:
    """Minimal ``state_dict`` / ``load_state_dict`` stand-in (no torch)."""

    def __init__(self, values: dict[str, object]):
        self._sd = dict(values)

    def state_dict(self) -> dict[str, object]:
        return self._sd

    def load_state_dict(self, sd, strict=False, assign=True):
        unexpected = [k for k in sd if k not in self._sd]
        missing = [k for k in self._sd if k not in sd]
        for key, value in sd.items():
            if key in self._sd:
                self._sd[key] = value
        return SimpleNamespace(unexpected_keys=unexpected, missing_keys=missing)


def test_apply_writes_wrapper_keys_and_leaves_expert():
    model = _FakeModule({"vlm.w": 0, "expert.w": 9})
    n_applied, result = apply_aligned_vlm_state_dict(model, {"vlm.w": 7})
    assert n_applied == 1
    assert model._sd["vlm.w"] == 7
    assert model._sd["expert.w"] == 9
    assert result.unexpected_keys == []


def test_apply_strips_prefix_onto_inner_vlm():
    model = _FakeModule({"w": 0})
    n_applied, _ = apply_aligned_vlm_state_dict(model, {"vlm.w": 4})
    assert n_applied == 1
    assert model._sd["w"] == 4


def test_apply_raises_when_no_keys_match():
    model = _FakeModule({"other": 0})
    with pytest.raises(ValueError, match="No vlm.* tensors matched parameters"):
        apply_aligned_vlm_state_dict(model, {"vlm.w": 1}, checkpoint_dir="/tmp/ckpt")


def test_loader_still_filters_vlm_shards_and_raises_when_empty():
    src = _SFT_BASE.read_text(encoding="utf-8")
    assert "apply_aligned_vlm_state_dict" in src
    assert 'if key.startswith("vlm."):' in src
    assert 'raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")' in src


def test_load_alpamayo1_vlm_applies_wrapper_and_inner_keys(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    try:
        from finetune.sft.models.sft_base_model import load_alpamayo1_vlm
    except Exception as exc:  # transformers/hydra differ across CI images
        pytest.skip(f"sft_base_model import unavailable: {exc}")

    class InnerVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(2))

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vlm = InnerVLM()
            self.expert = torch.nn.Linear(2, 2, bias=False)

    shard = {"vlm.w": torch.tensor([3.0, 4.0]), "expert.weight": torch.ones(2, 2)}
    shard_name = "model-00001-of-00001.safetensors"
    save_file(shard, str(tmp_path / shard_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: shard_name for k in shard}}),
        encoding="utf-8",
    )

    wrapper = Wrapper()
    expert_before = wrapper.expert.weight.detach().clone()
    load_alpamayo1_vlm(str(tmp_path), wrapper)
    assert torch.equal(wrapper.vlm.w.data, torch.tensor([3.0, 4.0]))
    # Expert must stay at its constructed value (Stage-1 overlay is VLM-only).
    assert torch.equal(wrapper.expert.weight.data, expert_before)

    inner = InnerVLM()
    load_alpamayo1_vlm(str(tmp_path), inner)
    assert torch.equal(inner.w.data, torch.tensor([3.0, 4.0]))


def test_load_alpamayo1_vlm_raises_on_key_mismatch(tmp_path: Path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    try:
        from finetune.sft.models.sft_base_model import load_alpamayo1_vlm
    except Exception as exc:
        pytest.skip(f"sft_base_model import unavailable: {exc}")

    class Other(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.other = torch.nn.Parameter(torch.zeros(2))

    shard_name = "model-00001-of-00001.safetensors"
    save_file({"vlm.w": torch.tensor([1.0, 2.0])}, str(tmp_path / shard_name))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"vlm.w": shard_name}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No vlm.* tensors matched parameters"):
        load_alpamayo1_vlm(str(tmp_path), Other())
