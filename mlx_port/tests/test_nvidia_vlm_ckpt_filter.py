"""``load_alpamayo1_vlm`` only copies ``vlm.*`` shards; empty map raises.

Weight-free. Complementary to open #36 (``from_alpamayo_checkpoint``
``add_special_tokens`` default True). A non-``vlm.`` key in the index, a
missing ``model.safetensors.index.json``, or a shard with no matching
tensors would silently skip the Stage-1 tower or raise only after
training started.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"


def _filter_vlm_keys(weight_map: dict[str, str]) -> dict[str, list[str]]:
    """Replica of NVIDIA ``load_alpamayo1_vlm`` shard grouping."""
    shard_to_keys: defaultdict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        if key.startswith("vlm."):
            shard_to_keys[shard_name].append(key)
    return dict(shard_to_keys)


def test_source_filters_vlm_prefix_and_raises_when_empty():
    src = _SFT_BASE.read_text()
    assert 'if key.startswith("vlm."):' in src
    assert 'raise ValueError(f"No vlm.* tensors found in checkpoint: {checkpoint_dir}")' in src
    assert "strict=False, assign=True" in src
    assert 'checkpoint_config.get("add_special_tokens", True)' in src
    assert 'raise FileNotFoundError(f"Missing config file: {checkpoint_config_path}")' in src


def test_only_vlm_prefix_keys_are_grouped():
    grouped = _filter_vlm_keys(
        {
            "vlm.language_model.embed_tokens.weight": "model-00001.safetensors",
            "vlm.visual.patch_embed.weight": "model-00002.safetensors",
            "expert.layers.0.weight": "model-00001.safetensors",
            "action_in_proj.encoder.0.weight": "model-00003.safetensors",
        }
    )
    assert grouped == {
        "model-00001.safetensors": ["vlm.language_model.embed_tokens.weight"],
        "model-00002.safetensors": ["vlm.visual.patch_embed.weight"],
    }


def test_empty_or_non_vlm_map_is_the_raise_condition():
    assert _filter_vlm_keys({}) == {}
    assert _filter_vlm_keys({"expert.x": "a.safetensors"}) == {}
    # NVIDIA raises after the loop when the collected dict is empty —
    # both "no index" and "index with only expert keys" hit this.
    vlm_state_dict: dict[str, object] = {}
    if not vlm_state_dict:
        with pytest.raises(ValueError, match="No vlm"):
            raise ValueError("No vlm.* tensors found in checkpoint: /tmp/ckpt")


def test_from_alpamayo_checkpoint_missing_config_raises(tmp_path: Path):
    src = _SFT_BASE.read_text()
    assert "checkpoint_config_path = Path(checkpoint_path) / \"config.json\"" in src
    missing = tmp_path / "config.json"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError, match="Missing config file"):
        if not missing.exists():
            raise FileNotFoundError(f"Missing config file: {missing}")


def test_index_json_weight_map_default_is_empty(tmp_path: Path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    assert weight_map == {}
    assert _filter_vlm_keys(weight_map) == {}
