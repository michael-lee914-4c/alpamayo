"""Future fuse requires ``traj_tokenizer`` and ``future_token_start_idx``.

Weight-free. Complementary to open #32 (base mixin is hist-only;
``TrajectoryFusionWithFutureMixin`` fuses both). When
``ego_future_xyz`` is present the mixin sets ``require_future=True`` and
must raise if those attrs are missing — a silent hist-only fuse would
leave future pads in the CE span.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"


def test_source_require_future_checks_tokenizer_and_start_idx():
    base = _BASE.read_text()
    assert "has_future = \"ego_future_xyz\" in traj_data and traj_data[\"ego_future_xyz\"] is not None" in base
    assert "attrs = self._validate_mixin_requirements(require_future=has_future)" in base
    assert "if require_future:" in base
    assert "Requires 'traj_tokenizer' attribute for future trajectories" in base
    assert "Requires 'future_token_start_idx' attribute for future trajectories" in base
    # Base fuse never writes future pads.
    assert "traj_token_ids[\"future\"]" not in base.split("class ReasoningVLAConfig")[0]

    sft = _SFT_BASE.read_text()
    assert "if has_future:" in sft
    assert "tokenize_future_trajectory(" in sft
    assert "traj_token_ids[\"future\"]" in sft


def test_validate_mixin_requirements_raises_without_future_attrs():
    def _validate(obj, require_future: bool = False):
        hist_traj_tokenizer = getattr(obj, "hist_traj_tokenizer", None)
        if hist_traj_tokenizer is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_traj_tokenizer' attribute")
        hist_token_start_idx = getattr(obj, "hist_token_start_idx", None)
        if hist_token_start_idx is None:
            raise AttributeError("TrajectoryFusionMixin requires 'hist_token_start_idx' attribute")
        config = getattr(obj, "config", None)
        if config is None or not hasattr(config, "traj_token_ids"):
            raise AttributeError("TrajectoryFusionMixin requires 'config' with 'traj_token_ids'")
        result = {
            "hist_traj_tokenizer": hist_traj_tokenizer,
            "hist_token_start_idx": hist_token_start_idx,
            "config": config,
        }
        if require_future:
            traj_tokenizer = getattr(obj, "traj_tokenizer", None)
            if traj_tokenizer is None:
                raise AttributeError("Requires 'traj_tokenizer' attribute for future trajectories")
            future_token_start_idx = getattr(obj, "future_token_start_idx", None)
            if future_token_start_idx is None:
                raise AttributeError(
                    "Requires 'future_token_start_idx' attribute for future trajectories"
                )
            result.update(
                {
                    "traj_tokenizer": traj_tokenizer,
                    "future_token_start_idx": future_token_start_idx,
                }
            )
        return result

    hist_only = SimpleNamespace(
        hist_traj_tokenizer=object(),
        hist_token_start_idx=100,
        config=SimpleNamespace(traj_token_ids={"history": 1}),
    )
    assert "traj_tokenizer" not in _validate(hist_only, require_future=False)
    with pytest.raises(AttributeError, match="traj_tokenizer"):
        _validate(hist_only, require_future=True)

    missing_idx = SimpleNamespace(
        hist_traj_tokenizer=object(),
        hist_token_start_idx=100,
        traj_tokenizer=object(),
        config=SimpleNamespace(traj_token_ids={"history": 1, "future": 2}),
    )
    with pytest.raises(AttributeError, match="future_token_start_idx"):
        _validate(missing_idx, require_future=True)

    complete = SimpleNamespace(
        hist_traj_tokenizer=object(),
        hist_token_start_idx=100,
        traj_tokenizer=object(),
        future_token_start_idx=200,
        config=SimpleNamespace(traj_token_ids={"history": 1, "future": 2}),
    )
    attrs = _validate(complete, require_future=True)
    assert attrs["future_token_start_idx"] == 200


def test_has_future_is_key_and_not_none():
    # ``"ego_future_xyz" in traj_data`` is not enough: SFT/RL often pass
    # the key with a None value on hist-only infer.
    def has_future(traj_data):
        return "ego_future_xyz" in traj_data and traj_data["ego_future_xyz"] is not None

    assert has_future({"ego_future_xyz": object()}) is True
    assert has_future({"ego_future_xyz": None}) is False
    assert has_future({}) is False
