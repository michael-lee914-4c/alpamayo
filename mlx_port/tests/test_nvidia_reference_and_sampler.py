"""RL reference fallback and ReasoningSampler drop extras.

Weight-free. Complementary to open #27 (DistanceMetrics last-group /
missing-pred ``{}``) and #32 (policy labels). ``get_reference_answer``
swallows fetch errors and missing ``ego_future_xyz`` as ``{}``.
``ReasoningSampler.evaluate`` hardcodes ``return_extra=False`` and only
prefixes ``pred_xyz`` / ``pred_rot`` — the docstring's cot / meta /
logprob keys are never written.
"""

from __future__ import annotations

from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[2]
_DATASET = _REPO / "finetune" / "rl" / "base_dataset.py"
_API = _REPO / "src" / "alpamayo_r1" / "metrics" / "metric_api.py"


def _get_reference_answer(dataset, idx):
    try:
        sample = dataset[idx]
    except Exception:
        return {}
    if not isinstance(sample, dict) or "ego_future_xyz" not in sample:
        return {}
    return {
        "ego_future_xyz": sample["ego_future_xyz"],
        "ego_future_rot": sample["ego_future_rot"],
        "ego_history_xyz": sample["ego_history_xyz"],
        "ego_history_rot": sample["ego_history_rot"],
        "egomotion_road_boundaries": sample.get("egomotion_road_boundaries", None),
        "egomotion_lanelines": sample.get("egomotion_lanelines", None),
        "ego_lwh": sample.get("ego_lwh", None),
        "ego_length_offset": sample.get("ego_length_offset", None),
        "obstacle_bbox_history": sample.get("obstacle_bbox_history", None),
        "obstacle_bbox_future": sample.get("obstacle_bbox_future", None),
        "cot": sample.get("cot", ""),
    }


def _sampler_evaluate(model, data_batch, output_batch, prefix=""):
    pred_xyz, pred_rot = model.sample_trajectories_from_data(
        data=data_batch,
        num_traj_samples=6,
        num_traj_sets=1,
        top_p=0.98,
        temperature=0.6,
        traj_only_generation=False,
        max_generation_length=256,
        return_extra=False,
    )
    output_batch.update(
        {
            f"{prefix}pred_xyz": pred_xyz,
            f"{prefix}pred_rot": pred_rot,
        }
    )
    output_batch.update(
        {
            "absolute_timestamps": data_batch.get("absolute_timestamps", None),
            "relative_timestamps": data_batch.get("relative_timestamps", None),
            "ego_history_xyz": data_batch.get("ego_history_xyz", None),
            "ego_history_rot": data_batch.get("ego_history_rot", None),
            "ego_future_xyz": data_batch.get("ego_future_xyz", None),
            "ego_future_rot": data_batch.get("ego_future_rot", None),
        }
    )
    return {}


def test_source_reference_empty_and_sampler_drops_extra():
    ds = _DATASET.read_text()
    assert "return {}" in ds
    assert '"ego_future_xyz" not in sample' in ds
    assert 'sample.get("cot", "")' in ds
    api = _API.read_text()
    assert "return_extra=False" in api
    assert "traj_only_generation=False" in api
    assert '"pred_xyz": pred_xyz' in api
    assert '"pred_rot": pred_rot' in api
    assert "pred_answer" not in api.split("output_batch.update")[1]


def test_reference_exception_and_missing_future_are_empty():
    class _Boom:
        def __getitem__(self, idx):
            raise KeyError(idx)

    assert _get_reference_answer(_Boom(), 0) == {}
    assert _get_reference_answer({0: {"cot": "x"}}, 0) == {}
    assert _get_reference_answer({0: "not-a-dict"}, 0) == {}


def test_reference_keeps_required_keys_and_optional_defaults():
    sample = {
        "ego_future_xyz": [1],
        "ego_future_rot": [2],
        "ego_history_xyz": [3],
        "ego_history_rot": [4],
        "extra_ignored": True,
    }
    got = _get_reference_answer({7: sample}, 7)
    assert got["ego_future_xyz"] == [1]
    assert got["ego_history_rot"] == [4]
    assert got["cot"] == ""
    assert got["ego_lwh"] is None
    assert "extra_ignored" not in got


def test_sampler_writes_only_xyz_rot_and_returns_empty_metrics():
    pred_xyz = torch.zeros(1, 1, 2, 4, 3)
    pred_rot = torch.eye(3).expand(1, 1, 2, 4, 3, 3).clone()

    class _Model:
        def sample_trajectories_from_data(self, **kwargs):
            assert kwargs["return_extra"] is False
            assert kwargs["traj_only_generation"] is False
            self.kwargs = kwargs
            return pred_xyz, pred_rot

    model = _Model()
    data = {
        "ego_future_xyz": torch.ones(1, 1, 4, 3),
        "absolute_timestamps": torch.tensor([1.0]),
    }
    out = {"keep": True}
    metrics = _sampler_evaluate(model, data, out, prefix="eval/")
    assert metrics == {}
    assert out["eval/pred_xyz"] is pred_xyz
    assert out["eval/pred_rot"] is pred_rot
    assert "eval/cot" not in out
    assert "pred_answer" not in out
    assert "logprob" not in out
    assert out["absolute_timestamps"].tolist() == [1.0]
    assert out["ego_future_xyz"] is data["ego_future_xyz"]
    assert out["keep"] is True
