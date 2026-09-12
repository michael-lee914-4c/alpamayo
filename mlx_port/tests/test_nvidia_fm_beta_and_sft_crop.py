"""FM beta inversion, Euler trace, SFT last-start crop, metric/ prefix.

Weight-free. Complementary to open #25 (interp/MSE/Euler Δ=1) and #20
(MLX KV crop) — this locks the CUDA beta ``t = 0.999*(1-u)`` flip, the
N+1 Euler trace, the SFT ``last start + 1`` crop column, IGNORE_INDEX
masking, and MetricRunner's ``metric/`` rewrite.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_flow_matching():
    _ensure_src_on_path()
    try:
        from alpamayo_r1.diffusion.flow_matching import FlowMatching
    except Exception as exc:
        pytest.skip(f"NVIDIA FlowMatching unavailable: {exc}")
    return FlowMatching


def test_beta_sampler_inverts_and_caps_below_one(torch):
    FlowMatching = _load_flow_matching()
    fm = FlowMatching(x_dims=(2,), train_timestep_sampler="beta")
    # Replace the Beta with a Dirac so the inversion is exact: u=0.2 → t=0.7992.
    class _Dirac:
        def sample(self, shape):
            return torch.full(shape, 0.2)

    fm.beta_dist = _Dirac()
    torch.manual_seed(0)
    data = fm.construct_training_data(torch.ones(4, 2))
    t = data["timesteps"].reshape(4)
    assert torch.allclose(t, torch.full((4,), 0.999 * (1.0 - 0.2)))
    assert float(t.max()) <= 0.999
    assert float(t.min()) > 0.0


def test_euler_return_all_steps_includes_noise_and_endpoint(torch):
    FlowMatching = _load_flow_matching()
    fm = FlowMatching(x_dims=(3,), num_inference_steps=4)

    def step_fn(*, x, t):
        return torch.ones_like(x)

    torch.manual_seed(0)
    xs, times = fm.sample(
        batch_size=2,
        step_fn=step_fn,
        return_all_steps=True,
        inference_step=4,
    )
    assert xs.shape == (2, 5, 3)
    assert times.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    # Constant v=1, dt=0.25 → each step adds 0.25; last - first = 1.
    assert torch.allclose(xs[:, -1] - xs[:, 0], torch.ones(2, 3), atol=1e-5)


def test_sft_crop_uses_last_future_start_plus_one(torch):
    # Mirrors TrainableAlpamayoR1.forward: last match column + 1 is the crop.
    future_start = 99
    input_ids = torch.tensor([[1, 99, 2, 99, 3, 4]], dtype=torch.long)
    last = (input_ids == future_start).nonzero(as_tuple=False)
    crop = last[-1, 1] + 1
    assert int(crop.item()) == 4
    src = (_REPO / "finetune" / "sft" / "models" / "sft_alpamayo_r1.py").read_text()
    assert "last_traj_future_start_idx[-1, 1] + 1" in src
    assert "IGNORE_INDEX" in src


def test_ignore_index_is_negative_100_and_masks_unlabeled(torch):
    src = (_SRC / "alpamayo_r1" / "models" / "base_model.py").read_text()
    assert "IGNORE_INDEX = -100" in src
    ignore = -100
    ids = torch.tensor([[7, 8, 9]])
    mask = torch.tensor([[True, False, True]])
    labels = torch.where(mask, ids, torch.tensor(ignore))
    assert labels.tolist() == [[7, -100, 9]]


def _load_metric_runner():
    _ensure_src_on_path()
    api = ModuleType("alpamayo_r1.metrics.metric_api")

    class Metric:
        def evaluate(self, model, data_batch, output_batch):
            raise NotImplementedError

    api.Metric = Metric
    sys.modules["alpamayo_r1.metrics.metric_api"] = api
    path = _SRC / "alpamayo_r1" / "metrics" / "metric_runner.py"
    spec = importlib.util.spec_from_file_location("nv_metric_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA MetricRunner unavailable: {exc}")
    return module, Metric


def test_metric_runner_prefixes_keys(torch):
    mod, Metric = _load_metric_runner()

    class _M(Metric):
        def evaluate(self, model, data_batch, output_batch):
            return {"ade": torch.tensor([1.5]), "min_ade": torch.tensor([0.8])}

    out = {"pred_xyz": torch.zeros(1)}
    runner = mod.MetricRunner(metrics=[_M()])
    runner.run(None, {}, out)
    assert set(out) >= {"pred_xyz", "metric/ade", "metric/min_ade"}
    assert "ade" not in out
    assert float(out["metric/ade"].item()) == pytest.approx(1.5)
