"""RL decode helpers that #22 did not cover: traj-logit mask and empty-span decode.

Weight-free. Needs torch (macos-26 mlx-ci). Loads modules by path so cosmos-rl
is not required.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_FINETUNE = _REPO / "finetune"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC), str(_FINETUNE)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _stub_transformers_if_missing() -> None:
    try:
        import transformers  # noqa: F401
        if hasattr(transformers, "AutoTokenizer"):
            return
    except ImportError:
        pass
    stub = sys.modules.get("transformers")
    if stub is None or not hasattr(stub, "AutoTokenizer"):
        stub = ModuleType("transformers")
    stub.AutoProcessor = getattr(stub, "AutoProcessor", object)
    stub.AutoTokenizer = getattr(stub, "AutoTokenizer", object)
    stub.StoppingCriteria = getattr(stub, "StoppingCriteria", type("StoppingCriteria", (), {}))
    gen = sys.modules.get("transformers.generation") or ModuleType("transformers.generation")
    logits = sys.modules.get("transformers.generation.logits_process") or ModuleType(
        "transformers.generation.logits_process"
    )
    if not hasattr(logits, "LogitsProcessor"):
        logits.LogitsProcessor = type("LogitsProcessor", (), {})
    stub.generation = gen
    sys.modules["transformers"] = stub
    sys.modules["transformers.generation"] = gen
    sys.modules["transformers.generation.logits_process"] = logits


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_logit_processor():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    path = _FINETUNE / "rl" / "utils" / "logit_processor.py"
    spec = importlib.util.spec_from_file_location("rl_logit_processor", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_logit_processor"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"RL logit processor unavailable: {exc}")
    return module


def _load_decode():
    _ensure_src_on_path()
    _stub_transformers_if_missing()
    if "alpamayo_r1.models.base_model" not in sys.modules:
        try:
            import alpamayo_r1.models.base_model  # noqa: F401
        except Exception:
            sys.modules.pop("alpamayo_r1.models.base_model", None)
            import alpamayo_r1  # noqa: F401

            models = ModuleType("alpamayo_r1.models")
            models.__path__ = [str(_SRC / "alpamayo_r1" / "models")]
            sys.modules["alpamayo_r1.models"] = models
            sys.modules["alpamayo_r1.models.base_model"] = SimpleNamespace(
                SPECIAL_TOKENS={
                    "traj_future_start": "<|traj_future_start|>",
                    "traj_future_end": "<|traj_future_end|>",
                }
            )
    if "alpamayo_r1.models.token_utils" not in sys.modules:
        try:
            import alpamayo_r1.models.token_utils  # noqa: F401
        except Exception:
            sys.modules.pop("alpamayo_r1.models.token_utils", None)
            sys.modules["alpamayo_r1.models.token_utils"] = SimpleNamespace(
                extract_traj_tokens=lambda *a, **k: (_ for _ in ()).throw(
                    RuntimeError("extract_traj_tokens stub")
                )
            )
    path = _FINETUNE / "rl" / "utils" / "trajectory_decode.py"
    spec = importlib.util.spec_from_file_location("rl_trajectory_decode", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["rl_trajectory_decode"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"RL trajectory_decode unavailable: {exc}")
    return module


def test_mask_discrete_traj_logits_is_contiguous_range(torch):
    mod = _load_logit_processor()
    proc = mod.MaskDiscreteTrajectoryLogitsProcessor(traj_token_offset=4, traj_vocab_size=3)
    scores = torch.arange(10, dtype=torch.float32).repeat(2, 1)
    out = proc(torch.zeros(2, 1, dtype=torch.long), scores)
    assert torch.isneginf(out[:, 4:7]).all()
    assert not torch.isneginf(out[:, :4]).any()
    assert not torch.isneginf(out[:, 7:]).any()
    assert out is scores


def test_decode_rollout_empty_inner_span_is_zero_tokens(torch):
    dec = _load_decode()

    class _Tok:
        def __call__(self, text):
            raise AssertionError(f"tokenizer should not run on empty inner span, got {text!r}")

        def convert_tokens_to_ids(self, _name):
            return 0

    class _TrajTok:
        vocab_size = 10

        def decode(self, hist_xyz, hist_rot, tokens):
            assert tokens.shape == (1, 8)
            assert int(tokens.sum().item()) == 0
            return hist_xyz, hist_rot, None

    xyz = torch.zeros(1, 4, 3)
    rot = torch.eye(3).expand(1, 4, 3, 3).clone()
    pred_xyz, pred_rot = dec.decode_rollout_trajectory(
        "noise <|traj_future_start|>   ",
        xyz,
        rot,
        tokenizer=_Tok(),
        traj_tokenizer=_TrajTok(),
        model_config=SimpleNamespace(traj_token_start_idx=100, tokens_per_future_traj=8),
    )
    assert pred_xyz.shape == xyz.shape
    assert pred_rot.shape == rot.shape


def test_decode_rollout_trims_to_min_batch(torch):
    dec = _load_decode()
    dec.extract_traj_tokens = lambda *a, **k: torch.arange(2, dtype=torch.long).view(1, 2)

    class _Tok:
        def __call__(self, text):
            return {"input_ids": [7, 101, 102, 8]}

        def convert_tokens_to_ids(self, name):
            return 0

    class _TrajTok:
        vocab_size = 10

        def decode(self, hist_xyz, hist_rot, tokens):
            assert hist_xyz.shape[0] == tokens.shape[0] == 1
            return hist_xyz, hist_rot, None

    xyz = torch.zeros(2, 4, 3)
    rot = torch.eye(3).expand(2, 4, 3, 3).clone()
    pred_xyz, _ = dec.decode_rollout_trajectory(
        "<|traj_future_start|><i1><i2><|traj_future_end|>",
        xyz,
        rot,
        tokenizer=_Tok(),
        traj_tokenizer=_TrajTok(),
        model_config=SimpleNamespace(traj_token_start_idx=100, tokens_per_future_traj=2),
    )
    assert pred_xyz.shape[0] == 1
