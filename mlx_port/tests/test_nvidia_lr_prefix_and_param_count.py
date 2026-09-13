"""SFT longest-prefix LR groups, param-count depth, comfort score shift.

Weight-free. Complementary to open #24/#25 (reward cfg / ADE 3 m gate)
and the YAML ``lr_multiplier`` check in ``test_sft_nvidia_parity`` —
those never locked longest-prefix matching, frozen-vs-trainable counts,
or the ``comfort - 1`` shift that maps [0, 1] comfort onto [-1, 0].
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_FINETUNE = _REPO / "finetune"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC), str(_FINETUNE)):
        if path not in sys.path:
            sys.path.insert(0, path)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _longest_prefix(name: str, lr_multiplier: dict[str, float]) -> str:
    """Mirrors ReasoningVLA_Trainer.create_optimizer prefix matching."""
    matched_prefix = None
    for prefix in lr_multiplier.keys():
        if name.startswith(prefix):
            if matched_prefix is None or len(prefix) > len(matched_prefix):
                matched_prefix = prefix
    return matched_prefix if matched_prefix else "default"


def test_lr_multiplier_uses_longest_prefix():
    src = (_FINETUNE / "sft" / "trainer.py").read_text()
    assert "use longest match" in src
    assert "name.startswith(prefix)" in src
    assert 'group_key = matched_prefix if matched_prefix else "default"' in src

    mult = {"vlm": 0.5, "vlm.visual": 0.1, "vlm.language_model": 1.0}
    assert _longest_prefix("vlm.visual.blocks.0.weight", mult) == "vlm.visual"
    assert _longest_prefix("vlm.language_model.layers.0", mult) == "vlm.language_model"
    assert _longest_prefix("vlm.other", mult) == "vlm"
    assert _longest_prefix("expert.layers.0", mult) == "default"
    # Unmatched groups stay at 1.0x the base LR.
    assert (1.0 if _longest_prefix("action_in_proj.norm.weight", mult) == "default" else None) == 1.0


def _load_get_param_count():
    _ensure_src_on_path()
    if "rich" not in sys.modules:
        rich = ModuleType("rich")
        console = ModuleType("rich.console")
        pretty = ModuleType("rich.pretty")

        class Console:
            def capture(self):
                class _Cap:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def get(self):
                        return ""

                return _Cap()

        console.Console = Console
        pretty.pprint = lambda *a, **k: None
        rich.console = console
        rich.pretty = pretty
        sys.modules["rich"] = rich
        sys.modules["rich.console"] = console
        sys.modules["rich.pretty"] = pretty
    path = _SRC / "alpamayo_r1" / "common" / "misc.py"
    spec = importlib.util.spec_from_file_location("nv_misc_param_count", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        pytest.skip(f"NVIDIA misc.get_param_count unavailable: {exc}")
    return module.get_param_count


def test_get_param_count_rejects_depth_lt_1_and_skips_frozen(torch):
    get_param_count = _load_get_param_count()
    try:
        get_param_count(torch.nn.Linear(2, 2), depth=0)
    except ValueError as exc:
        assert "depth" in str(exc)
    else:
        raise AssertionError("expected ValueError when depth < 1")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Linear(2, 2, bias=False)
            self.b = torch.nn.Linear(2, 2, bias=False)

    model = Tiny()
    model.b.weight.requires_grad = False
    counts = get_param_count(model, depth=1)
    assert counts["total_params"] == 8
    assert counts["trainable_params"] == 4
    assert counts["a"] == 4
    assert counts["b"] == 4


def test_comfort_score_is_shifted_to_minus_one_one():
    src = (_FINETUNE / "rl" / "rewards" / "aggregated_reward.py").read_text()
    assert "comfort_score = comfort_score - 1.0" in src
    # All metrics 1 → mean 1 → 0. All metrics 0 → mean 0 → -1.
    all_ok = {"a": 1.0, "b": 1.0, "c": 1.0}
    all_fail = {"a": 0.0, "b": 0.0, "c": 0.0}
    mixed = {"a": 1.0, "b": 0.0}
    assert float(sum(all_ok.values()) / len(all_ok)) - 1.0 == pytest.approx(0.0)
    assert float(sum(all_fail.values()) / len(all_fail)) - 1.0 == pytest.approx(-1.0)
    assert float(sum(mixed.values()) / len(mixed)) - 1.0 == pytest.approx(-0.5)
