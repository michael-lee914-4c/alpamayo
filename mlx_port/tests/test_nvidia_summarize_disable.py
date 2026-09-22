"""``summarize_metric`` drops ``_std`` when N=1 or ``disable_summary``.

Weight-free. Complementary to open #38 (rank≠2 raise) and #24 (mean /
mean_square / stdev). A one-sample group or an eval that asked for raw
per-group scores must not grow ``min_ade_std`` keys; those leak into
``metric/`` averages as if they were extra ADE values.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_UTILS = _SRC / "alpamayo_r1" / "metrics" / "metric_utils.py"


def _ensure_src_on_path() -> None:
    for path in (str(_REPO), str(_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_summarize():
    _ensure_src_on_path()
    pytest.importorskip("torch")
    try:
        from alpamayo_r1.metrics.metric_utils import summarize_metric

        return summarize_metric
    except Exception:
        spec = importlib.util.spec_from_file_location("nv_metric_utils_disable", _UTILS)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {_UTILS}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.summarize_metric


def test_source_std_requires_n_gt_1_and_not_disabled():
    src = _UTILS.read_text()
    assert "if val.shape[1] > 1 and not disable_summary:" in src
    assert 'result[metric_name + "_std"] = std' in src
    assert "if val.ndim != 2:" in src


def test_n1_omits_std_even_when_summary_enabled():
    torch = pytest.importorskip("torch")
    summarize_metric = _load_summarize()
    val = torch.tensor([[1.5]], dtype=torch.float32)
    out = summarize_metric({"min_ade": val}, disable_summary=False)
    assert list(out) == ["min_ade"]
    torch.testing.assert_close(out["min_ade"], torch.tensor([1.5]))


def test_disable_summary_keeps_mean_and_drops_std():
    torch = pytest.importorskip("torch")
    summarize_metric = _load_summarize()
    val = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32)
    disabled = summarize_metric({"min_ade": val}, disable_summary=True)
    assert list(disabled) == ["min_ade"]
    torch.testing.assert_close(disabled["min_ade"], torch.tensor([2.0, 3.0]))

    enabled = summarize_metric({"min_ade": val}, disable_summary=False)
    assert "min_ade_std" in enabled
    # var = E[x^2] - mean^2 = 5 - 4 = 1 → std = 1 for both rows.
    torch.testing.assert_close(enabled["min_ade_std"], torch.tensor([1.0, 1.0]))
