"""Grok CoC judge helpers — no network. Complementary to Jaccard tests in #17–#21."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_coc_grok.py"


def _load_eval():
    spec = importlib.util.spec_from_file_location("eval_coc_grok", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_coc_grok"] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_api_key_prefers_cli_then_xai_then_grok(monkeypatch):
    ev = _load_eval()
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    try:
        ev._resolve_api_key(None)
    except SystemExit as exc:
        assert "XAI_API_KEY" in str(exc)
    else:
        raise AssertionError("expected SystemExit when no Grok token is set")

    monkeypatch.setenv("GROK_API_KEY", "  grok-only  ")
    monkeypatch.setenv("XAI_API_KEY", "  xai-token  ")
    assert ev._resolve_api_key(None) == "xai-token"
    assert ev._resolve_api_key("  cli-token  ") == "cli-token"
    monkeypatch.setenv("XAI_API_KEY", "   ")
    assert ev._resolve_api_key(None) == "grok-only"


def test_build_user_prompt_includes_gt_primary_and_optional_top5():
    ev = _load_eval()
    rec = {
        "clip_id": "abc",
        "event_cluster": "pedestrian",
        "gt_coc_texts": ["Yield to the pedestrian."],
        "pred_coc": "Stop for the person in the crosswalk.",
    }
    prompt = ev._build_user_prompt(rec)
    assert "clip_id: abc" in prompt
    assert "Yield to the pedestrian." in prompt
    assert "Stop for the person in the crosswalk." in prompt
    assert "Other candidate CoCs" not in prompt

    rec["top5_coc"] = [
        {"rank": 2, "first_p": 0.4, "first_token": "Slow", "pred_coc": "Slow for the walker."}
    ]
    with_top5 = ev._build_user_prompt(rec)
    assert "Other candidate CoCs" in with_top5
    assert "Slow for the walker." in with_top5
    assert "#2" in with_top5


def test_html_report_counts_verdicts_and_escapes_pred():
    ev = _load_eval()
    html = ev._html_report(
        [
            {
                "clip_id": "aaaaaaaa-bbbb",
                "verdict": "match",
                "label_alignment": 5,
                "same_hazard": True,
                "same_action": True,
                "gt_coc_texts": ["Yield"],
                "pred_coc": "<script>alert(1)</script>",
                "rationale": "same hazard",
            },
            {
                "clip_id": "cccccccc-dddd",
                "verdict": "mismatch",
                "label_alignment": 1,
                "same_hazard": False,
                "same_action": False,
                "gt_coc_texts": ["Stop"],
                "pred_coc": "Turn left",
                "rationale": "different",
            },
        ],
        "grok-4.6",
        "2026-09-08 10:00 UTC",
    )
    assert "match=1" in html and "mismatch=1" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
