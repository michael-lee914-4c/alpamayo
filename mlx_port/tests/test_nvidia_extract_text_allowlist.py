"""``extract_text_tokens`` never extracts ``question`` despite SPECIAL_TOKENS.

Weight-free. Complementary to open #17/#33 (CoC ``extract_between``
missing-start/end). The docstring lists question/answers, and
``SPECIAL_TOKENS_KEYS`` defines ``question_start`` / ``question_end``,
but the allowlist is only ``cot`` / ``meta_action`` / ``answer``.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


def _to_special_token(token: str) -> str:
    return "<|" + token + "|>"


def _extract_between_special_tokens(decoded_batch, token):
    start_token = _to_special_token(f"{token}_start")
    end_token = _to_special_token(f"{token}_end")
    out = []
    for s in decoded_batch:
        before_end, sep, _ = s.partition(end_token)
        if not sep:
            out.append("")
            continue
        i = before_end.rfind(start_token)
        if i != -1:
            out.append(before_end[i + len(start_token) :].strip())
        else:
            out.append(before_end.strip())
    return out


def _extract_text_tokens(decoded_batch):
    extract_tokens = ["cot", "meta_action", "answer"]
    return {token: _extract_between_special_tokens(decoded_batch, token) for token in extract_tokens}


def test_source_allowlist_omits_question():
    src = _TOKEN_UTILS.read_text()
    assert 'extract_tokens = ["cot", "meta_action", "answer"]' in src
    assert "question" not in src.split("extract_tokens =")[1].split("\n", 1)[0]
    keys = _BASE.read_text()
    assert '"question_start"' in keys
    assert '"question_end"' in keys
    assert '"answer_start"' in keys


def test_question_span_is_dropped_even_when_present():
    text = (
        "<|question_start|>what is the gap?<|question_end|>"
        "<|cot_start|>slow down<|cot_end|>"
        "<|meta_action_start|>yield<|meta_action_end|>"
        "<|answer_start|>stop<|answer_end|>"
    )
    got = _extract_text_tokens([text])
    assert set(got) == {"cot", "meta_action", "answer"}
    assert "question" not in got
    assert got["cot"] == ["slow down"]
    assert got["meta_action"] == ["yield"]
    assert got["answer"] == ["stop"]


def test_missing_end_is_empty_not_the_rest_of_the_string():
    got = _extract_text_tokens(["<|cot_start|>partial with no end"])
    assert got["cot"] == [""]
    assert got["meta_action"] == [""]
    assert got["answer"] == [""]
