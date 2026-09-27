"""``construct_user_prompt`` with no known components is exactly ``\".\"``.

Weight-free. Complementary to open #34 (unknown / user-side names →
``\".\"``) and #41 (train uses ``components_order``, gen uses
``components_prompt``). An empty or all-unknown list still joins to
``\".\"`` — not ``\"\"`` and not a trailing ``\", then .\"``. A port that
skips the join on empty input would change the user turn and the
image-token alignment.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CONV = _REPO / "src" / "alpamayo_r1" / "chat_template" / "conversation.py"

_TEMPLATE = {
    "cot": "output the chain-of-thought reasoning of the driving process",
    "meta_action": "output meta actions",
    "traj_future": "output the future trajectory",
}


def _user_prompt(components: list[str]) -> str:
    prompt_components = [ _TEMPLATE[c] for c in components if c in _TEMPLATE ]
    return ", then ".join(prompt_components) + "."


def test_source_always_appends_period_after_join():
    src = _CONV.read_text()
    body = src.split("def construct_user_prompt")[1].split("def construct_image")[0]
    assert 'prompt = ", then ".join(prompt_components) + "."' in body
    assert "if prompt_components" not in body


def test_empty_and_unknown_collapse_to_dot():
    assert _user_prompt([]) == "."
    assert _user_prompt(["image", "traj_history", "prompt"]) == "."
    assert _user_prompt(["traj_future"]) == "output the future trajectory."
    assert _user_prompt(["cot", "traj_future"]) == (
        "output the chain-of-thought reasoning of the driving process, "
        "then output the future trajectory."
    )
