"""``construct_user_prompt`` silently drops unknown component names.

Weight-free. Complementary to open #24 (train-vs-gen prompt /
ask-for-last). Only ``cot`` / ``meta_action`` / ``traj_future`` are
emitted. ``image``, ``prompt``, ``traj_history``, and typos become an
empty join, so the user prompt is just ``\".\"``. Generation mode reads
``components_prompt``; train mode reads ``components_order``.
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


def _construct_user_prompt(components_order, components_prompt, generation_mode):
    components = components_prompt if generation_mode else components_order
    prompt_components = []
    for component in components:
        if component in _TEMPLATE:
            prompt_components.append(_TEMPLATE[component])
    prompt = ", then ".join(prompt_components) + "."
    return [{"type": "text", "text": prompt}]


def test_source_uses_template_membership_and_mode_switch():
    src = _CONV.read_text()
    assert "components = components_prompt if generation_mode else components_order" in src
    assert "if component in template.keys():" in src
    assert 'prompt = ", then ".join(prompt_components) + "."' in src
    assert '"cot": "output the chain-of-thought reasoning of the driving process"' in src


def test_unknown_and_user_side_components_yield_dot_only():
    order = ["image", "prompt", "traj_history", "not_a_component"]
    got = _construct_user_prompt(order, ["traj_future"], generation_mode=False)
    assert got == [{"type": "text", "text": "."}]


def test_generation_mode_reads_prompt_list_not_order():
    order = ["cot", "traj_future"]
    prompt = ["traj_future"]
    train = _construct_user_prompt(order, prompt, generation_mode=False)
    gen = _construct_user_prompt(order, prompt, generation_mode=True)
    assert "chain-of-thought" in train[0]["text"]
    assert "future trajectory" in train[0]["text"]
    assert "chain-of-thought" not in gen[0]["text"]
    assert gen[0]["text"] == "output the future trajectory."


def test_known_components_join_with_then():
    got = _construct_user_prompt(
        ["cot", "meta_action", "traj_future"],
        [],
        generation_mode=False,
    )
    assert got[0]["text"] == (
        "output the chain-of-thought reasoning of the driving process, "
        "then output meta actions, then output the future trajectory."
    )
