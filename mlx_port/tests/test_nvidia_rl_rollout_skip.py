"""Training rollout drops a prompt unless it has ≥2 non-empty completions.

Weight-free. Complementary to open #33 (empty ``gen_ids`` still decode)
and #32 (policy last-start). Empty string is replaced by ``eos_token``
for the count, so ``n=2`` with one real + one empty is skipped. Kept
rows still store the *original* ``response[i]``, not the eos-substituted
texts.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ROLLOUT = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "rollout.py"


def _filter_training_completions(completion, eos_token="</s>"):
    total_generation_count = len(completion)
    empty_generation_count = 0
    output_texts = []
    for output_text in completion:
        if output_text == "":
            output_text = eos_token
            empty_generation_count += 1
        output_texts.append(output_text)
    skip_output = (total_generation_count - empty_generation_count) <= 1
    return skip_output, output_texts


def test_source_skips_one_or_zero_nonempty_and_keeps_original_response():
    src = _ROLLOUT.read_text()
    assert "if output_text == \"\":" in src
    assert "output_text = self.tokenizer.eos_token" in src
    assert "empty_generation_count += 1" in src
    assert "skip_output = (total_generation_count - empty_generation_count) <= 1" in src
    assert "completions=response[i]" in src
    assert "completions=[]" in src
    assert "if is_validation:" in src


def test_one_real_and_one_empty_is_skipped():
    skip, texts = _filter_training_completions(["turn left", ""])
    assert skip is True
    assert texts == ["turn left", "</s>"]


def test_two_real_completions_are_kept():
    skip, texts = _filter_training_completions(["a", "b"])
    assert skip is False
    assert texts == ["a", "b"]


def test_two_empty_and_three_with_one_empty():
    skip_empty, texts_empty = _filter_training_completions(["", ""])
    assert skip_empty is True
    assert texts_empty == ["</s>", "</s>"]
    skip_three, texts_three = _filter_training_completions(["a", "b", ""])
    assert skip_three is False
    assert texts_three == ["a", "b", "</s>"]


def test_whitespace_only_is_not_treated_as_empty():
    skip, texts = _filter_training_completions([" ", "\n"])
    assert skip is False
    assert texts == [" ", "\n"]
