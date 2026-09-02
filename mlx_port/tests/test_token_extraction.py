"""CoC / special-token extraction and traj-logit masking (no weights)."""

import mlx.core as mx
import numpy as np

from mlx_port.models.token_utils_mlx import (
    ExpertLogitsProcessor,
    extract_between_special_tokens,
    extract_text_tokens,
)


def test_extract_between_special_tokens_inner_text():
    texts = [
        "prefix <|cot_start|>Slow yield to the pedestrian.<|cot_end|> suffix",
        "<|meta_action_start|>go straight<|meta_action_end|>",
    ]
    assert extract_between_special_tokens(texts, "cot") == [
        "Slow yield to the pedestrian.",
        texts[1].strip(),
    ]
    assert extract_between_special_tokens(texts, "meta_action") == [
        texts[0].strip(),
        "go straight",
    ]


def test_extract_missing_markers_returns_whole_text():
    """MLX fallback for early rollouts that never emitted the pair."""
    raw = "Stop for the pedestrian in the crosswalk."
    assert extract_between_special_tokens([raw], "cot") == [raw]


def test_extract_missing_end_returns_whole_text():
    raw = "<|cot_start|>partial thought without an end"
    assert extract_between_special_tokens([raw], "cot") == [raw]


def test_extract_uses_first_start_token():
    raw = "<|cot_start|>old draft <|cot_start|>final wording<|cot_end|>"
    assert extract_between_special_tokens([raw], "cot") == [
        "old draft <|cot_start|>final wording"
    ]


def test_extract_empty_span_is_empty_string():
    raw = "<|cot_start|><|cot_end|>"
    assert extract_between_special_tokens([raw], "cot") == [""]


def test_extract_text_tokens_without_batch_decode():
    out = extract_text_tokens(object(), mx.array([[1, 2, 3]]))
    assert set(out) == {"cot", "meta_action", "answer"}
    assert out["cot"] == ["(CoC extraction requires full tokenizer)"]
    assert out["meta_action"] == [""]
    assert out["answer"] == [""]


def test_extract_text_tokens_decodes_cot_and_meta_action():
    class _Tok:
        def batch_decode(self, token_lists, skip_special_tokens=False):
            del skip_special_tokens
            assert token_lists == [[1, 2, 3]]
            return [
                "<|cot_start|>Yield.<|cot_end|>"
                "<|meta_action_start|>stop<|meta_action_end|>"
            ]

    out = extract_text_tokens(_Tok(), mx.array([[1, 2, 3]]))
    assert out["cot"] == ["Yield."]
    assert out["meta_action"] == ["stop"]
    assert out["answer"] == [
        "<|cot_start|>Yield.<|cot_end|><|meta_action_start|>stop<|meta_action_end|>"
    ]


def test_expert_logits_processor_masks_exact_noncontiguous_ids():
    scores = mx.zeros((2, 20))
    proc = ExpertLogitsProcessor(
        traj_token_offset=0,
        traj_vocab_size=20,
        traj_token_ids=[3, 7, 18],
    )
    out = proc(mx.array([[1]]), scores)
    assert float(out[0, 3]) == float("-inf")
    assert float(out[1, 7]) == float("-inf")
    assert float(out[0, 18]) == float("-inf")
    assert float(out[0, 4]) == 0.0
    assert float(out[0, 0]) == 0.0


def test_expert_logits_processor_exact_ids_ignore_out_of_range():
    scores = mx.ones((1, 8))
    proc = ExpertLogitsProcessor(traj_token_ids=[-1, 3, 99])
    out = proc(None, scores)
    assert float(out[0, 3]) == float("-inf")
    assert float(np.asarray(out)[0, 0]) == 1.0
    assert float(np.asarray(out)[0, 7]) == 1.0


def test_expert_logits_processor_exact_ids_override_contiguous_range():
    scores = mx.zeros((1, 10))
    proc = ExpertLogitsProcessor(
        traj_token_offset=0,
        traj_vocab_size=10,
        traj_token_ids=[2],
    )
    out = proc(None, scores)
    assert float(out[0, 2]) == float("-inf")
    assert float(out[0, 1]) == 0.0
    assert float(out[0, 3]) == 0.0
