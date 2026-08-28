"""Tokenizer ID layout must match Alpamayo-R1-10B config.json.

The checkpoint has no tokenizer.json. NVIDIA builds IDs at runtime by adding
all <i0>..<i{traj_vocab_size-1}> first, then SPECIAL_TOKENS. Inserting specials
after only 768 discrete tokens maps <|prompt_start|> onto checkpoint row <i768>,
which is why CoC decode showed placeholder tokens.
"""

import json

from transformers import AutoTokenizer

from mlx_port.processor import LOCAL_QWEN_PROCESSOR_PATH
from mlx_port.vlm_loader import (
    DEFAULT_TRAJ_VOCAB_SIZE,
    SPECIAL_TOKENS,
    _add_alpamayo_tokens,
)

ALPAMAYO_CONFIG = "pre-trained/Alpamayo-R1-10B/config.json"


def _load_alpamayo_cfg() -> dict:
    with open(ALPAMAYO_CONFIG) as f:
        return json.load(f)


def test_default_traj_vocab_matches_checkpoint():
    cfg = _load_alpamayo_cfg()
    assert DEFAULT_TRAJ_VOCAB_SIZE == cfg["traj_vocab_size"] == 4000


def test_add_order_matches_alpamayo_config_ids():
    cfg = _load_alpamayo_cfg()
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_QWEN_PROCESSOR_PATH, trust_remote_code=True
    )
    _add_alpamayo_tokens(tokenizer, traj_vocab_size=cfg["traj_vocab_size"])

    assert len(tokenizer) == cfg["vocab_size"]
    assert tokenizer.traj_token_start_idx == cfg["traj_token_start_idx"]
    assert tokenizer.convert_tokens_to_ids("<i0>") == cfg["traj_token_start_idx"]
    assert tokenizer.convert_tokens_to_ids("<i3999>") == cfg["traj_token_start_idx"] + 3999

    for name, expected_id in cfg["traj_token_ids"].items():
        assert tokenizer.traj_token_ids[name] == expected_id, name

    # First new special after the 4000 discrete tokens (image_pad already exists).
    assert tokenizer.convert_tokens_to_ids("<|prompt_start|>") == 155669
    assert tokenizer.convert_tokens_to_ids("<|cot_start|>") == 155677
    assert tokenizer.convert_tokens_to_ids("<|image_pad|>") == 151655


def test_split_add_order_collides_with_checkpoint_specials():
    """Regression: 768 <iN>, then specials, then the remaining <iN>."""
    cfg = _load_alpamayo_cfg()
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_QWEN_PROCESSOR_PATH, trust_remote_code=True
    )
    _add_alpamayo_tokens(tokenizer, traj_vocab_size=768)
    tokenizer.add_tokens([f"<i{v}>" for v in range(768, cfg["traj_vocab_size"])])

    assert tokenizer.convert_tokens_to_ids("<|prompt_start|>") != 155669
    assert tokenizer.convert_tokens_to_ids("<|prompt_start|>") == cfg["traj_token_start_idx"] + 768
    assert tokenizer.convert_ids_to_tokens(cfg["traj_token_ids"]["future"]) == "<i3988>"
    assert SPECIAL_TOKENS["cot_start"] == "<|cot_start|>"
