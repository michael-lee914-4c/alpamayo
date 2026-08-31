"""Side-by-side NVIDIA Stage 1/2 SFT vs this MLX port.

Imports NVIDIA helpers under ``PYTHONPATH=src``. Documents the two
intentional divergences (hist embedding offset, Mac QLoRA optimizer).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
import yaml

try:
    from alpamayo_r1.chat_template.conversation import (
        construct_system_prompt,
        construct_traj_future,
        construct_traj_history,
        construct_user_prompt,
    )
    from alpamayo_r1.models.base_model import IGNORE_INDEX as NV_IGNORE
    from alpamayo_r1.models.base_model import SPECIAL_TOKENS
    from alpamayo_r1.utils.get_label_mask import get_label_mask
    from alpamayo_r1.utils.get_label_mask import get_role_eos_mask as nv_role_eos

    _NVIDIA_IMPORT_ERROR = None
except Exception as exc:  # transformers/hydra differ between Mac .venv and mlx-ci
    _NVIDIA_IMPORT_ERROR = exc
    NV_IGNORE = -100
    SPECIAL_TOKENS = {
        "traj_future_start": "<|traj_future_start|>",
        "traj_future_end": "<|traj_future_end|>",
        "traj_future": "<|traj_future|>",
    }


def _require_nvidia():
    if _NVIDIA_IMPORT_ERROR is not None:
        pytest.skip(f"NVIDIA SFT imports unavailable: {_NVIDIA_IMPORT_ERROR}")

from mlx_port.models.alpamayo_r1_mlx import FlowMatching
from mlx_port.models.expert_mlx import expert_non_causal_train_mask
from mlx_port.models.trajectory_tokenizer_mlx import DiscreteTrajectoryTokenizerMLX
from mlx_port.processor import (
    DEFAULT_FUTURE_TRAJ_TOKENS,
    DEFAULT_HISTORY_TRAJ_TOKENS,
    SFT_STAGE1_USER_PROMPT,
    create_message,
)
from mlx_port.train_step import (
    IGNORE_INDEX,
    expert_train_position_ids,
    sft_stage1_labels_mask,
    stage1_two_mean_ce,
    traj_future_keep_len,
)

_REPO = Path(__file__).resolve().parents[2]
_ALPAMAYO_CFG = _REPO / "pre-trained" / "Alpamayo-R1-10B" / "config.json"
_SFT_STAGE1 = _REPO / "finetune" / "sft" / "configs" / "sft_stage1.yaml"
_SFT_STAGE2 = _REPO / "finetune" / "sft" / "configs" / "sft_stage2.yaml"
_SFT_BASE = _REPO / "finetune" / "sft" / "configs" / "sft_base.yaml"
_VLA_PROC = _REPO / "finetune" / "sft" / "configs" / "vla_processor.yaml"
_ALPAMAYO_R1_MLX = _REPO / "mlx_port" / "models" / "alpamayo_r1_mlx.py"


class _SharedTok:
    """IDs shared by NVIDIA ``get_label_mask`` and MLX ``sft_stage1_labels_mask``."""

    def __init__(self):
        self._ids = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "assistant": 3,
            "system": 4,
            SPECIAL_TOKENS["traj_future_start"]: 10,
            SPECIAL_TOKENS["traj_future_end"]: 11,
            SPECIAL_TOKENS["traj_future"]: 12,
            "<|traj_future_start|>": 10,
            "<|traj_future_end|>": 11,
        }

    def convert_tokens_to_ids(self, name):
        if name not in self._ids:
            raise KeyError(name)
        return self._ids[name]


class _StubAction:
    def __init__(self, n_wp=4):
        self.n_wp = n_wp

    def get_action_space_dims(self):
        return (self.n_wp, 2)

    def traj_to_action(self, hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=None):
        del hist_xyz, hist_rot, fut_rot, t0_states
        return np.asarray(fut_xyz)[..., :2].astype(np.float64)


def _nv_shifted_ce(logits: torch.Tensor, labels: torch.Tensor, labels_mask: torch.Tensor):
    """NVIDIA ``TrainableReasoningVLA._compute_next_token_loss`` (mean, no token_mask)."""
    if labels_mask[:, 1:].sum() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    shift_labels = labels[..., 1:]
    shift_logits = logits[..., :-1, :].float()
    shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
    shift_logits = shift_logits[labels_mask[:, 1:]].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits, shift_labels, ignore_index=NV_IGNORE, reduction="mean"
    )


def _nv_two_mean(logits: torch.Tensor, labels: torch.Tensor, *, start: int, vocab: int, fut_start: int, fut_end: int):
    traj_mask = (
        ((labels >= start) & (labels < start + vocab))
        | (labels == fut_start)
        | (labels == fut_end)
    )
    ce_future = _nv_shifted_ce(logits, labels, traj_mask)
    labels = labels.clone()
    labels[traj_mask] = NV_IGNORE
    ce_others = _nv_shifted_ce(logits, labels, labels != NV_IGNORE)
    return ce_future + ce_others, ce_future, ce_others


def test_nvidia_yaml_label_components_and_prompts():
    _require_nvidia()
    vla = yaml.safe_load(_VLA_PROC.read_text())
    assert vla["components_order"] == ["image", "traj_history", "prompt", "traj_future"]
    assert vla["components_prompt"] == ["traj_future"]
    assert vla["label_components"] == ["traj_future"]

    nv_prompt = construct_user_prompt(
        components_order=vla["components_order"],
        components_prompt=vla["components_prompt"],
        generation_mode=False,
    )[0]["text"]
    assert nv_prompt == SFT_STAGE1_USER_PROMPT
    assert nv_prompt == "output the future trajectory."

    nv_sys = construct_system_prompt()[0]["text"]
    frames = np.zeros((2, 3, 8, 8), dtype=np.uint8)
    mlx_msg = create_message(frames, sft_stage="stage1")
    assert mlx_msg[0]["content"][0]["text"] == nv_sys

    nv_hist = construct_traj_history(DEFAULT_HISTORY_TRAJ_TOKENS)[0]["text"]
    nv_fut = construct_traj_future(DEFAULT_FUTURE_TRAJ_TOKENS)[0]["text"]
    user_texts = [c["text"] for c in mlx_msg[1]["content"] if c.get("type") == "text"]
    assert user_texts[0] == nv_hist
    assert user_texts[1] == nv_prompt
    assert mlx_msg[2]["content"][0]["text"] == nv_fut
    assert nv_fut.count("<|traj_future|>") == 128
    assert nv_hist.count("<|traj_history|>") == 48


def test_stage1_user_content_is_nvidia_three_parts():
    """NVIDIA ``build_conversation``: images, hist text, prompt text as separate items."""
    frames = np.zeros((4, 3, 8, 8), dtype=np.uint8)
    msg = create_message(frames, sft_stage="stage1")
    kinds = [c["type"] for c in msg[1]["content"]]
    assert kinds == ["image", "image", "image", "image", "text", "text"]
    infer = create_message(frames)
    infer_kinds = [c["type"] for c in infer[1]["content"]]
    assert infer_kinds.count("text") == 1
    assert "chain-of-thought" in infer[1]["content"][-1]["text"]


def test_nvidia_label_mask_matches_mlx():
    _require_nvidia()
    tok = _SharedTok()
    # system / user / assistant: only assistant eos + traj span
    ids = [[1, 4, 9, 2, 1, 3, 10, 20, 21, 11, 2, 99]]
    nv_ids = torch.tensor(ids, dtype=torch.long)
    nv = get_label_mask(nv_ids, tok, ["traj_future"])
    nv = nv | nv_role_eos(nv_ids, tok)
    mlx = sft_stage1_labels_mask(mx.array(ids, dtype=mx.int32), tok)
    assert np.array_equal(nv.numpy(), np.asarray(mlx))
    assert int(nv.sum().item()) == 5


def test_two_mean_ce_matches_nvidia_torch():
    start, vocab, fut_s, fut_e = 20, 10, 10, 11
    labels_np = np.array([[NV_IGNORE, 10, 22, 11, 2]], dtype=np.int64)
    logits_np = np.zeros((1, 5, 40), dtype=np.float32)
    logits_np[0, 0, 10] = 4.0
    logits_np[0, 1, 22] = 1.5
    logits_np[0, 2, 11] = 0.2
    logits_np[0, 3, 2] = -1.0
    nv_total, nv_f, nv_o = _nv_two_mean(
        torch.from_numpy(logits_np),
        torch.from_numpy(labels_np),
        start=start,
        vocab=vocab,
        fut_start=fut_s,
        fut_end=fut_e,
    )
    model = SimpleNamespace(
        future_token_start_idx=start,
        traj_vocab_size=vocab,
        traj_token_ids={"future_start": fut_s, "future_end": fut_e},
    )
    mlx_total, mlx_f, mlx_o, n_f, n_o = stage1_two_mean_ce(
        mx.array(logits_np), mx.array(labels_np, dtype=mx.int32), model
    )
    mx.eval(mlx_total, mlx_f, mlx_o)
    assert n_f == 3
    assert n_o == 1
    assert abs(float(mlx_f.item()) - float(nv_f.item())) < 1e-5
    assert abs(float(mlx_o.item()) - float(nv_o.item())) < 1e-5
    assert abs(float(mlx_total.item()) - float(nv_total.item())) < 1e-5
    assert IGNORE_INDEX == NV_IGNORE == -100


def test_discrete_encode_matches_nvidia_round_clamp():
    _require_nvidia()
    from alpamayo_r1.action_space.discrete_action_space import DiscreteTrajectoryTokenizer

    class _TorchStub:
        def get_action_space_dims(self):
            return (4, 2)

        def traj_to_action(self, hist_xyz, hist_rot, fut_xyz, fut_rot, t0_states=None):
            del hist_xyz, hist_rot, fut_rot, t0_states
            return fut_xyz[..., :2]

    import hydra.utils as hyu

    orig = hyu.instantiate
    hyu.instantiate = lambda cfg, **kw: _TorchStub()
    try:
        nv = DiscreteTrajectoryTokenizer(
            action_space_cfg={},
            dims_min=[-10.0, -10.0],
            dims_max=[10.0, 10.0],
            num_bins=3000,
        )
    finally:
        hyu.instantiate = orig

    mlx = DiscreteTrajectoryTokenizerMLX(
        action_space=_StubAction(n_wp=4),
        dims_min=[-10.0, -10.0],
        dims_max=[10.0, 10.0],
        num_bins=3000,
    )
    hist_xyz = torch.zeros(2, 2, 3)
    hist_rot = torch.eye(3).expand(2, 2, 3, 3).clone()
    # Include 0 (→ 1500), a half-bin (1499.5 → 1500), and a clamp edge.
    fut_xyz = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [10.0, -10.0, 0.0], [-10.0, 10.0, 0.0], [0.0, 1.0 / 2999.0 * 20.0 - 10.0, 0.0]],
            [[3.0, -3.0, 0.0], [0.0, 0.0, 0.0], [9.9, -9.9, 0.0], [0.5, 0.5, 0.0]],
        ],
        dtype=torch.float32,
    )
    fut_rot = torch.eye(3).expand(2, 4, 3, 3).clone()
    nv_ids = nv.encode(hist_xyz, hist_rot, fut_xyz, fut_rot).cpu().numpy()
    mlx_ids = np.array(mlx.encode(hist_xyz.numpy(), hist_rot.numpy(), fut_xyz.numpy(), fut_rot.numpy()))
    assert nv_ids.shape == (2, 8)
    assert np.array_equal(nv_ids, mlx_ids)
    assert int(nv_ids[0, 0]) == 1500


def test_cfm_algebra_matches_nvidia():
    x = np.array([[[1.0, -2.0], [0.5, 0.25]]], dtype=np.float32)
    noise = np.array([[[0.1, 0.2], [-0.3, 0.4]]], dtype=np.float32)
    t = np.array([[[[0.3]]]], dtype=np.float32)
    nv_noisy = t * x + (1.0 - t) * noise
    nv_target = x - noise
    nv_loss = float(np.mean((nv_target - nv_target) ** 2))
    assert nv_loss == 0.0
    fm = FlowMatching(x_dims=(2, 2), train_timestep_sampler="uniform")
    data = {
        "x": mx.array(x),
        "noise": mx.array(noise),
        "noisy_x": mx.array(nv_noisy.astype(np.float32)),
        "timesteps": mx.array(t.astype(np.float32)),
    }
    loss = fm.compute_loss_from_pred(data, mx.array(nv_target))
    assert float(loss.item()) < 1e-6
    rebuilt = t * x + (1.0 - t) * noise
    assert np.allclose(rebuilt, nv_noisy)


def test_crop_and_expert_position_ids_match_nvidia():
    # NVIDIA: last_traj_future_start_idx = nonzero(...)[-1, 1] + 1
    ids = mx.array([[1, 2, 7, 4, 7, 9]], dtype=mx.int32)
    assert traj_future_keep_len(ids, 7) == 5
    # NVIDIA: arange(T) + rope_deltas + kv_len as (3, B, T)
    pos = expert_train_position_ids(4, 2, rope_deltas=3, prefix_len=10)
    assert tuple(pos.shape) == (3, 2, 4)
    assert int(pos[0, 0, 0].item()) == 13
    assert int(pos[2, 1, 3].item()) == 16


def test_expert_train_mask_is_non_causal_zeros():
    mask = expert_non_causal_train_mask(2, 64, 100)
    assert tuple(mask.shape) == (2, 1, 64, 164)
    assert float(mx.max(mx.abs(mask)).item()) == 0.0
    try:
        expert_non_causal_train_mask(0, 64, 100)
    except ValueError as exc:
        assert "batch" in str(exc)
    else:
        raise AssertionError("expected ValueError for batch=0")


def test_config_json_and_sft_yamls():
    cfg = json.loads(_ALPAMAYO_CFG.read_text())
    assert cfg["expert_non_causal_attention"] is True
    assert cfg["tokens_per_future_traj"] == 128
    assert cfg["tokens_per_history_traj"] == 48
    assert cfg["traj_vocab_size"] == 4000
    assert cfg["traj_token_start_idx"] == 151669
    assert cfg["traj_tokenizer_cfg"]["num_bins"] == 3000
    assert cfg["traj_tokenizer_cfg"]["dims_min"] == [-10, -10]
    assert DEFAULT_FUTURE_TRAJ_TOKENS == 128
    assert DEFAULT_HISTORY_TRAJ_TOKENS == 48

    stage1 = yaml.safe_load(_SFT_STAGE1.read_text())
    assert float(stage1["trainer"]["learning_rate"]) == 1e-5
    assert float(stage1["trainer"]["lr_multiplier"]["vlm.model.visual"]) == 0.1
    stage2 = yaml.safe_load(_SFT_STAGE2.read_text())
    assert "ar1_expert" in str(stage2["defaults"])
    base = yaml.safe_load(_SFT_BASE.read_text())
    assert base["data"]["train_dataset"]["chunk_ids"] == "0-99"
    assert base["data"]["train_dataset"]["use_default_keyframe"] is True


def test_hist_offset_stays_at_i0_on_purpose():
    """NVIDIA ``ReasoningVLA`` does ``hist_token_start_idx += traj_tokenizer.vocab_size``.

    This port keeps hist at ``<i0>`` so Stage-2 KV matches signed infer. Do not
    add the +3000 offset unless asked.
    """
    src = _ALPAMAYO_R1_MLX.read_text()
    live = [
        ln
        for ln in src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    assert any("model.hist_token_start_idx = tokenizer.traj_token_start_idx" in ln for ln in live)
    assert any("model.future_token_start_idx = tokenizer.traj_token_start_idx" in ln for ln in live)
    assert not any("hist_token_start_idx +=" in ln for ln in live)
    assert "hist_token_start_idx += self.traj_tokenizer.vocab_size" in (
        (_REPO / "src" / "alpamayo_r1" / "models" / "base_model.py").read_text()
    )


def test_image_flatten_is_camera_major_like_nvidia():
    """NVIDIA ``construct_image`` is cameras then frames. MLX ``flatten(0, 1)`` matches."""
    cams, frames, c, h, w = 4, 4, 3, 2, 2
    stacked = np.arange(cams * frames * c * h * w, dtype=np.uint8).reshape(cams, frames, c, h, w)
    flat = stacked.reshape(cams * frames, c, h, w)
    nvidia_order = []
    for cam in stacked:
        for frame in cam:
            nvidia_order.append(frame)
    nvidia_order = np.stack(nvidia_order, axis=0)
    assert np.array_equal(flat, nvidia_order)


def test_qwen_chat_template_joins_split_user_text_without_separator():
    """Two user text items vs one concatenated string — Qwen template must match."""
    qwen = _REPO / "pre-trained" / "Qwen3-VL-8B-Instruct"
    if os.environ.get("ALPAMAYO_CI_NO_WEIGHTS") or not qwen.exists():
        pytest.skip(f"Qwen processor not on this machine ({qwen})")
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(str(qwen), trust_remote_code=True)
    hist = (
        "<|traj_history_start|>"
        + "<|traj_history|>" * 4
        + "<|traj_history_end|>"
    )
    prompt = SFT_STAGE1_USER_PROMPT
    split = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": hist},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "<|traj_future_start|>"}]},
    ]
    joined = [
        split[0],
        {"role": "user", "content": [{"type": "text", "text": hist + prompt}]},
        split[2],
    ]
    a = proc.apply_chat_template(split, tokenize=False, add_generation_prompt=False)
    b = proc.apply_chat_template(joined, tokenize=False, add_generation_prompt=False)
    assert a == b
