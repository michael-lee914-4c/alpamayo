"""T4.3 train graph: teacher-forced CE + one CFM draw. No generate, no Euler."""

import subprocess
import sys
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_port.models.alpamayo_r1_mlx import ActionInProj, ActionOutProj, FlowMatching
from mlx_port.models.token_utils_mlx import replace_pad_token
from mlx_port.processor import create_message
from mlx_port.scripts.time_train_step import _event_coc, _image_batch_from_tokenized
from mlx_port.models.expert_mlx import expert_non_causal_train_mask
from mlx_port.train_step import (
    IGNORE_INDEX,
    TRAIN_DOMINANT,
    TRAIN_MS_KEYS,
    TrainStepTimes,
    append_traj_future_start,
    apply_labels_mask,
    assert_stage2_trainables,
    drop_n_traj_group,
    get_role_eos_mask,
    labels_mask_between,
    assert_train_graph,
    cfm_expert_forward,
    expert_train_position_ids,
    freeze_vlm,
    mean_train_times,
    prepare_stage2_trainables,
    print_train_table,
    sft_expert_update,
    sft_stage1_labels_mask,
    sft_train_step,
    shifted_cross_entropy,
    stage1_two_mean_ce,
    traj_future_keep_len,
    unfreeze_expert,
)


class _StubVLM:
    def __init__(self, vocab: int = 16):
        self.calls = 0
        self.vocab = vocab
        self.language_model = SimpleNamespace(_rope_deltas=mx.array([0]))

    def __call__(self, input_ids, cache=None, **kwargs):
        self.calls += 1
        if int(input_ids.shape[-1]) == 1:
            raise RuntimeError("stub VLM saw a one-token decode call")
        b, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        logits = mx.zeros((b, length, self.vocab), dtype=mx.float32)
        for i in range(length - 1):
            nxt = int(input_ids[0, i + 1]) % self.vocab
            logits = logits.at[0, i, nxt].add(20.0)
        return SimpleNamespace(logits=logits)


class _StubExpert:
    def __init__(self):
        self.calls = 0
        self.last_mask = None

    def __call__(self, inputs_embeds, position_ids=None, cache=None, mask=None):
        self.calls += 1
        self.last_mask = mask
        return inputs_embeds


class _CountingSampleFM(FlowMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_calls = 0

    def sample(self, *args, **kwargs):
        self.sample_calls += 1
        raise RuntimeError("FlowMatching.sample must not run in the train graph")


def _toy_model(seq_vocab: int = 16, n_wp: int = 8):
    vlm = _StubVLM(vocab=seq_vocab)
    fm = _CountingSampleFM(x_dims=(n_wp, 2), train_timestep_sampler="uniform")
    return SimpleNamespace(
        vlm=vlm,
        expert=_StubExpert(),
        action_in_proj=ActionInProj(
            in_dims=(n_wp, 2),
            out_dim=32,
            num_enc_layers=1,
            hidden_size=32,
        ),
        action_out_proj=ActionOutProj(32, 2),
        diffusion=fm,
        action_space=None,
        traj_token_ids={"future_start": 7},
        tokenizer=None,
    )


def test_construct_training_data_and_zero_cfm_loss():
    mx.random.seed(0)
    np.random.seed(0)
    fm = FlowMatching(x_dims=(4, 2), train_timestep_sampler="uniform")
    x = mx.ones((2, 4, 2), dtype=mx.float32)
    data = fm.construct_training_data(x)
    assert tuple(data["noisy_x"].shape) == (2, 4, 2)
    t = data["timesteps"]
    expected = t * data["x"] + (1 - t) * data["noise"]
    assert bool(mx.allclose(data["noisy_x"], expected, atol=1e-5))
    target = data["x"] - data["noise"]
    loss = fm.compute_loss_from_pred(data, target)
    assert float(loss.item()) < 1e-6


def test_shifted_ce_perfect_and_ignore():
    labels = mx.array([[1, 2, 3, IGNORE_INDEX]], dtype=mx.int32)
    logits = mx.zeros((1, 4, 8), dtype=mx.float32)
    logits = logits.at[0, 0, 2].add(20.0)
    logits = logits.at[0, 1, 3].add(20.0)
    logits = logits.at[0, 2, 0].add(20.0)
    loss = shifted_cross_entropy(logits, labels)
    assert float(loss.item()) < 1e-4
    masked = apply_labels_mask(mx.array([[1, 2, 3]]), mx.array([[True, False, True]]))
    assert int(masked[0, 1].item()) == IGNORE_INDEX
    assert int(masked[0, 0].item()) == 1


def test_traj_future_keep_len_and_position_ids():
    ids = mx.array([[1, 2, 7, 4, 5]], dtype=mx.int32)
    assert traj_future_keep_len(ids, 7) == 3
    try:
        traj_future_keep_len(mx.array([[1, 2, 3]]), 7)
    except ValueError as exc:
        assert "traj_future_start" in str(exc)
    else:
        raise AssertionError("expected ValueError when marker is missing")
    pos = expert_train_position_ids(4, 1, rope_deltas=5, prefix_len=10)
    assert tuple(pos.shape) == (3, 1, 4)
    assert int(pos[0, 0, 0].item()) == 15
    assert int(pos[0, 0, 3].item()) == 18


def test_stage1_all_false_labels_mask_is_zero_ce():
    model = _toy_model()
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    mask = mx.zeros(ids.shape, dtype=mx.bool_)
    out = sft_train_step(
        model, {"input_ids": ids, "labels_mask": mask}, stage="stage1"
    )
    assert float(out.loss.item()) == 0.0
    assert model.vlm.calls == 1
    assert model.expert.calls == 0


def test_stage1_one_vlm_no_expert_no_euler():
    model = _toy_model()
    ids = mx.array([[1, 2, 3, 4, 7, 8]], dtype=mx.int32)
    out = sft_train_step(model, {"input_ids": ids}, stage="stage1")
    assert model.vlm.calls == 1
    assert model.expert.calls == 0
    assert model.diffusion.sample_calls == 0
    assert out.times.n_vlm_forwards == 1
    assert out.times.n_expert_forwards == 0
    assert out.times.n_euler_steps == 0
    assert out.times.n_decode_tokens == 0
    assert out.cfm_mse is None
    assert out.vlm_ce is not None
    assert_train_graph(out.times)
    mx.eval(out.loss)


def test_stage2_one_vlm_one_expert_no_euler():
    mx.random.seed(1)
    model = _toy_model()
    ids = mx.array([[1, 2, 7, 4]], dtype=mx.int32)
    action = mx.zeros((1, 8, 2), dtype=mx.float32)
    out = sft_train_step(
        model, {"input_ids": ids, "action": action}, stage="stage2"
    )
    assert model.vlm.calls == 1
    assert model.expert.calls == 1
    assert model.diffusion.sample_calls == 0
    assert out.times.n_expert_forwards == 1
    assert out.vlm_ce is None
    assert out.cfm_mse is not None
    assert_train_graph(out.times)


def test_joint_adds_ce_and_cfm():
    mx.random.seed(2)
    model = _toy_model()
    ids = mx.array([[1, 2, 7, 4, 5]], dtype=mx.int32)
    action = mx.ones((1, 8, 2), dtype=mx.float32) * 0.1
    out = sft_train_step(
        model, {"input_ids": ids, "action": action}, stage="joint"
    )
    assert model.vlm.calls == 1
    assert model.expert.calls == 1
    mx.eval(out.loss, out.vlm_ce, out.cfm_mse)
    assert abs(float(out.loss.item()) - float((out.vlm_ce + out.cfm_mse).item())) < 1e-5


def test_unknown_stage_and_missing_action_raise():
    model = _toy_model()
    ids = mx.array([[1, 2, 7]], dtype=mx.int32)
    try:
        sft_train_step(model, {"input_ids": ids}, stage="rollout")
    except ValueError as exc:
        assert "stage" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown stage")
    try:
        sft_train_step(model, {"input_ids": ids}, stage="stage2")
    except ValueError as exc:
        assert "action" in str(exc) or "ego" in str(exc)
    else:
        raise AssertionError("expected ValueError when stage2 has no action")


def test_cfm_forward_does_not_call_sample(monkeypatch):
    mx.random.seed(3)
    model = _toy_model()
    called = {"n": 0}
    real_sample = model.diffusion.sample

    def boom(*args, **kwargs):
        called["n"] += 1
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(model.diffusion, "sample", boom)
    action = mx.zeros((1, 8, 2), dtype=mx.float32)
    loss, pred, times = cfm_expert_forward(model, action)
    mx.eval(loss, pred)
    assert called["n"] == 0
    assert times.n_euler_steps == 0
    assert tuple(pred.shape) == (1, 8, 2)
    mask = model.expert.last_mask
    assert mask is not None
    assert tuple(mask.shape) == (1, 1, 8, 8)
    assert float(mx.max(mx.abs(mask)).item()) == 0.0


def test_assert_train_graph_rejects_decode_and_euler():
    try:
        assert_train_graph(
            type("T", (), {"n_euler_steps": 10, "n_decode_tokens": 0, "n_vlm_forwards": 1})()
        )
    except RuntimeError as exc:
        assert "Euler" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for Euler")
    try:
        assert_train_graph(
            type("T", (), {"n_euler_steps": 0, "n_decode_tokens": 12, "n_vlm_forwards": 1})()
        )
    except RuntimeError as exc:
        assert "decoded" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for decode")


def test_drop_n_traj_group_squeezes_loader_dim():
    xyz = mx.zeros((1, 1, 16, 3), dtype=mx.float32)
    rot = mx.zeros((1, 1, 16, 3, 3), dtype=mx.float32)
    xyz_b, rot_b = drop_n_traj_group(xyz, rot)
    assert tuple(xyz_b.shape) == (1, 16, 3)
    assert tuple(rot_b.shape) == (1, 16, 3, 3)
    xyz2, rot2 = drop_n_traj_group(xyz_b, rot_b)
    assert tuple(xyz2.shape) == (1, 16, 3)
    assert tuple(rot2.shape) == (1, 16, 3, 3)


def test_labels_mask_between_one_span():
    ids = mx.array([[1, 2, 9, 4, 5, 8, 6]], dtype=mx.int32)
    mask = labels_mask_between(ids, 9, 8)
    assert tuple(mask.shape) == (1, 7)
    assert [bool(mask[0, i].item()) for i in range(7)] == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    try:
        labels_mask_between(mx.array([[1, 2, 3]]), 9, 8)
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected ValueError when span markers are missing")
    try:
        labels_mask_between(mx.array([[8, 1, 9]]), 9, 8)
    except ValueError as exc:
        assert "precedes" in str(exc)
    else:
        raise AssertionError("expected ValueError when end precedes start")


def test_create_message_teacher_cot_completes_assistant():
    frames = np.zeros((16, 3, 8, 8), dtype=np.uint8)
    infer = create_message(frames)
    assert infer[2]["content"][0]["text"] == "<|cot_start|>"
    user = infer[1]["content"][-1]["text"]
    assert "chain-of-thought" in user
    train = create_message(frames, teacher_cot="Yield to the pedestrian.")
    text = train[2]["content"][0]["text"]
    assert text == (
        "<|cot_start|>Yield to the pedestrian.<|cot_end|><|traj_future_start|>"
    )
    try:
        create_message(frames, teacher_cot="   ")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError for empty teacher_cot")


def test_create_message_sft_stage1_is_nvidia_traj_future():
    frames = np.zeros((16, 3, 8, 8), dtype=np.uint8)
    msg = create_message(frames, sft_stage="stage1", num_future_traj_tokens=8)
    user_texts = [c["text"] for c in msg[1]["content"] if c.get("type") == "text"]
    assert user_texts[-1] == "output the future trajectory."
    assert "<|traj_history_start|>" in user_texts[0]
    assert all("chain-of-thought" not in t for t in user_texts)
    asst = msg[2]["content"][0]["text"]
    assert asst == (
        "<|traj_future_start|>" + "<|traj_future|>" * 8 + "<|traj_future_end|>"
    )
    try:
        create_message(frames, teacher_cot="x", sft_stage="stage1")
    except ValueError as exc:
        assert "exclusive" in str(exc)
    else:
        raise AssertionError("expected ValueError when teacher_cot and sft_stage both set")
    try:
        create_message(frames, sft_stage="stage2")
    except ValueError as exc:
        assert "stage1" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown sft_stage")
    try:
        create_message(frames, sft_stage="stage1", num_future_traj_tokens=0)
    except ValueError as exc:
        assert "num_future_traj_tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero future pads")
    try:
        create_message(frames, num_history_traj_tokens=0)
    except ValueError as exc:
        assert "num_history_traj_tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero history pads")
    try:
        create_message(np.zeros((3, 8, 8), dtype=np.uint8))
    except ValueError as exc:
        assert "expected 4" in str(exc)
    else:
        raise AssertionError("expected ValueError for rank-3 frames")


class _FakeTok:
    def __init__(self):
        self._ids = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "assistant": 3,
            "system": 4,
            "<|traj_future_start|>": 10,
            "<|traj_future_end|>": 11,
        }

    def convert_tokens_to_ids(self, name):
        return self._ids.get(name)


def test_sft_stage1_labels_mask_is_traj_span_plus_assistant_im_end():
    # system / user / assistant blocks; only assistant eos is labeled besides traj span
    ids = mx.array(
        [[1, 4, 9, 2, 1, 3, 10, 20, 21, 11, 2, 99]],
        dtype=mx.int32,
    )
    tok = _FakeTok()
    mask = sft_stage1_labels_mask(ids, tok)
    got = [bool(mask[0, i].item()) for i in range(ids.shape[1])]
    # traj 10..11 inclusive + last assistant im_end at index 10
    assert got == [
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    eos = get_role_eos_mask(ids, tok)
    assert bool(eos[0, 3].item()) is False
    assert bool(eos[0, 10].item()) is True


def test_stage1_two_mean_ce_splits_traj_and_im_end():
    model = SimpleNamespace(
        future_token_start_idx=20,
        traj_vocab_size=10,
        traj_token_ids={"future_start": 10, "future_end": 11},
    )
    # positions: 0 ignored by shift source; labels[1:]= start, bin, end, im_end
    labels = mx.array([[IGNORE_INDEX, 10, 22, 11, 2]], dtype=mx.int32)
    logits = mx.zeros((1, 5, 40), dtype=mx.float32)
    logits = logits.at[0, 0, 10].add(20.0)
    logits = logits.at[0, 1, 22].add(20.0)
    logits = logits.at[0, 2, 11].add(20.0)
    logits = logits.at[0, 3, 2].add(20.0)
    total, ce_f, ce_o, n_f, n_o = stage1_two_mean_ce(logits, labels, model)
    mx.eval(total, ce_f, ce_o)
    assert n_f == 3
    assert n_o == 1
    assert float(ce_f.item()) < 1e-4
    assert float(ce_o.item()) < 1e-4
    assert abs(float(total.item()) - float((ce_f + ce_o).item())) < 1e-5


def test_replace_pad_token_requires_exact_count():
    ids = mx.array([[1, 5, 5, 2]], dtype=mx.int32)
    out = replace_pad_token(ids, mx.array([[9, 8]]), 5)
    assert [int(out[0, i].item()) for i in range(4)] == [1, 9, 8, 2]
    try:
        replace_pad_token(ids, mx.array([[9]]), 5)
    except ValueError as exc:
        assert "pad tokens" in str(exc)
    else:
        raise AssertionError("expected ValueError when pad count mismatches")


def test_event_coc_requires_matching_t0():
    gt = {
        "events": [
            {"event_start_timestamp": 10, "coc": "  Yield.  "},
            {"event_start_timestamp": 20, "coc": "Other."},
        ]
    }
    assert _event_coc(gt, 10) == "Yield."
    try:
        _event_coc(gt, 99)
    except RuntimeError as exc:
        assert "no CoC" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when t0 has no CoC")


def test_append_traj_future_start_once():
    ids = mx.array([[1, 2, 3]], dtype=mx.int32)
    out = append_traj_future_start(ids, 7)
    assert tuple(out.shape) == (1, 4)
    assert int(out[0, 3].item()) == 7
    again = append_traj_future_start(out, 7)
    assert tuple(again.shape) == (1, 4)
    assert int(again[0, 3].item()) == 7


def test_image_batch_from_tokenized_transposes_hwc():
    hwc = np.zeros((2, 4, 8, 8, 3), dtype=np.float32)
    out = _image_batch_from_tokenized(
        {"pixel_values": hwc, "image_grid_thw": np.ones((2, 3), dtype=np.int32)}
    )
    assert tuple(out["pixel_values"].shape) == (2, 3, 4, 8, 8)
    flats = np.zeros((16, 1536), dtype=np.float32)
    out2 = _image_batch_from_tokenized({"pixel_values": flats})
    assert tuple(out2["pixel_values"].shape) == (16, 1536)


def test_time_train_step_from_clip_in_help():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.time_train_step", "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    for flag in (
        "--from-clip",
        "--teacher-cot",
        "--expert-update",
        "--expert-bf16",
        "--expert-lora",
        "--train-action-proj",
        "--t0-us",
        "--report",
    ):
        if flag not in proc.stdout:
            raise AssertionError(f"{flag} missing from help")


def test_time_train_step_teacher_cot_requires_from_clip():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.time_train_step", "--teacher-cot"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --teacher-cot has no --from-clip")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "from-clip" in err


def test_time_train_step_expert_update_requires_stage2():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--expert-update",
            "--stage",
            "stage1",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --expert-update is not stage2")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "stage2" in err


def test_time_train_step_expert_update_all4_requires_expert_bf16():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--expert-update",
            "--stage",
            "stage2",
            "--quantize-all",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError(
            "expected non-zero exit when --expert-update --quantize-all has no --expert-bf16"
        )
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "expert-bf16" in err


def test_time_train_step_expert_lora_requires_stage2():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--expert-lora",
            "--stage",
            "stage1",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --expert-lora is not stage2")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "stage2" in err


def test_time_train_step_expert_lora_exclusive_with_expert_update():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--expert-lora",
            "--expert-update",
            "--stage",
            "stage2",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --expert-lora and --expert-update")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "exclusive" in err


def test_time_train_step_expert_lora_exclusive_with_lora():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--expert-lora",
            "--lora",
            "--stage",
            "stage2",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --expert-lora and --lora")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "exclusive" in err or "stage1" in err


def test_time_train_step_train_action_proj_requires_expert_lora():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--train-action-proj",
            "--stage",
            "stage2",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError(
            "expected non-zero exit when --train-action-proj has no --expert-lora"
        )
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "expert-lora" in err


def test_time_train_step_rejects_renamed_dense_flags():
    for old, new in (
        ("--dense-expert", "--expert-bf16"),
        ("--expert-dense", "--train-action-proj"),
    ):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlx_port.scripts.time_train_step",
                old,
                "--stage",
                "stage2",
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            raise AssertionError(f"expected non-zero exit for renamed {old}")
        err = (proc.stderr or "") + (proc.stdout or "")
        if new not in err:
            raise AssertionError(f"{old} should mention {new}: {err}")


def test_freeze_vlm_then_unfreeze_expert_leaves_no_vlm_trainables():
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mlx_port.lora import inject_backbone_lora
    from mlx_port.tests.test_lora import TinyHost

    class Host(TinyHost):
        def __init__(self):
            super().__init__(n=2, d=32)
            self.action_in_proj = nn.Linear(32, 32, bias=False)
            self.action_out_proj = nn.Linear(32, 32, bias=False)

    host = Host()
    inject_backbone_lora(host, rank=4, expected_layers=2, vision=False)
    freeze_vlm(host)
    unfreeze_expert(host)
    assert_stage2_trainables(host)
    flat = dict(tree_flatten(host.trainable_parameters()))
    assert not any(k.startswith("vlm.") for k in flat)
    assert any(k.startswith("expert.") for k in flat)
    assert any(k.startswith("action_in_proj.") for k in flat)
    packed = nn.QuantizedLinear.from_linear(host.expert, group_size=32, bits=4)
    host.expert = packed
    host.expert.unfreeze()
    try:
        assert_stage2_trainables(host)
    except RuntimeError as exc:
        assert "QuantizedLinear" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for packed expert")


def test_time_train_step_rejects_exclusive_quant_flags():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.time_train_step",
            "--quantize-lm",
            "--quantize-all",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when both quant flags are set")
    err = (proc.stderr or "") + (proc.stdout or "")
    assert "exclusive" in err


def test_sft_expert_update_qlora_leaves_packed_expert_frozen():
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from mlx_port.lora import (
        assert_only_lora_trainable,
        inject_expert_lora,
        packed_weight_fingerprint,
    )
    from mlx_port.tests.test_lora import TinyStage2Host, _quantize_expert_leaves

    mx.random.seed(0)
    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    host.action_in_proj = ActionInProj(
        in_dims=(4, 2),
        out_dim=32,
        num_enc_layers=1,
        hidden_size=32,
    )
    host.action_out_proj = ActionOutProj(32, 2)
    host.diffusion = FlowMatching(x_dims=(4, 2), train_timestep_sampler="uniform")
    host.traj_token_ids = {"future_start": 7}
    host.expert_non_causal_attention = True
    _quantize_expert_leaves(host)
    inject_expert_lora(host, rank=4, expected_layers=1)
    prepare_stage2_trainables(host)
    assert_only_lora_trainable(host)
    fp0 = packed_weight_fingerprint(host)
    before = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    opt = optim.Adam(learning_rate=1e-3)
    ids = mx.array([[1, 2, 7, 4]], dtype=mx.int32)
    action = mx.zeros((1, 4, 2), dtype=mx.float32)
    loss = sft_expert_update(host, {"input_ids": ids, "action": action}, opt)
    assert np.isfinite(loss.loss)
    assert packed_weight_fingerprint(host) == fp0
    after = {k: np.array(v) for k, v in tree_flatten(host.trainable_parameters())}
    moved = [k for k in before if not np.allclose(before[k], after[k], atol=0.0)]
    if not moved:
        raise AssertionError("expert LoRA A/B did not move after sft_expert_update")
    assert all("lora_" in k and k.startswith("expert.") for k in moved)
    assert_only_lora_trainable(host)
    assert_stage2_trainables(host)


def test_sft_expert_update_train_action_proj_moves_action_proj():
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from mlx_port.lora import (
        inject_expert_lora,
        packed_weight_fingerprint,
    )
    from mlx_port.tests.test_lora import TinyStage2Host, _quantize_expert_leaves

    mx.random.seed(1)
    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    host.action_in_proj = ActionInProj(
        in_dims=(4, 2),
        out_dim=32,
        num_enc_layers=1,
        hidden_size=32,
    )
    host.action_out_proj = ActionOutProj(32, 2)
    host.diffusion = FlowMatching(x_dims=(4, 2), train_timestep_sampler="uniform")
    host.traj_token_ids = {"future_start": 7}
    host.expert_non_causal_attention = True
    _quantize_expert_leaves(host)
    inject_expert_lora(host, rank=4, expected_layers=1)
    prepare_stage2_trainables(host, train_action_proj=True)
    fp0 = packed_weight_fingerprint(host)
    action_in0 = np.array(host.action_in_proj.norm.weight)
    lora0 = {
        k: np.array(v)
        for k, v in tree_flatten(host.trainable_parameters())
        if "lora_" in k
    }
    opt = optim.Adam(learning_rate=1e-2)
    ids = mx.array([[1, 2, 7, 4]], dtype=mx.int32)
    action = mx.ones((1, 4, 2), dtype=mx.float32) * 0.1
    loss = sft_expert_update(
        host, {"input_ids": ids, "action": action}, opt, train_action_proj=True
    )
    assert np.isfinite(loss.loss)
    assert packed_weight_fingerprint(host) == fp0
    if np.allclose(action_in0, np.array(host.action_in_proj.norm.weight)):
        raise AssertionError("action_in_proj did not move with train_action_proj")
    lora1 = {
        k: np.array(v)
        for k, v in tree_flatten(host.trainable_parameters())
        if "lora_" in k
    }
    if all(np.allclose(lora0[k], lora1[k]) for k in lora0):
        raise AssertionError("expert LoRA A/B did not move with train_action_proj")
    assert_stage2_trainables(host)


def test_prepare_stage2_train_action_proj_requires_expert_lora():
    from mlx_port.tests.test_lora import TinyStage2Host

    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    try:
        prepare_stage2_trainables(host, train_action_proj=True)
    except RuntimeError as exc:
        assert "expert LoRA" in str(exc)
    else:
        raise AssertionError(
            "expected RuntimeError when train_action_proj has no expert LoRA"
        )


def test_train_step_times_as_dict_names_dominant():
    times = TrainStepTimes(
        encode_ms=100.0,
        backbone_ms=800.0,
        expert_ms=50.0,
        loss_ms=10.0,
        fwd_bwd_ms=0.0,
        adam_ms=0.0,
        total_ms=960.0,
        n_vlm_forwards=1,
    ).as_dict()
    for key in TRAIN_MS_KEYS:
        assert key in times
    assert times["dominant_stage"] == "backbone"
    assert times["dominant_stage"] in TRAIN_DOMINANT
    assert times["total_ms"] == 960.0
    assert times["dtype"] == "bfloat16"
    print_train_table(times)


def test_mean_train_times_is_arithmetic_mean():
    a = TrainStepTimes(backbone_ms=100.0, fwd_bwd_ms=200.0, total_ms=300.0).as_dict()
    b = TrainStepTimes(backbone_ms=300.0, fwd_bwd_ms=400.0, total_ms=700.0).as_dict()
    mid = mean_train_times([a, b])
    assert mid["backbone_ms"] == 200.0
    assert mid["fwd_bwd_ms"] == 300.0
    assert mid["dominant_stage"] == "fwd_bwd"


def test_mean_train_times_rejects_empty():
    try:
        mean_train_times([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("expected ValueError for an empty trial list")


def test_sft_expert_update_returns_fwd_bwd_and_adam_times():
    import mlx.optimizers as optim

    from mlx_port.lora import inject_expert_lora
    from mlx_port.tests.test_lora import TinyStage2Host, _quantize_expert_leaves

    mx.random.seed(2)
    host = TinyStage2Host(n_vlm=1, n_expert=1, d=32)
    host.action_in_proj = ActionInProj(
        in_dims=(4, 2),
        out_dim=32,
        num_enc_layers=1,
        hidden_size=32,
    )
    host.action_out_proj = ActionOutProj(32, 2)
    host.diffusion = FlowMatching(x_dims=(4, 2), train_timestep_sampler="uniform")
    host.traj_token_ids = {"future_start": 7}
    host.expert_non_causal_attention = True
    _quantize_expert_leaves(host)
    inject_expert_lora(host, rank=4, expected_layers=1)
    prepare_stage2_trainables(host)
    opt = optim.Adam(learning_rate=1e-3)
    ids = mx.array([[1, 2, 7, 4]], dtype=mx.int32)
    action = mx.zeros((1, 4, 2), dtype=mx.float32)
    out = sft_expert_update(host, {"input_ids": ids, "action": action}, opt)
    assert np.isfinite(out.loss)
    assert out.times.fwd_bwd_ms > 0.0
    assert out.times.adam_ms >= 0.0
    assert out.times.total_ms >= out.times.fwd_bwd_ms
    assert out.times.n_vlm_forwards == 1
    assert out.times.n_expert_forwards == 1
    table = out.times.as_dict()
    assert table["dominant_stage"] in TRAIN_DOMINANT


def test_time_train_step_t0_us_requires_from_clip():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.time_train_step", "--t0-us", "5100000"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --t0-us without --from-clip")
    if "requires --from-clip" not in (proc.stderr + proc.stdout):
        raise AssertionError(proc.stderr)
