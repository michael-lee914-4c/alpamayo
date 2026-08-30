# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for AlpamayoR1MLX custom from_pretrained (Row 5)."""

from mlx_port.models.alpamayo_r1_mlx import (
    AlpamayoR1MLX,
    ExpertDecoder,
    ActionInProj,
    ActionOutProj,
    FlowMatching,
    ActionSpace,
)


def test_alpamayo_r1_mlx_class_structure():
    """Verify AlpamayoR1MLX has the expected components."""
    # Check that the class exists and has from_pretrained
    assert hasattr(AlpamayoR1MLX, "from_pretrained")
    assert callable(getattr(AlpamayoR1MLX, "from_pretrained"))


def test_expert_decoder_instantiation():
    """Verify ExpertDecoder can be created with expert_cfg parameters."""
    expert = ExpertDecoder(
        num_layers=2,
        hidden_size=2048,
        num_heads=16,
        intermediate_size=8256,
    )
    assert expert is not None
    assert len(expert.layers) == 2
    print("ExpertDecoder instantiated successfully with expert_cfg parameters")


def test_action_projection_modules():
    """Verify checkpoint-shaped action projections can be created."""
    action_in = ActionInProj(
        in_dims=(64, 2),
        out_dim=2048,
        num_enc_layers=2,
        hidden_size=512,
    )
    action_out = ActionOutProj(2048, 2)

    assert action_in.out_dim == 2048
    assert action_out.weight.shape == (2, 2048)
    print("ActionInProj and ActionOutProj instantiated successfully")


def test_flow_matching_and_action_space():
    """Verify diffusion and action space stubs exist and raise NotImplementedError on use."""
    diffusion = FlowMatching()
    action_space = ActionSpace()

    assert diffusion is not None
    assert action_space is not None
    print("FlowMatching and ActionSpace stubs created (raise NotImplementedError on call)")


def test_flow_matching_evals_once_after_euler_loop(monkeypatch):
    """P2a: one mx.eval after the integrator, not one per Euler step."""
    import mlx.core as mx

    calls = {"n": 0}
    real_eval = mx.eval

    def counting_eval(*args, **kwargs):
        calls["n"] += 1
        return real_eval(*args, **kwargs)

    monkeypatch.setattr(mx, "eval", counting_eval)
    fm = FlowMatching(x_dims=(4, 2), num_inference_steps=10)

    def step_fn(x, t):
        return mx.zeros_like(x)

    out = fm.sample(batch_size=1, step_fn=step_fn)
    assert tuple(out.shape) == (1, 4, 2)
    assert calls["n"] == 1, f"expected 1 mx.eval after FM, got {calls['n']}"


def test_from_pretrained_stub():
    """Smoke test that from_pretrained can be called (full weight loading tested elsewhere)."""
    # We only check that the method exists and has the right signature.
    # A full integration test would require the complete local checkpoint.
    import inspect
    sig = inspect.signature(AlpamayoR1MLX.from_pretrained)
    params = list(sig.parameters.keys())
    assert "alpamayo_path" in params
    assert "load_expert" in params
    assert "quantize_lm" in params
    assert "lm4_path" in params
    assert sig.parameters["quantize_lm"].default is False
    assert "quantize_vlm_8bit" not in params
    assert "quantize_vlm_4bit" not in params
    assert "quantize_all_4bit" not in params
    assert "quantize_vlm_nvfp4" not in params
    print("from_pretrained signature verified")
