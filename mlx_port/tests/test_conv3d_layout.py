"""Conv3D layout: Alpamayo PyTorch weight vs mlx_vlm / AlpamayoPatchEmbed."""

from glob import glob
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from safetensors import safe_open

from mlx_port.models.alpamayo_r1_mlx import AlpamayoPatchEmbed
from mlx_vlm.models.qwen3_vl.vision import PatchEmbed as MlxPatchEmbed

CHECKPOINT = Path("/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B")
WEIGHT_KEY = "vlm.model.visual.patch_embed.proj.weight"
BIAS_KEY = "vlm.model.visual.patch_embed.proj.bias"

OUT, IN_CH, KT, KH, KW = 1152, 3, 2, 16, 16
MLX_WEIGHT_SHAPE = (OUT, KT, KH, KW, IN_CH)  # (O, kD, kH, kW, I)
PT_WEIGHT_SHAPE = (OUT, IN_CH, KT, KH, KW)  # (O, I, kD, kH, kW)


def _load_pt_conv():
    weight = bias = None
    for shard in sorted(glob(str(CHECKPOINT / "model-*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            keys = set(f.keys())
            if WEIGHT_KEY in keys:
                weight = f.get_tensor(WEIGHT_KEY).float()
            if BIAS_KEY in keys:
                bias = f.get_tensor(BIAS_KEY).float()
        if weight is not None and bias is not None:
            break
    assert weight is not None and bias is not None
    return weight, bias


def test_checkpoint_is_pytorch_conv3d_layout():
    weight, bias = _load_pt_conv()
    assert tuple(weight.shape) == PT_WEIGHT_SHAPE
    assert tuple(bias.shape) == (OUT,)
    mlx_ref = nn.Conv3d(IN_CH, OUT, kernel_size=[KT, KH, KW], stride=[KT, KH, KW], bias=True)
    assert tuple(mlx_ref.weight.shape) == MLX_WEIGHT_SHAPE
    assert tuple(weight.shape) != tuple(mlx_ref.weight.shape)


def _torch_gold(pt_w, pt_b, x_nchw):
    return torch.nn.functional.conv3d(
        torch.from_numpy(x_nchw),
        pt_w,
        pt_b,
        stride=(KT, KH, KW),
    ).reshape(x_nchw.shape[0], OUT)


def test_alpamayo_patch_embed_matches_torch():
    """Production pair: raw PT weight + NCHW input (no moveaxis). Matches HF Conv3d."""
    pt_w, pt_b = _load_pt_conv()
    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((4, IN_CH, KT, KH, KW)).astype(np.float32)
    gold = _torch_gold(pt_w, pt_b, x_nchw)

    proj = nn.Conv3d(IN_CH, OUT, kernel_size=[KT, KH, KW], stride=[KT, KH, KW], bias=True)
    proj.weight = mx.array(pt_w.numpy())
    proj.bias = mx.array(pt_b.numpy())
    embed = AlpamayoPatchEmbed(proj, IN_CH, KT, KH)
    got = np.array(embed(mx.array(x_nchw)).astype(mx.float32))
    max_abs = float(np.max(np.abs(got - gold.numpy())))
    print(f"[CONV3D] AlpamayoPatchEmbed + PT weight vs torch max_abs={max_abs:.6f}")
    print(f"[CONV3D] weight on module={tuple(np.array(proj.weight).shape)}")
    assert max_abs < 1e-3


def test_crossed_layouts_do_not_match_torch():
    """Stock moveaxis + unsanitized PT weight (or the reverse) is the real bug pair."""
    pt_w, pt_b = _load_pt_conv()
    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((4, IN_CH, KT, KH, KW)).astype(np.float32)
    x_flat = x_nchw.reshape(4, -1)
    gold = _torch_gold(pt_w, pt_b, x_nchw).numpy()

    # Stock mlx PatchEmbed (moveaxis to NHWC) with raw PT weight.
    stock = MlxPatchEmbed(patch_size=KH, temporal_patch_size=KT, in_channels=IN_CH, hidden_size=OUT)
    stock.proj.weight = mx.array(pt_w.numpy())
    stock.proj.bias = mx.array(pt_b.numpy())
    try:
        crossed = np.array(stock(mx.array(x_flat)).astype(mx.float32))
        crossed_err = float(np.max(np.abs(crossed - gold)))
        print(f"[CONV3D] stock moveaxis + PT weight vs torch max_abs={crossed_err:.4f}")
        assert crossed_err > 1.0
    except ValueError as e:
        print(f"[CONV3D] stock moveaxis + PT weight rejected: {e}")

    # AlpamayoPatchEmbed (NCHW) with sanitized MLX weight.
    mlx_w = np.transpose(pt_w.numpy(), (0, 2, 3, 4, 1))
    proj = nn.Conv3d(IN_CH, OUT, kernel_size=[KT, KH, KW], stride=[KT, KH, KW], bias=True)
    proj.weight = mx.array(mlx_w)
    proj.bias = mx.array(pt_b.numpy())
    embed = AlpamayoPatchEmbed(proj, IN_CH, KT, KH)
    try:
        rev = np.array(embed(mx.array(x_nchw)).astype(mx.float32))
        rev_err = float(np.max(np.abs(rev - gold)))
        print(f"[CONV3D] AlpamayoPatchEmbed + MLX weight vs torch max_abs={rev_err:.4f}")
        assert rev_err > 1.0
    except ValueError as e:
        print(f"[CONV3D] AlpamayoPatchEmbed + MLX weight rejected: {e}")


def test_sanitized_mlx_patch_embed_matches_torch():
    """mlx_vlm sanitize: transpose (0,2,3,4,1) + stock moveaxis to channels-last."""
    pt_w, pt_b = _load_pt_conv()
    rng = np.random.default_rng(0)
    x_nchw = rng.standard_normal((4, IN_CH, KT, KH, KW)).astype(np.float32)
    x_flat = x_nchw.reshape(4, -1)

    gold = torch.nn.functional.conv3d(
        torch.from_numpy(x_nchw),
        pt_w,
        pt_b,
        stride=(KT, KH, KW),
    ).reshape(4, OUT)

    mlx_w = np.transpose(pt_w.numpy(), (0, 2, 3, 4, 1))
    assert mlx_w.shape == MLX_WEIGHT_SHAPE

    embed = MlxPatchEmbed(
        patch_size=KH,
        temporal_patch_size=KT,
        in_channels=IN_CH,
        hidden_size=OUT,
    )
    embed.proj.weight = mx.array(mlx_w)
    embed.proj.bias = mx.array(pt_b.numpy())
    got = np.array(embed(mx.array(x_flat)).astype(mx.float32))
    max_abs = float(np.max(np.abs(got - gold.numpy())))
    print(f"[CONV3D] sanitized mlx PatchEmbed vs torch max_abs={max_abs:.6f}")
    assert max_abs < 1e-3


def test_inference_thwc_unflatten_scrambles_hf_pack():
    """Processor flats are C*T*H*W. inference.py used to reshape as T*H*W*C."""
    pt_w, pt_b = _load_pt_conv()
    rng = np.random.default_rng(1)
    x_nchw = rng.standard_normal((4, IN_CH, KT, KH, KW)).astype(np.float32)
    x_flat = x_nchw.reshape(4, -1)  # HF / processor order
    gold = _torch_gold(pt_w, pt_b, x_nchw).numpy()

    proj = nn.Conv3d(IN_CH, OUT, kernel_size=[KT, KH, KW], stride=[KT, KH, KW], bias=True)
    proj.weight = mx.array(pt_w.numpy())
    proj.bias = mx.array(pt_b.numpy())
    embed = AlpamayoPatchEmbed(proj, IN_CH, KT, KH)

    ok = np.array(embed(mx.array(x_flat)).astype(mx.float32))
    ok_err = float(np.max(np.abs(ok - gold)))
    print(f"[CONV3D] 2D HF pack through AlpamayoPatchEmbed max_abs={ok_err:.6f}")
    assert ok_err < 1e-3

    # Current inference.py: reshape (N, T, H, W, C) then transpose to NCHW.
    scrambled = x_flat.reshape(4, KT, KH, KW, IN_CH).transpose(0, 4, 1, 2, 3)
    bad = np.array(embed(mx.array(scrambled)).astype(mx.float32))
    bad_err = float(np.max(np.abs(bad - gold)))
    print(f"[CONV3D] inference T,H,W,C unflatten vs torch max_abs={bad_err:.4f}")
    assert bad_err > 1.0
