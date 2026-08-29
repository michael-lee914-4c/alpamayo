# MLX port map (Stage 1c Phase 0)

Filled from code on 2026-08-29, not from memory. One inference window =
encode → prefill → CoC decode → 10-step flow-matching → `action_to_traj`.

Do not start Phase 3 quant until Gate G1 in
`alpamayo1-mlx-speedup-plan.html` is green.

## Four stages → files

| Stage | What runs | Primary files |
|---|---|---|
| Encode | Qwen3-VL vision tower over 16 RGB frames | `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoModel.get_input_embeddings`), `mlx_port/models/alpamayo_r1_mlx.py` (`AlpamayoPatchEmbed`), `mlx_port/vlm_loader.py` |
| Prefill | Backbone over vision + ego traj tokens + prompt | `mlx_port/inference.py` (`_run_single_vlm_generation` first `vlm(...)`), `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoLanguageModel`) |
| Decode | Autoregressive CoC, one token per VLM call | `mlx_port/inference.py` (decode loop + `sample_next_token`), `mlx_port/models/token_utils_mlx.py` (`ExpertLogitsProcessor`, `StopAfterEOS`) |
| Action | 10× expert Euler steps on VLM KV, then unicycle | `mlx_port/inference.py` (`_sample_one_trajectory`), `mlx_port/models/expert_mlx.py`, `mlx_port/models/action_in_proj_mlx.py`, `mlx_port/models/alpamayo_r1_mlx.py` (`FlowMatching`, `ActionSpace`) |

Entry: `sample_trajectories_from_data_with_vlm_rollout` in `mlx_port/inference.py`.

## Encode

- **Class:** mlx_vlm Qwen3-VL `vision_tower` with `AlpamayoPatchEmbed` (channels-first Conv3D; stock mlx_vlm `moveaxis` is bypassed).
- **Input:** 4 cameras × `DEFAULT_NUM_FRAMES=4` = 16 RGB frames (`mlx_port/processor.py`). Native 1080×1920; processor `MIN_PIXELS=163840`, `MAX_PIXELS=196608`.
- **Layout:** processor flats are HF `C*T*H*W` (1536 = 3×2×16×16). `AlpamayoPatchEmbed` views 2D as `(N, 3, 2, 16, 16)`. Do not reshape as `(N, T, H, W, C)`.
- **Grid:** NVIDIA eval is 16×`[1, H, W]` (one temporal group per image). Do not collapse to 4×`[4, H, W]`.
- **Tokens:** live greedy prefix is ~32k (`offset`/`prefix_len` ≈ 32777 on the default clip). Vision is most of that. Ego history is 48 discrete tokens (`DEFAULT_HISTORY_TRAJ_TOKENS`).

## Prefill / backbone

- **Class:** `AlpamayoLanguageModel` / `AlpamayoModel` wrapping mlx_vlm Qwen3-VL text (`mlx_port/models/alpamayo_qwen3vl.py`).
- **Lineage:** Cosmos-Reason / Qwen3-VL-8B. 36 layers. Hidden 4096. GQA 8 KV heads, head_dim 128.
- **KV:** `mlx_lm.models.cache.KVCache`, one list of 36 caches created before the first forward. Prefill writes the long prefix; decode appends.
- **RoPE:** `get_rope_index` + cached `_rope_deltas` (default-clip live: −31680). Do not re-enable temporal grouping on greedy e2e.

## Decode / generation loop

- Manual loop (not HF `generate`). Default `max_generation_length=256`. Official / eval `num_traj_samples=1`.
- Sampling: greedy `T=0` or NVIDIA `T=0.6`, `top_p=0.98` (`sample_next_token` in `inference.py`).
- Stop: `StopAfterEOS` on `<|traj_future_start|>` plus one token. Do not mask `<|im_end|>`.
- **Not compiled.** No `mx.compile` in `mlx_port/`.
- **Host sync every token:** `mx.eval(outputs.logits)` after each decode forward; `int(next_token.item())` every token; `sample_next_token` upcasts to float32 and goes through NumPy softmax / nucleus.

## Action expert

- **Class:** `AlpamayoExpert` — mlx_vlm Qwen3-VL attention wrapper (`mlx_port/models/expert_mlx.py`). Not stock `mlx_lm` Qwen3 (that RoPEs at `cache.offset` and ignores NVIDIA `position_ids`).
- **Width:** 2048 / 16 Q heads / 8 KV / 36 layers (GQA kept from VLM). `action_in_proj` → `(B, 64, 2048)`; `action_out_proj` is `Linear(2048, 2)`.
- **KV in:** same VLM cache list. `position_ids = arange(64) + rope_deltas + offset`. Additive pad mask. Crop cache back to prefill length after each Euler step (`trim_cache` + `sync_cache_idx`).
- **FM:** `FlowMatching`, Euler, `num_inference_steps=10` (`alpamayo_r1_mlx.py`). `x ← x + dt·v`. `n_waypoints=64`.
- **t0:** `dxy_theta_to_v_without_v0`. Then `action_to_traj`.

## Current knobs (infer)

| Knob | Value in port |
|---|---|
| `num_traj_samples` | 1 (API default and e2e / traj-sample) |
| `num_traj_sets` | 1 |
| `max_generation_length` | 256 |
| temperature / top_p | 0.0 / 1.0 greedy; 0.6 / 0.98 NVIDIA draw |
| FM steps | 10 |
| compiled? | no (encode / prefill / decode / FM) |

## Dtype (T0.2)

| Piece | Dtype |
|---|---|
| VLM / vision / backbone weights | `mx.bfloat16` (`AlpamayoR1MLX.from_pretrained(..., dtype=mx.bfloat16)`) |
| Expert weights | `mx.bfloat16` (same load) |
| `action_in_proj` hot path | **casts `x` and `t` to float32** (`action_in_proj_mlx.py`) then LayerNorm → expert bf16 |
| `action_out_proj` | follows expert hidden (bf16) |
| Logits / nucleus | float32 on host |
| `ActionSpace` solvers | float32 / float64 NumPy Cholesky — post-FM, not the 10 expert forwards |

float32 Fourier / encoder inside `action_in_proj` is a Phase 2 item (hot: 10× per window), not Phase 3 quant.

## `mx.eval` / `.item()` / NumPy in one window (T0.2)

Counted for `num_traj_samples=1`, CoC length `N` tokens, FM steps 10.

| Site | Count | Why it hurts |
|---|---|---|
| After VLM prefill (`mx.eval(outputs.logits)`) | 1 | OK — one barrier after encode+prefill |
| Prefill logits → NumPy (`np.array(raw_last)`) | 1 | Host copy of vocab logits |
| After first decode token (`mx.eval(next_token)`) | 1 | Extra |
| After every decode forward (`mx.eval(outputs.logits)`) | **N** | Dominant if CoC is long |
| `int(next_token.item())` per token | **N** | Forces the step |
| `sample_next_token` NumPy softmax / top-p | **N** | Host sample |
| `FlowMatching.sample` `mx.eval(x)` per Euler step | **10** | Plan: one eval after the FM loop |
| `mx.clear_cache()` every 3 FM steps | 3–4 | Extra sync / allocator churn |
| `np.asarray(sampled)` after FM | 1 | Debug / extra dict |

**None of encode / prefill / decode / FM is `mx.compile`d.**

Likely 110 s split (unmeasured — Phase 1): prefill over ~32k tokens + per-token decode eval + 10 uncompiled expert forwards. Do not spend a week on ego history.

## Train vs infer (for later T2.4 / T4.3)

There is no SFT train step in this port yet. If a future loop calls
`sample_trajectories_from_data_with_vlm_rollout` every optimizer step, that is
the bug: NVIDIA Stage 1/2 is teacher-forced CE / one CFM draw, not a 256-token
CoC + 10 Euler steps.
