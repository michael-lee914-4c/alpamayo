# MLX port map (Stage 1c Phase 0)

Filled from code on 2026-08-29, not from memory. One inference window =
encode → prefill → CoC decode → 10-step flow-matching → `action_to_traj`.

Gate G1 signed off 2026-08-29 (user). Canonical window: P2f greedy e2e,
encode 1044 · prefill 3645 · decode 856 / 13 tok · FM 630 / 10 steps.
Dominant = prefill. Phase 3 T3.1 (language-tower affine 4-bit) is the current increment
(`stage1c-p3-t31-affine-lm`). Do not 4-bit the expert or vision first.

## Four stages → files

| Stage | What runs | Primary files |
|---|---|---|
| Encode | Qwen3-VL vision tower over 16 RGB frames | `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoModel.get_input_embeddings`), `mlx_port/models/alpamayo_r1_mlx.py` (`AlpamayoPatchEmbed`), `mlx_port/vlm_loader.py` |
| Prefill | First language-model forward on the prompt (`seq_len > 1`) + `mx.eval(logits)` | `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoModel.__call__`), `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoLanguageModel`) |
| Decode | Autoregressive CoC, one token per VLM call | `mlx_port/inference.py` (decode loop + `sample_next_token`), `mlx_port/models/token_utils_mlx.py` (`ExpertLogitsProcessor`, `StopAfterEOS`) |
| Action | 10× expert Euler steps on VLM KV, then unicycle | `mlx_port/inference.py` (`_sample_one_trajectory`), `mlx_port/models/expert_mlx.py`, `mlx_port/models/action_in_proj_mlx.py`, `mlx_port/models/alpamayo_r1_mlx.py` (`FlowMatching`, `ActionSpace`) |

Entry: `sample_trajectories_from_data_with_vlm_rollout` in `mlx_port/inference.py`.

## Encode

- **Class:** mlx_vlm Qwen3-VL `vision_tower` with `AlpamayoPatchEmbed` (channels-first Conv3D; stock mlx_vlm `moveaxis` is bypassed).
- **Input:** 4 cameras × `DEFAULT_NUM_FRAMES=4` = 16 RGB frames (`mlx_port/processor.py`). Native 1080×1920.
- **Pixel budget:** `MIN_PIXELS=163840`, `MAX_PIXELS=196608` (NVIDIA helper / SFT). `get_processor` calls `bind_image_pixel_budget` after load. Qwen3-VL JSON `size.longest_edge=16777216` is ignored. 1080×1920 → ~320×576, grid `[1, 20, 36]` (~180 tokens/frame). Before the bind, live greedy was native `68×120` (~32k prefix).
- **Layout:** processor flats are HF `C*T*H*W` (1536 = 3×2×16×16). `AlpamayoPatchEmbed` views 2D as `(N, 3, 2, 16, 16)`. Do not reshape as `(N, T, H, W, C)`.
- **Grid:** NVIDIA eval is 16×`[1, H, W]` (one temporal group per image). Do not collapse to 4×`[4, H, W]`.
- **Tokens:** greedy e2e 2026-08-29 15:25: `input_ids` `(1, 3006)`, `image_grid_thw` 16×`[1,20,36]`, `pixel_values` `(11520, 1536)`. Vision is 16×180 = 2880 pads. Ego history is 48 discrete tokens (`DEFAULT_HISTORY_TRAJ_TOKENS`).

## Prefill / backbone

- **Class:** `AlpamayoLanguageModel` / `AlpamayoModel` wrapping mlx_vlm Qwen3-VL text (`mlx_port/models/alpamayo_qwen3vl.py`).
- **Lineage:** Cosmos-Reason / Qwen3-VL-8B. 36 layers. Hidden 4096. GQA 8 KV heads, head_dim 128.
- **KV:** `mlx_lm.models.cache.KVCache`, one list of 36 caches created before the first forward. Prefill writes the long prefix; decode appends.
- **RoPE:** `get_rope_index` + cached `_rope_deltas` (default-clip live: −31680). Do not re-enable temporal grouping on greedy e2e.

## Decode / generation loop

- Manual loop (not HF `generate`). Default `max_generation_length=256`. Official / eval `num_traj_samples=1`.
- Sampling: greedy `T=0` or NVIDIA `T=0.6`, `top_p=0.98` (`sample_next_token` in `inference.py`).
- Stop: `StopAfterEOS` on `<|traj_future_start|>` plus one token. Do not mask `<|im_end|>`.
- **Prefill compiled.** Each Qwen3-VL decoder layer is `mx.compile`d for `seq_len > 1` (`mlx_port/models/compiled_backbone.py`). Decode and FM are not compiled.
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
| compiled? | prefill yes (decoder layers, seq>1); encode / decode / FM no |

## Dtype (T0.2)

| Piece | Dtype |
|---|---|
| VLM / vision / backbone weights | `mx.bfloat16` load. Default path stays dense (`quantize_lm=False` / unset / `ALPAMAYO_QUANT=none`). Opt-in T3.1: `quantize_lm=True` or `ALPAMAYO_QUANT=lm4` uses `{alpamayo}/mlx_lm4/language_model.safetensors` (6.02 GiB, 252 QuantizedLinear) when present; otherwise live-packs and saves there. `lm_head` + `embed_tokens` stay bf16 in the packed file. Vision / expert stay bf16. The VLM-W8/W4 / all4 / nvfp4 walks were probes and are gone. |
| Expert weights | `mx.bfloat16` (same load; never 4-bit in T3.1) |
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
| `FlowMatching.sample` `mx.eval(x)` after the Euler loop | **1** | P2a (was 10 per-step evals) |
| `mx.clear_cache()` inside FM | **0** | Removed with the per-step barrier; was every 3 steps |
| `np.asarray(sampled)` after FM | 1 | Debug / extra dict |

Prefill decoder layers are `mx.compile`d (`seq_len > 1`). Encode, decode, and FM are not.

Stage wall-clock (T1.1): set `ALPAMAYO_STAGE_TIMERS=1`. Prints a `[STAGE]` line and
puts `extra["stage_times"]` with G1 fields. Off by default so the e2e path is
unchanged.

**Prefill is the first token.** Time `language_model(...)` on the prompt
(`seq_len > 1`) plus one `mx.eval` of logits, inside `AlpamayoModel.__call__`.
After token 1, each one-token LM call is decode (warm KV). `decode_tok` is
tokens after the first. Encode is still `vision_tower` only.

Measure with greedy e2e: `test_end_to_end_inference_prints_coc_vlm_only` and
`ALPAMAYO_STAGE_TIMERS=1`. Do not use a separate multi-window warmup test.

P2f greedy window (2026-08-29 15:25): encode 1.0 s · prefill 3.6 s · decode 0.9 s · FM 0.6 s. Prefix is 3006 tokens after the pixel-budget bind. Do not spend a week on ego history.

T3.1 greedy (2026-08-29 20:51, `quantize_lm=True`): 252 QuantizedLinear, `lm_head` + `embed_tokens` dense. encode 1070 · prefill 4427 · decode 695 / 13 tok · FM 1299. CoC pin held. Prefill is not faster than P2f (first W4 compile). G3 not signed. Opt-in memory recipe (−9.30 GiB weights, Metal peak ~45 GB vs P2f 55.10). Default load is the signed P2f bf16 path.

Packed T3.1 language tower (2026-08-30): `pre-trained/Alpamayo-R1-10B/mlx_lm4/language_model.safetensors` (6.02 GiB) + `config.json`. First save was live-pack (`save_quantized_lm`, no expert, 8.0 s). Reload skipped 399 Alpamayo language keys, loaded 252 QuantizedLinear from disk (2.8 s, `load_expert=False`). Greedy e2e from disk (`ALPAMAYO_QUANT=lm4`): 22.73 s · encode 1041 · prefill 3856 · decode 306 / 13 · FM 587 · CoC pin held · minADE 4.097 m · Metal 21.24 GB · RSS 40.61 GB. `ALPAMAYO_LM4_DIR` / `lm4_path` override the directory. Incomplete pair raises.

Prefill quant chase closed 2026-08-29. One-shot greedy, same clip / seq=3006. Metal dequants tiles into FP16/BF16 MMA — no native 4-bit compute on M4. Every extra recipe was slower on prefill than P2f 3645 ms. Code for those walks is deleted; numbers stay here.

| Recipe | Packed | encode | prefill | decode | FM | wall | Metal peak |
|---|---|---|---|---|---|---|---|
| P2f bf16 | none | 1044 | **3645** | 856 / 13 | **630** | 27.66 s | 55.10 GB |
| T3.1 W4 LM | 252 decoder | 1070 | 4427 | 695 / 13 | 1299 | 36.01 s | (not copied) |
| VLM W8 | 342 + 2 embed; 27 fc2 dense | 1253 | 4134 | 849 / 13 | 759 | 32.13 s | 46.98 GB |
| VLM W4 | same 342 + 2 | 1187 | 6538 | **664** / 13 | 1411 | 34.51 s | 45.41 GB |
| all4 | VLM W4 + 252 expert | 1129 | 4466 | 677 / 13 | 1131 | 28.52 s | 45.34 GB |
| nvfp4 | 369 + 2 (fc2 packs at gs16) | 1269 | 5897 | 726 / 12 | 1378 | 30.41 s | 45.40 GB |

Decode W4 is the only real speed signal (−160 ms). Metal peak drops ~10 GB once the decoder is 4-bit; packing vision / head / expert does not move that peak (sampled during first `vlm()`). RSS high-water stayed ~63 GB. minADE on these shots is unseeded FM — ignore. CoC pin held on 12–13 greedy tokens.

5-clip T=0.6 after P2f (`reports/traj_sample_5_t06/`, 2026-08-30 00:30 UTC): all clips `tokens=3006`, 16×`[1,20,36]`. Mean minADE 7.85 → 4.56 m vs native-32k snapshot `results_pre_p2f.json`. User reviewed plots 2026-08-29 and signed quality as good. Yield clips still overshoot (pred speed does not settle to ~0).

Path check 2026-08-30 03:58 / 04:00 UTC (current code, `--no-reuse`): dense bf16 default and T3.1 opt-in both complete. Reports `traj_sample_5_t06_bf16/` · `traj_sample_5_t06_t31/` · side-by-side `traj_sample_5_t06_bf16_vs_t31/`. Mean minADE 4.56 / 4.24 m. CoC same on clips 2–4. Load logs: `[QUANT] language tower dense bf16` vs `252 QuantizedLinear`.

5-clip T3.1 disk sanity 2026-08-30 04:18 UTC (`ALPAMAYO_QUANT=lm4 --quantize-lm --no-reuse`, `reports/traj_sample_5_t06_t31_disk/`): skipped 399 language keys, loaded 252 QuantizedLinear from `mlx_lm4/`. Mean minADE 4.24 m. CoC and ADE match live-pack `traj_sample_5_t06_t31` to many decimals. Wall 96.9 s. Signed `traj_sample_5_t06/` left untouched.

## Train vs infer (for later T2.4 / T4.3)

There is no SFT train step in this port yet. If a future loop calls
`sample_trajectories_from_data_with_vlm_rollout` every optimizer step, that is
the bug: NVIDIA Stage 1/2 is teacher-forced CE / one CFM draw, not a 256-token
CoC + 10 Euler steps.
