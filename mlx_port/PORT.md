# MLX port map (Stage 1c Phase 0)

Filled from code on 2026-08-29, not from memory. One inference window =
encode → prefill → CoC decode → 10-step flow-matching → `action_to_traj`.

Gate G1 signed off 2026-08-29 (user). Gate G3 signed off 2026-08-31 (user):
quality is human inspection of the 5-clip plots, not a 50-clip ADE band.
Canonical window: P2f greedy e2e,
encode 1044 · prefill 3645 · decode 856 / 13 tok · FM 630 / 10 steps.
Dominant = prefill. T3.1 / all4 stay opt-in memory. Do not 4-bit the expert
or vision as the default load.

## Four stages → files

| Stage | What runs | Primary files |
|---|---|---|
| Encode | Qwen3-VL vision tower over 16 RGB frames | `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoModel.get_input_embeddings`), `mlx_port/models/alpamayo_r1_mlx.py` (`AlpamayoPatchEmbed`), `mlx_port/vlm_loader.py` |
| Prefill | First language-model forward on the prompt (`seq_len > 1`) + `mx.eval(logits)` | `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoModel.__call__`), `mlx_port/models/alpamayo_qwen3vl.py` (`AlpamayoLanguageModel`) |
| Decode | Autoregressive CoC, one token per VLM call | `mlx_port/inference.py` (decode loop + `sample_next_token`), `mlx_port/models/token_utils_mlx.py` (`ExpertLogitsProcessor`, `StopAfterEOS`) |
| Action | 10× expert Euler steps on VLM KV, then unicycle | `mlx_port/inference.py` (`_sample_one_trajectory`), `mlx_port/models/expert_mlx.py`, `mlx_port/models/action_in_proj_mlx.py`, `mlx_port/models/alpamayo_r1_mlx.py` (`FlowMatching`, `ActionSpace`) |

Entry: `sample_trajectories_from_data_with_vlm_rollout` in `mlx_port/inference.py`.

Progress HTML and smoke JSON live in `mlx_port/reports/`. User how-tos live in
`mlx_port/doc/` (LoRA vs dense: `mlx_port/doc/train_lora_vs_dense.html`).
Script defaults resolve those dirs from `mlx_port/paths.py`.

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
| VLM / vision / backbone weights | `mx.bfloat16` load. Default path stays dense (`quantize_lm=False` / unset / `ALPAMAYO_QUANT=none`). Opt-in T3.1: `quantize_lm=True` or `ALPAMAYO_QUANT=lm4` uses `{alpamayo}/mlx_lm4/language_model.safetensors` (6.02 GiB, 252 QuantizedLinear). Opt-in all4: `quantize_all=True` or `ALPAMAYO_QUANT=all4` uses `{alpamayo}/mlx_all4/` (`vlm.safetensors` 4.79 GiB + `expert.safetensors` 1.19 GiB). Exclusive. `vlm8|vlm4|nvfp4` raise. |
| Expert weights | `mx.bfloat16` on the default and T3.1 paths. all4 packs the diffusion expert (252 QuantizedLinear). Action-in/out stay bf16. |
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

T3.1 greedy (2026-08-29 20:51, `quantize_lm=True`): 252 QuantizedLinear, `lm_head` + `embed_tokens` dense. encode 1070 · prefill 4427 · decode 695 / 13 tok · FM 1299. CoC pin held. Prefill is not faster than P2f (first W4 compile). G3 signed 2026-08-31 on qualitative inspection (ADE band waived). Opt-in memory recipe (−9.30 GiB weights, Metal peak ~45 GB vs P2f 55.10). Default load is the signed P2f bf16 path.

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

5-clip T=0.6 after P2f (`mlx_port/reports/traj_sample_5_t06/`, 2026-08-30 00:30 UTC): all clips `tokens=3006`, 16×`[1,20,36]`. Mean minADE 7.85 → 4.56 m vs native-32k snapshot `results_pre_p2f.json`. User reviewed plots 2026-08-29 and signed quality as good; that inspection is the G3 quality close (2026-08-31). Yield clips still overshoot (pred speed does not settle to ~0). ADE numbers stay in the report; they are not a G3 pass/fail.

Path check 2026-08-30 03:58 / 04:00 UTC (current code, `--no-reuse`): dense bf16 default and T3.1 opt-in both complete. Reports `traj_sample_5_t06_bf16/` · `traj_sample_5_t06_t31/` · side-by-side `traj_sample_5_t06_bf16_vs_t31/`. Mean minADE 4.56 / 4.24 m. CoC same on clips 2–4. Load logs: `[QUANT] language tower dense bf16` vs `252 QuantizedLinear`.

5-clip T3.1 disk sanity 2026-08-30 04:18 UTC (`ALPAMAYO_QUANT=lm4 --quantize-lm --no-reuse`, `mlx_port/reports/traj_sample_5_t06_t31_disk/`): skipped 399 language keys, loaded 252 QuantizedLinear from `mlx_lm4/`. Mean minADE 4.24 m. CoC and ADE match live-pack `traj_sample_5_t06_t31` to many decimals. Wall 96.9 s. Signed `traj_sample_5_t06/` left untouched.

Packed all4 (2026-08-30, `stage1c-p3-all4-disk`): `{alpamayo}/mlx_all4/` — VLM 342 QuantizedLinear + 2 QuantizedEmbedding (27 vision `linear_fc2` listed dense, in=4304 ≉ 64) 4.79 GiB; expert 252 QuantizedLinear 1.19 GiB. Incomplete trio raises. Disk load still pulls Alpamayo `patch_embed` so Conv3D is channels-first before `load_weights`. Greedy e2e (`ALPAMAYO_QUANT=all4`): 28.15 s · encode 1116 · prefill 3821 · decode 280 / 13 · FM 487 · CoC pin held · minADE 5.217 m · Metal 9.66 GB · RSS 27.67 GB. Prefill still not below P2f 3645. 5-clip T=0.6 (`mlx_port/reports/traj_sample_5_t06_all4_disk/`): mean minADE 3.18 m · 122.4 s wall. T=0.6 is not a quality ranking.

### all4 vs T3.1 leaves

Weight-only affine-4 gs64. Pack axis = Linear / Embedding last dim. Activations stay bf16. Walks are `vlm` then `expert` — never `AlpamayoR1MLX`. Any packable Linear left dense raises.

| Piece | T3.1 `lm4` | all4 |
|---|---|---|
| Language decoder `q/k/v/o` + `gate/up/down` (36×7=252) | 4-bit | 4-bit |
| `lm_head` + `embed_tokens` (vocab 155697) | bf16 | 4-bit |
| Vision `attn.qkv` / `proj` / `linear_fc1` (27 each) | bf16 | 4-bit |
| Vision `mlp.linear_fc2` (27×, in=4304) | bf16 | **bf16** (`4304 % 64 = 16`, listed) |
| Vision merger + 3 deepstack `fc1`/`fc2` (in=4608) | bf16 | 4-bit |
| Vision `pos_embed` | bf16 | QuantizedEmbedding |
| Conv3D `patch_embed.proj`, all RMS/LayerNorms | bf16 | bf16 (no `to_quantized`) |
| Expert decoder 36×7 (hidden 2048 / MLP 8256) | bf16 | 4-bit (all last dims ÷64) |
| `action_in_proj` / `action_out_proj` | bf16 (Fourier `freqs` fp32) | same — siblings, not in the walker |
| `FlowMatching`, history tokenizer | no GEMM | no GEMM |

## Train vs infer (NVIDIA SFT recipe)

Illustrated LoRA vs dense vs mixture map (Stage 1 VLM + Stage 2 expert):
`mlx_port/doc/train_lora_vs_dense.html`.
Load expert as bf16: `--expert-bf16` (was `--dense-expert`).
Adam leftover action I/O with expert LoRA: `--train-action-proj`
(was `--expert-dense`). Old names error with the rename.

SFT is `mlx_port/train_step.py` (`sft_train_step`). It is not the infer
rollout. Default `--from-clip` is NVIDIA public SFT:

- **Stage 1:** teacher-forced VLM CE on discrete traj-future (128 fused
  action bins + `<|traj_future_start|>` / `<|traj_future_end|>`) plus
  assistant `<|im_end|>`. Two-mean CE (`future_traj` + `others`). No expert.
- **Stage 2:** freeze VLM (including Stage-1 LoRA A/B), same fused string,
  crop KV at `<|traj_future_start|>`, one CFM draw + one expert forward.
  Expert attn is non-causal (`expert_non_causal_train_mask` zeros
  `(B,1,T,prefix+T)` — mlx_vlm would otherwise install a causal mask when
  `mask is None`). Default train is dense bf16 expert + action proj
  (`quantize_expert=False` / `--expert-bf16`). Packed
  `QuantizedLinear.weight` cannot Adam-step: `--expert-update` with
  `--quantize-all` requires `--expert-bf16`. Opt-in `--expert-lora`
  packs the all4 expert and QLoRAs the 36 decoder layers
  (`q/k/v/o/gate/up/down`); `action_in_proj` / `action_out_proj` stay
  dense and frozen unless `--train-action-proj` also Adam-steps them.
  Exclusive with `--expert-update`. Script:
  `mlx_port/scripts/sft_stage2_small.py`.
- **`joint`:** CE + CFM (`cotrain_vlm`). `--teacher-cot` is paper 5.2 CoC CE.

The step raises if it decoded tokens or ran Euler. Infer
`sample_trajectories_from_data_with_vlm_rollout` is unchanged (256-token CoC +
10 Euler steps). Hist fusion at infer stays on `<i0>` (signed). Future IDs
also start at `<i0>` (`future_token_start_idx=151669`). NVIDIA
`ReasoningVLA` offsets hist by `traj_tokenizer.vocab_size` (3000); this port
does not. Stage-1 user content matches NVIDIA `build_conversation` (images,
hist text, prompt text as separate items). Parity suite:
`mlx_port/tests/test_sft_nvidia_parity.py`.

all4 PAI Stage 1 (2026-08-31): seq=3124 · n_ce=131 · n_future=130 ·
n_others=1 · CE 3.77 · encode 1106 · backbone 3945 · total 5143 ms ·
1 VLM / 0 expert / 0 Euler · Metal 10.58 GB / RSS 23.66 GB.
all4 PAI Stage 2: same string · CFM 0.73 · expert 70 ms · total 5145 ms ·
1 VLM / 1 expert · Metal 7.99 GB / RSS 24.95 GB.

## QLoRA (T4.1)

`mlx_port/lora.py` wraps decoder `q/k/v/o/gate/up/down` (36×7=252) with
mlx_lm `LoRALinear` (rank 8, scale 20). Vision scope is
`--lora-vision full|merger|none` (default `full`):
`full` = 27 blocks `qkv`/`proj`/`fc1`/`fc2` + merger + 3 deepstack (116);
`merger` = merger + 3 deepstack only (8); `none` = language only.
After freeze, only `lora_a` / `lora_b` train. Conv3D `patch_embed`,
LayerNorms, `pos_embed`, `lm_head`, embeddings stay frozen.
`inject_backbone_lora` still skips the expert. Opt-in
`inject_expert_lora` wraps the same 36×7 leaves on the diffusion
decoder (`expert.language_model.model.layers`). Action in/out stay
dense. Packed `QuantizedLinear.weight` must hash-match after a step
(language + vision + expert when packed). Train uninstalls
`CompiledPrefillLayer` (compile+grad at seq=3024 hit ~200 GB Metal,
killed). Vision LoRA keeps encode on the tape
(`freeze_vision_features` raises). Language-only still encodes once and
`stop_gradient`s. `time_train_step.py --lora` is Stage-1 only.
`--expert-lora` is Stage-2 only. `--train-action-proj` requires `--expert-lora`
and Adam-steps leftover dense action in/out (writes `dense.safetensors`
next to expert adapters). Packed decoder ints stay frozen.

all4 dummy Stage-1 (2026-08-31, seq=64, lr=1e-5, 50 steps, language-only):
loss 19.12 → 0.89 · ~250 ms/step · wall 12.8 s · Metal 10.66 GB / RSS 23.80 GB.
lr=1e-4 overshoots dummy CE. PAI language-only (eager, vision stop-grad,
CoC-span CE): loss 2.59 · 11.6 s · Metal 66.48 GB / RSS 24.99 GB.
PAI Stage-1 + vision `full` (2026-08-31, seq=3124, n_ce=131, encode on tape):
736 arrays / 26.2M · loss 3.7714 · 24.8 s · Metal 107.47 GB / RSS 23.27 GB.
Packed `682:0b69306e…` unchanged.
PAI Stage-1 + vision `merger` (2026-08-31): 520 arrays / 22.4M · loss 3.7714 ·
12.5 s · Metal 76.79 GB / RSS 25.01 GB. Packed `601:1c370532…` unchanged.
First-step CE matches Stage-1 forward 3.7684 (LoRA B is zeros).

Small-scale Stage-1 (2026-08-31): 8 non-CoC clips, 4/4 split, t0=5.1 s,
`--lora-vision none`, all4, 10 Adam steps lr 1e-5. Per-clip train drops on
revisit. Eval mean 2.087 → 2.016. Metal 78.55 GB / RSS 26.15 GB.
30-clip follow-up (15/15, same recipe): eval 2.738 → 2.495 (down every
step). Train 2.652 / 2.843 on ten distinct clips (no revisit). Metal
82.31 GB / RSS 28.11 GB. Script: `mlx_port/scripts/sft_stage1_small.py`.
JSON: `mlx_port/reports/sft_stage1_small_10step.json`,
`mlx_port/reports/sft_stage1_small_30clip.json`.
QLoRA A/B (2026-08-31): `save_lora_adapters` overwrites
`mlx_port/reports/qlora/sft_stage1_small/adapters.safetensors` (plus `adapter_config.json`).
`--lora-save-every` default 10; same filename each save. 8-clip save sanity
(2026-08-31): eval 2.0874→2.0165 matches the first 8-clip; wrote
`mlx_port/reports/qlora/sft_stage1_small/adapters.safetensors` (87.4 MB, step=10).

Stage-2 8-clip (2026-08-31): freeze those adapters, dense bf16 expert,
lr 1e-4, same 4/4 split. `traj_to_action` kappa now matches NVIDIA
`solve_xs_eq_y` (the `dtheta/s` form made clip 7744 CFM ~11k). After the
fix: train 0.35→2.17 clip-wise, eval 0.202→0.310, ~4.8 s/step, Metal
32.3 GB / RSS 33.9 GB. Packed VLM fingerprint unchanged. Script:
`mlx_port/scripts/sft_stage2_small.py`. `--expert-lora` is the packed-expert
QLoRA path (252 decoder leaves; action proj frozen). Expert adapters
write to `mlx_port/reports/qlora/sft_stage2_small/` (gitignored), not the Stage-1
file. 8-clip `--expert-lora` (2026-08-31): all4 VLM+expert, 252 expert
leaves / 13.0 M, lr 1e-4, same 4/4. eval 0.198→0.185, train clip-wise
1.45→0.19, ~4.4 s/step, Metal 11.72 GB / RSS 26.31 GB. Packed
`1097:9b90d23a…` unchanged. JSON:
`mlx_port/reports/sft_stage2_small_8clip_expert_qlora.json`. Adapters 50 MB /
504 arrays / step=10.
`--train-action-proj` 8-clip (2026-08-31, ran as `--expert-dense`): same split / lr. eval 0.188→0.219
(spike 0.494). train 1.75→0.29 clip-wise. Metal 11.74 / RSS 26.18.
Packed hash match. `dense.safetensors` 2.6 MB / 15 arrays. JSON:
`mlx_port/reports/sft_stage2_small_8clip_expert_qlora_dense.json`.

G4 candidate 8-clip 2-epoch (2026-08-31, awaiting user): `--epochs 2` → 8
Adam steps on the same seed-0 4/4. Stage 1 language QLoRA all4: eval
1.853→1.827, per-clip train down on revisit, Metal 78.68 GB. Adapters
`mlx_port/reports/qlora/sft_stage1_small_8clip_2ep/` (83 MB, step=8).
Stage 2 loads those adapters, `--expert-lora --train-action-proj`: eval
0.260→0.180, Metal 11.66 GB, packed `1097:9b90d23a…`. Adapters 50 MB +
`dense.safetensors` 2.6 MB under
`mlx_port/reports/qlora/sft_stage2_small_8clip_2ep/`.
JSON: `mlx_port/reports/sft_stage1_small_8clip_2ep.json`,
`mlx_port/reports/sft_stage2_small_8clip_2ep.json`. G4 not signed.

Train stage wall-clock (2026-08-31, T1.1 analog): same G4 clip
`77447940…`, t0=5.1 s, seq=3124. Prints a `[TRAIN]` line
(tokenize / encode_cache / encode / backbone / expert / loss /
fwd_bwd / adam). Adam evals `loss, grads` then `parameters` so
backward is not lazy-attributed to `opt.update`. Combined JSON:
`mlx_port/reports/train_stage_times.json`.

| | Infer P2f | Train S1 | Train S2 |
|---|---|---|---|
| tokenize | — | 15.2 s once | 12.8 s once |
| encode | 1044 ms | 1111 ms cache, then 0 | same |
| backbone | 3645 ms compiled | 4279 ms eager | 3984 ms compiled (frozen) |
| decode | 856 / 13 | 0 | 0 |
| FM / expert | 630 / 10 Euler | 0 | 73 ms / 1 CFM |
| fwd+bwd | — | **13592 ms** | 4085 ms (≈ fwd) |
| Adam | — | 63 ms | 70 ms |
| dominant | prefill | fwd_bwd | fwd_bwd ≈ backbone |
| Metal | 55.10 GB | 78.54 GB | 12.27 GB |

Stage 1 Adam is ~3× the materialized forward (LoRA backward at 3124).
Stage 2 Adam matches forward because KV is `stop_gradient`. The 8-clip
script wall is eval-bound (4 clips × backbone after every step).
