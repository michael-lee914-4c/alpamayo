# Alpamayo 1 MLX port

Apple Silicon / MLX reimplementation of Alpamayo 1 **inference** and
**small-scale SFT**. It is not a drop-in replacement for the CUDA
package in `src/alpamayo_r1` or the Hydra trainers under `finetune/`.

This page is the developer landing page: setup, public APIs, workflows,
and pitfalls. The lab log (signed timings, quant walks, G1–G4 notes)
lives in [PORT.md](PORT.md). LoRA vs dense training maps are in
[doc/train_lora_vs_dense.html](doc/train_lora_vs_dense.html).

## Requirements

| Requirement | Constraint |
| ----------- | ---------- |
| **OS / accelerator** | macOS + Apple Silicon (Metal). Linux CI cannot import `mlx`. |
| **Python** | 3.12.x |
| **Install** | `uv pip install --group mlx-ci` — **not** `uv sync`. The default project deps pull CUDA packages (`flash-attn`, `vllm`, …). |
| **Weights** | Local `Alpamayo-R1-10B` directory with `config.json` + safetensors. Hub-only ids are not accepted by `AlpamayoR1MLX.from_pretrained`. |
| **VLM processor** | Local Qwen3-VL-8B-Instruct processor path (see defaults below). |
| **PAI clips** | Local Physical AI AV snapshot for data scripts / e2e tests. |

Import the package from the repo root (`PYTHONPATH=<repo>:<repo>/src`).
CI sets that in [`.github/workflows/mlx-unit-tests.yml`](../.github/workflows/mlx-unit-tests.yml).

## Default local paths

Scripts and tests hardcode one developer's checkout. Override them.

| Asset | Default in code | Override |
| ----- | --------------- | -------- |
| Alpamayo checkpoint | `/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B` | `--alpamayo-path` / `AlpamayoR1MLX.from_pretrained(...)` |
| Qwen3-VL processor | `/Users/michaellee/Projects/alpamayo/pre-trained/Qwen3-VL-8B-Instruct` | `vlm_model_path=` on `from_pretrained` |
| PAI-CoC snapshot | `/Volumes/MicronSSD/pai_coc` | `--local-dir` / `load_physical_aiavdataset(..., local_dir=...)` |
| Reports / HTML | `mlx_port/reports/` | `--report` / `--report-dir` |
| LoRA adapters | `mlx_port/reports/qlora/` (gitignored) | `--lora-save-dir` / `--expert-lora-save-dir` |

Packed quant checkpoints, when present, live next to the Alpamayo
weights:

- T3.1 language tower: `{alpamayo}/mlx_lm4/language_model.safetensors` + `config.json`
- all4 VLM + expert: `{alpamayo}/mlx_all4/vlm.safetensors` + `expert.safetensors` + `config.json`

Incomplete pairs/trios raise. Override with `ALPAMAYO_LM4_DIR` /
`ALPAMAYO_ALL4_DIR` or the `lm4_path=` / `all4_path=` kwargs.

## Architecture (one inference window)

```
16 RGB frames (4 cameras × 4)  →  encode (vision tower)
prompt (seq_len > 1)           →  prefill (compiled decoder layers)
autoregressive CoC             →  decode (one VLM call / token)
10 Euler FM steps on VLM KV    →  action expert → unicycle `action_to_traj`
```

- **Prefill is the first token.** Later one-token LM calls are decode.
- Prefill decoder layers are `mx.compile`d (`seq_len > 1`). Encode,
  decode, and flow-matching are not.
- Official / eval default is `num_traj_samples=1`.
- Stop after `<|traj_future_start|>` plus one token. Do not mask
  `<|im_end|>`.
- Vision grid is 16×`[1, H, W]` (one temporal group per image). Do not
  collapse to 4×`[4, H, W]`.
- Pixel budget is bound after processor load: `MIN_PIXELS=163840`,
  `MAX_PIXELS=196608`. Native 1080×1920 becomes ~320×576
  (`[1, 20, 36]` ≈ 180 tokens/frame). The Qwen JSON
  `size.longest_edge=16777216` is ignored.

Entry: `sample_trajectories_from_data_with_vlm_rollout` in
[`inference.py`](inference.py).

## Setup

```bash
cd /path/to/alpamayo
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --group mlx-ci
export PYTHONPATH="$PWD:$PWD/src"
```

`scripts/apply_mlx_vlm_patches.sh` looks for `patches/mlx_vlm_generate.patch`.
That file is not in this tree; the port subclasses mlx_vlm instead of
patching it. Do not treat the script as a required setup step.

## Load and infer

Default load is **dense bf16**. Opt-in 4-bit is memory-only; Metal
dequantizes into FP16/BF16 MMA, so prefill is not faster than dense.

```python
import mlx.core as mx
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from mlx_port.inference import sample_trajectories_from_data_with_vlm_rollout
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.processor import (
    DEFAULT_NUM_FRAMES,
    alpamayo_apply_chat_template,
    create_message,
    get_processor,
)

model = AlpamayoR1MLX.from_pretrained(
    "/path/to/Alpamayo-R1-10B",
    load_expert=True,
    dtype=mx.bfloat16,
)
processor = get_processor(model.tokenizer)

data = load_physical_aiavdataset(
    clip_id,
    t0_us=5_100_000,          # NVIDIA default keyframe; CoC eval uses the event t0
    local_dir="/path/to/pai",
    num_frames=DEFAULT_NUM_FRAMES,
)
messages = create_message(data["image_frames"].flatten(0, 1))
tokenized = alpamayo_apply_chat_template(
    processor,
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="np",
)

pred_xyz, pred_rot, extra = sample_trajectories_from_data_with_vlm_rollout(
    model,
    {
        "tokenized_data": tokenized,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    },
    temperature=0.6,   # 0.0 = greedy
    top_p=0.98,
    num_traj_samples=1,
    max_generation_length=256,
    return_extra=True,
)
print(extra["cot"][0])
```

5-clip local sample (writes XY/speed plots under `mlx_port/reports/`):

```bash
python -m mlx_port.scripts.run_local_traj_sample
python -m mlx_port.scripts.run_local_traj_sample --quantize-lm --no-reuse
python -m mlx_port.scripts.run_local_traj_sample --quantize-all --no-reuse
```

`--quantize-lm` and `--quantize-all` are exclusive. Stage timers:

```bash
ALPAMAYO_STAGE_TIMERS=1 python -m pytest mlx_port/tests/test_end_to_end_inference.py -k greedy -s
```

## Quantization

`ALPAMAYO_QUANT` may be unset, `none`, `lm4`, or `all4`. Anything else
raises. Env overrides the `quantize_lm=` / `quantize_all=` kwargs.

| Mode | What is 4-bit | What stays bf16 | Typical use |
| ---- | ------------- | --------------- | ----------- |
| unset / `none` | — | everything | Signed default infer (P2f) |
| `lm4` | language decoder 36×7 | vision, expert, `lm_head`, embeddings | Opt-in memory (T3.1) |
| `all4` | VLM + expert decoder | action in/out; vision `linear_fc2` (`in=4304` ≉ 64) | Opt-in memory; Stage-1 SFT smoke |

Do **not** 4-bit the expert or vision as the default infer load.

Pack once, then reload from disk:

```bash
python -m mlx_port.scripts.save_quantized_lm --alpamayo-path /path/to/Alpamayo-R1-10B
python -m mlx_port.scripts.save_quantized_all --alpamayo-path /path/to/Alpamayo-R1-10B
```

`--force` deletes an existing packed dir and live-packs again.

Stage-2 full expert FT needs a **dense** expert: packed
`QuantizedLinear.weight` cannot Adam-step. Use `--expert-bf16` on
`time_train_step` (all4 VLM, dense expert) or `--expert-lora` (pack the
expert and train A/B only).

## SFT (train graph ≠ infer graph)

`sft_train_step` in [`train_step.py`](train_step.py) is the official
train graph. It **raises** if the step decoded tokens or ran Euler.
Do not call `sample_trajectories_from_data_with_vlm_rollout` or
`FlowMatching.sample` from training.

| Stage | What trains | Loss | Script |
| ----- | ----------- | ---- | ------ |
| **stage1** | language QLoRA A/B (vision frozen in the smoke) | two-mean CE: `future_traj` + `others` (assistant `<|im_end|>`) | `mlx_port/scripts/sft_stage1_small.py` |
| **stage2** | diffusion expert (VLM + Stage-1 adapters frozen) | one CFM draw + one expert forward | `mlx_port/scripts/sft_stage2_small.py` |
| **joint** | CE + CFM (`cotrain_vlm`) | both | `time_train_step.py --stage joint` (no CLI smoke script) |

Stage-1/2 smokes:

- 50/50 split on **non-CoC** local clips (`exclude_coc=True`). The
  default CoC eval clip is rejected if it appears in the traj pool.
- Keyframe `t0_us=5_100_000` (NVIDIA `use_default_keyframe`), not a CoC
  event.
- `--n-clips` must be ≥ 2 (default 8 → 4 train / 4 eval).
- `--epochs` and `--steps` are exclusive. Default is 10 Adam steps.
- Stage-1 adapters overwrite `adapters.safetensors` every
  `--lora-save-every` steps (default 10); the last step is always saved.
- Stage-2 `--expert-lora` writes a **different** directory
  (`reports/qlora/sft_stage2_small/`). `--train-action-proj` also writes
  `dense.safetensors` (action in/out). Old flags `--dense-expert` /
  `--expert-dense` error with the rename.

```bash
# Stage 1: language QLoRA, all4 VLM (default)
python -m mlx_port.scripts.sft_stage1_small \
  --alpamayo-path /path/to/Alpamayo-R1-10B \
  --local-dir /path/to/pai_coc \
  --epochs 2

# Stage 2: load those adapters, dense bf16 expert FT
python -m mlx_port.scripts.sft_stage2_small \
  --alpamayo-path /path/to/Alpamayo-R1-10B \
  --local-dir /path/to/pai_coc \
  --lora-adapter-dir mlx_port/reports/qlora/sft_stage1_small \
  --epochs 2

# Stage 2 QLoRA expert + leftover dense action proj
python -m mlx_port.scripts.sft_stage2_small \
  --expert-lora --train-action-proj \
  --lora-adapter-dir mlx_port/reports/qlora/sft_stage1_small
```

Single-step timing / graph checks:

```bash
python -m mlx_port.scripts.time_train_step --from-clip --stage stage1 --lora --lora-vision none
python -m mlx_port.scripts.time_train_step --from-clip --stage stage2 --quantize-all --expert-bf16 --expert-update
```

`--from-clip` ignores `--seq-len` and tokenizes a real PAI window
(seq ≈ 3124 after the pixel-budget bind).

### Train vs NVIDIA notes that have bitten people

- Hist fusion at infer stays on `<i0>`. Future IDs also start at
  `<i0>` (`future_token_start_idx=151669`). NVIDIA `ReasoningVLA`
  offsets hist by `traj_tokenizer.vocab_size` (3000); this port does
  not.
- Stage-2 expert attention is **non-causal**
  (`expert_non_causal_train_mask` zeros `(B,1,T,prefix+T)`). Leaving
  `mask is None` lets mlx_vlm install a causal mask.
- `traj_to_action` uses NVIDIA `solve_xs_eq_y`. The older `dtheta/s`
  form produced huge CFM on some clips.
- Stage-1 smoke freezes the vision tower (`--lora-vision` is not
  exposed there). `time_train_step --lora --lora-vision full|merger|none`
  is the vision-LoRA probe.
- Adapter `rank` / `scale` passed to Stage 2 must match the Stage-1
  inject (defaults 8 / 20). A CLI mismatch rescales the frozen VLM.
- `labels_mask_between` / `traj_future_keep_len` are **row-0**
  (smoke batches are `B=1`).

## Public interfaces

| Module | Role |
| ------ | ---- |
| `mlx_port.models.alpamayo_r1_mlx.AlpamayoR1MLX` | Load + fuse + expert. `from_pretrained(alpamayo_path, ...)` |
| `mlx_port.inference.sample_trajectories_from_data_with_vlm_rollout` | Infer window (CoC + 10 Euler steps) |
| `mlx_port.processor` | `create_message`, `get_processor`, `alpamayo_apply_chat_template`, pixel-budget bind |
| `mlx_port.train_step.sft_train_step` | `stage=` `stage1` / `stage2` / `joint` |
| `mlx_port.lora` | `inject_backbone_lora`, `inject_expert_lora`, `save_lora_adapters`, `load_lora_adapters` |
| `mlx_port.gt_eval` | Local PAI-CoC listing, CoC Jaccard, XY minADE |
| `alpamayo_r1.load_physical_aiavdataset.load_physical_aiavdataset` | Shared clip loader (CUDA + MLX) |

`from_pretrained` knobs (env `ALPAMAYO_QUANT` wins when set):

| Kwarg | Default | Meaning |
| ----- | ------- | ------- |
| `load_expert` | `True` | Required when `quantize_all` is on |
| `quantize_lm` | `False` | T3.1 language-tower pack |
| `quantize_all` | `False` | all4 VLM + expert (exclusive with `quantize_lm`) |
| `quantize_expert` | `True` | When all4: pack expert. Set `False` so Stage-2 Adam can update it |
| `dtype` | `mx.bfloat16` | Load dtype |

## Tests and CI

```bash
export PYTHONPATH="$PWD:$PWD/src"
export ALPAMAYO_CI_NO_WEIGHTS=1          # skip weight / SSD tests
python -m pytest mlx_port/tests -m "not slow" -q --tb=short
```

- Required GitHub check: **macos-26 (no weights)**
  ([`.github/workflows/mlx-unit-tests.yml`](../.github/workflows/mlx-unit-tests.yml)).
- `conftest.py` also skips files when `physical_ai_av`, the Alpamayo
  checkpoint, the Qwen processor, or `/Volumes/MicronSSD/pai_coc` are
  missing.
- `@pytest.mark.slow` needs local weights, Metal, or the PAI-CoC SSD.
- Linux runners cannot load `libmlx.so`. Do not expect
  `import mlx` to work there.

NVIDIA SFT string/token parity (no weight load when
`ALPAMAYO_CI_NO_WEIGHTS=1`): `mlx_port/tests/test_sft_nvidia_parity.py`.

## Environment variables

| Variable | Effect |
| -------- | ------ |
| `ALPAMAYO_QUANT` | `none` / `lm4` / `all4` (overrides kwargs) |
| `ALPAMAYO_LM4_DIR` | Packed T3.1 directory |
| `ALPAMAYO_ALL4_DIR` | Packed all4 directory |
| `ALPAMAYO_STAGE_TIMERS` | `1` prints `[STAGE]` encode/prefill/decode/FM |
| `ALPAMAYO_DEBUG` | `1` extra decode / RoPE prints |
| `ALPAMAYO_CI_NO_WEIGHTS` | Skip weight-load tests in `conftest.py` |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | Set in CI; useful once weights are local |

## Troubleshooting

| Symptom | Likely cause | What to do |
| ------- | ------------ | ---------- |
| `FileNotFoundError: …/config.json` | Hub id or wrong folder | Pass the local Alpamayo directory that contains `config.json` |
| CoC decodes as placeholder `<iN>` / wrong specials | Tokenizer added fewer than `traj_vocab_size` (4000) `<iN>` tokens before specials | Use `load_vlm_with_alpamayo_tokens`; see `test_tokenizer_id_parity.py` |
| Prefix ~32k tokens / huge prefill | Pixel budget not bound | `get_processor` must call `bind_image_pixel_budget` |
| `image_grid_thw` is 4×`[4,H,W]` | Temporal grouping re-enabled | Keep NVIDIA's 16×`[1,H,W]` |
| Stage-2 CFM in the thousands | Wrong `traj_to_action` kappa | Must be `solve_xs_eq_y`, not `dtheta/s` |
| `QuantizedLinear` / Adam error on Stage 2 | Packed expert cannot train | `--expert-bf16 --expert-update` or `--expert-lora` |
| `--expert-dense` / `--dense-expert` errors | Renamed | `--train-action-proj` / `--expert-bf16` |
| CoC clip in Stage-1/2 split | `select_non_coc_clips` leaked | Pool excludes `reasoning/ood_reasoning.parquet` |
| Metal OOM on Stage-1 LoRA + compile | Compiled prefill + grad at seq≈3124 | Train uninstalls `CompiledPrefillLayer` |
| `import mlx` fails on Linux | No Metal build | Use macos-26 CI or a Mac |

## Layout

```
mlx_port/
├── README.md                 # this page
├── PORT.md                   # signed timings / quant walks
├── inference.py              # infer rollout
├── train_step.py             # SFT graphs
├── lora.py                   # QLoRA inject / save / load
├── processor.py              # messages + pixel budget
├── vlm_loader.py             # Qwen load + Alpamayo tokens
├── gt_eval.py                # PAI-CoC listing / ADE / CoC score
├── paths.py                  # reports/ and doc/ roots
├── models/                   # VLM, expert, quant, RoPE, tokenizer
├── scripts/                  # infer / SFT / pack / timers
├── tests/
├── doc/                      # LoRA vs dense how-to
└── reports/                  # HTML / JSON smokes (some gitignored)
```
