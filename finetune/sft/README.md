# CUDA SFT package

Launch commands, GPU counts, and example metrics live in
[`docs/FINETUNE_SFT.md`](../../docs/FINETUNE_SFT.md). This page is the
developer map for `finetune/sft/`: two-stage architecture, weight-load
contracts, train vs eval graphs, and pitfalls verified on main
(`4f4c049`, PR #16).

This is the NVIDIA Hydra / DeepSpeed path. It is not the MLX smoke in
`mlx_port/scripts/sft_stage*.py` (`sft_train_step` never decodes or
runs Euler).

## Intent

Stage 1 teaches the VLM to emit discrete future-trajectory tokens.
Stage 2 freezes that VLM and trains the flow-matching action expert
for continuous 64-step unicycle actions. The two graphs are different
classes, different checkpoints, and different losses. Eval never uses
the Stage-2 train loss — it always rolls out like inference.

## Package map

| Path | Role |
| ---- | ---- |
| `train_hf.py` | Hydra train entry. Instantiates model / PAI / collate / callbacks, then `ReasoningVLA_Trainer.train()`. |
| `evaluate_hf.py` | Hydra eval entry. Rewrites `eval_ckpt` onto the model load path, runs `MetricRunner` on `val_dataset`. |
| `trainer.py` | `TrainingArguments.lr_multiplier` + prefix-grouped Adam. Unmatched params stay at 1.0×. |
| `models/sft_base_model.py` | Stage 1: `TrainableReasoningVLA` + `load_alpamayo1_vlm` + future-token fusion. |
| `models/sft_alpamayo_r1.py` | Stage 2: `TrainableAlpamayoR1` (CFM expert on cropped VLM KV). |
| `configs/sft_base.yaml` | Shared data, trainer, eval. `data.*.local_dir` is `???` (must override). |
| `configs/sft_stage1.yaml` | Stage 1: `ar1_base` + DeepSpeed ZeRO-2 + visual LR 0.1×. |
| `configs/sft_stage2.yaml` | Stage 2: `ar1_expert`, no DeepSpeed, no grad checkpoint. |
| `configs/models/ar1_base.yaml` | `_target_`: `TrainableReasoningVLA.from_alpamayo_checkpoint`. |
| `configs/models/ar1_expert.yaml` | `_target_`: `TrainableAlpamayoR1.from_pretrained`. |
| `configs/vla_processor.yaml` | Qwen conversation: `image`, `traj_history`, `prompt`, `traj_future`. |

`pyproject.toml` ships `alpamayo_r1` and `finetune/rl` only. `finetune.sft`
is importable from a checkout (`python -m` adds cwd). Run train/eval from
the repo root.

```bash
torchrun --nproc_per_node 8 -m finetune.sft.train_hf \
  --config-path pkg://finetune/sft/configs --config-name sft_stage1 \
  data.train_dataset.local_dir=/path/to/pai \
  data.val_dataset.local_dir=/path/to/pai \
  model.checkpoint_path=/path/to/Alpamayo-R1-10B
```

`pkg://finetune/sft/configs` needs `finetune/sft/configs/__init__.py` on
`PYTHONPATH`. Hydra `???` fields (`local_dir`, Stage-1 `checkpoint_path`)
raise at instantiate if left unset.

## Stage 1 — discrete VLM CE

`TrainableReasoningVLA.from_alpamayo_checkpoint(checkpoint_path, vlm_name_or_path)`:

1. Reads `{checkpoint_path}/config.json` (not the Hub model card).
2. Builds the wrapper + trajectory tokenizer from that config. Does
   **not** download the base Qwen3-VL weights.
3. `load_alpamayo1_vlm(checkpoint_path, model)` copies **`vlm.*`** tensors
   from `{checkpoint_path}/model.safetensors.index.json` onto the **full
   wrapper**. Keys match (`vlm.model…`, `vlm.lm_head…`).

A directory with only `model.safetensors` (no index) raises
`No vlm.* tensors found`. Sharded Trainer outputs
(`model.safetensors.index.json` + shards) are the expected layout.

### Train graph

`forward` fuses hist + future discrete tokens into the `<|traj_history|>` /
`<|traj_future|>` pads, then two-mean CE:

| Term | Tokens |
| ---- | ------ |
| `future_traj` | IDs in `[future_token_start_idx, future_token_start_idx + traj_vocab_size)` plus `<|traj_future_start|>` / `<|traj_future_end|>` |
| `others` | remaining labels that are not `IGNORE_INDEX` (−100) |

Collate already restricts `labels_mask` to the future span **and** the
assistant `<|im_end|>`, so `others` is typically that EOS. Empty
`labels_mask[:, 1:]` returns CE `0.0`. Logits are cloned before the
token mask; `nan_to_num` maps NaN CE to 0.

`lr_multiplier: {vlm.model.visual: 0.1}` (longest prefix wins). DeepSpeed
ZeRO-2 is on; `train_hf` then forces
`ds_config._dtype = torch.float32` so the trajectory encoder is not
cast to bf16 with the rest of the forward.

## Stage 2 — frozen VLM + CFM expert

`configs/models/ar1_expert.yaml` instantiates
`TrainableAlpamayoR1.from_pretrained` with:

| Kwarg | Bundled default | Meaning |
| ----- | --------------- | ------- |
| `pretrained_model_name_or_path` | `nvidia/Alpamayo-R1-10B` | Full Alpamayo 1 wrapper (VLM + expert + action proj). Prefer a local dir. |
| `cotrain_vlm` | `false` | `false` → freeze `self.vlm`, `torch.no_grad()` around the VLM, loss is CFM only. `true` adds VLM CE. |
| `stop_grad_from_vlm` | `true` (constructor default) | Detach cropped KV before the expert. |
| `stage1_vlm_checkpoint_path` | `null` | Intended Stage-1 Trainer dir. See [overlay](#stage-2-vlm-overlay-current-main). |

### Train graph (not inference)

1. Fuse hist + **teacher** future tokens (same pads as Stage 1).
2. VLM forward with `use_cache=True` (no decode).
3. Crop KV at the **last** `<|traj_future_start|>` in the flattened
   `nonzero` + 1. Default `per_device_train_batch_size` is 1; a larger
   batch uses one global index for every row.
4. One CFM draw (`diffusion.construct_training_data`) → `action_in_proj`
   → one expert forward (`is_causal=False` when
   `config.expert_non_causal_attention`) → `action_out_proj`.
5. Loss is `diffusion.compute_loss_from_pred`. No Euler, no
   `max_generation_length`.

`traj_to_action` prepends the last history state (treated as origin).
Expert RoPE is `arange(64) + rope_deltas + kv_len`, 3-way Qwen ids.

### Stage-2 VLM overlay (current main)

The launch line in `docs/FINETUNE_SFT.md` is:

```text
model.pretrained_model_name_or_path=/path/to/Alpamayo-R1-10B
model.stage1_vlm_checkpoint_path=/path/to/stage1/output/checkpoint-xxxx
```

On current main that overlay does **not** apply the Stage-1 VLM:

1. `__init__` calls `load_alpamayo1_vlm(path, self.vlm)`. The helper
   keeps keys named `vlm.*`. The dest is the inner Qwen3-VL
   (`model.*` / `lm_head.*` / `visual.*`). `strict=False` logs a full
   miss and keeps the release tower.
2. Hydra uses `from_pretrained`. HuggingFace constructs the module
   (step 1), then copies `pretrained_model_name_or_path` onto the
   wrapper, which would overwrite a working overlay anyway.

`from_alpamayo_checkpoint` (Stage 1) loads onto the **wrapper**, so
`vlm.*` keys match. Only Stage 2 is broken. Open fix: PR #49. Until
that lands, Stage 2 trains the expert against the release Alpamayo-R1
tower, not your Stage-1 run.

## Train vs eval

| | Train (`generation_mode: false`) | Eval (`generation_mode: true`) |
| - | -------------------------------- | ------------------------------ |
| Chat | Full assistant future span: start + `tokens_per_future_traj` pads + end | `continue_final_message=True`; last component is only `<|traj_future_start|>` |
| Collate `labels_mask` | Future span ∪ assistant `<|im_end|>` | All-false |
| Stage 1 `forward` | Two-mean CE on fused teacher tokens | Unused (`evaluate_hf` does not call `forward`) |
| Stage 2 `forward` | One CFM step on cropped KV | Unused |
| Predictions | None | `sample_trajectories_from_data` → CoC generate + 10-step FM (Stage 2) or discrete decode (Stage 1) |

`evaluate_hf` remaps `evaluate.eval_ckpt`:

| Model class | Field written |
| ----------- | ------------- |
| `TrainableReasoningVLA` | `model.checkpoint_path` (index + `vlm.*` shards) |
| `TrainableAlpamayoR1` | `model.pretrained_model_name_or_path` (full HF dir) |

Then `MetricRunner` runs `ReasoningSampler` then `DistanceMetrics`.
Sampler defaults: `num_traj_samples=6`, `num_traj_sets=1`, `top_p=0.98`,
`temperature=0.6`, `max_generation_length=256`. Stage 2 eval stops on
`<|traj_future_start|>` + one token (`StopAfterEOS`) and masks discrete
traj logits (`ExpertLogitsProcessor`).

`DistanceMetrics` scores the **last** traj group. Horizons are
`int(t / 0.1)` for 0.5 / 1 / 3 / 5 s. With `num_traj_sets=1` it skips
`_std` / `_sq`. Dummy logprob is zeros, so `ade` is the first sample.
`evaluate_hf` averages keys starting `metric/` and reprints them as
`val/metric/…`.

## Data knobs this package actually sets

`PAIDataset` + `QwenProcessor` (see `sft_base.yaml` / `vla_processor.yaml`):

| Knob | Bundled value | Constraint |
| ---- | ------------- | ---------- |
| `chunk_ids` | train `"0-99"`, val `"99-100"` | End-exclusive `range(start, end)`. `"0-10"` is 0–9. |
| `use_default_keyframe` | `true` | Every clip at `DEFAULT_T0_US = 5_100_000` (5.1 s). |
| `vla_preprocess_args.generation_mode` | train `false`, val `true` | Switches future pads vs `continue_final_message`. |
| `label_components` | `["traj_future"]` | Mask fills start…end inclusive. |
| `remove_unused_columns` | `false` | Required. HF Trainer would otherwise drop `tokenized_data` / ego tensors. |
| `include_extr_intr` / `reshape_tensors_for_rl` | off | RL-only. Leave off for this path. |

`local_dir` must contain `features.csv`, `clip_index.parquet`, and the
downloaded camera / egomotion blobs. Clip load, cameras, and the event-t0
filter are in `src/alpamayo_r1/data/pai.py` and
`load_physical_aiavdataset.py`.

Collate left-pads, `torch.cat`s every `tokenized_data` key except
`text`, then **re-tokenizes** `text` and overwrites `input_ids`. Do not
pre-stack fused ids on `tokenized_data["input_ids"]` — they are discarded.

## Pitfalls

- **Run from the repo root.** `finetune.sft` is not in the installed
  wheel. `pkg://finetune/sft/configs` fails if cwd is not on
  `PYTHONPATH`.
- **Stage 1 needs a local sharded Alpamayo dir**, not only a Hub id.
  `from_alpamayo_checkpoint` reads `config.json` + the safetensors
  index.
- **Stage 2 overlay is silent on current main.**
  `stage1_vlm_checkpoint_path` does not replace the release VLM
  (see [above](#stage-2-vlm-overlay-current-main)). Watch the
  `Loaded N VLM tensors … unexpected=` log; a full unexpected set
  means nothing applied.
- **Train loss is not minADE.** Stage 2 CFM can look healthy while
  eval rollout (CoC + Euler) is not. `val/metric/min_ade` is the
  number in `docs/FINETUNE_SFT.md`.
- **Do not enable DeepSpeed on Stage 2** (`sft_stage2.yaml` sets
  `deepspeed: null`). Stage 1 is the ZeRO-2 + grad-checkpoint recipe.
- **`cotrain_vlm: true` is not the bundled recipe.** It unfreezes the
  VLM and adds CE on the same fused string. Official scripts stay
  two-stage.
- **W&B is off** until you uncomment `- /wandb: default` in
  `sft_base.yaml`, set `trainer.report_to: wandb`, and fill
  `configs/wandb/default.yaml`.
- **Seed is hardcoded** (`misc.seed_everything(42)` in `train_hf.py`).
- **macOS `mlx-ci` does not run this package.**
  `.github/workflows/mlx-unit-tests.yml` is `pytest mlx_port/tests -m
  "not slow"` with `ALPAMAYO_CI_NO_WEIGHTS=1`. CUDA SFT/eval is
  manual on 8× H100.

## Related

- User launch guide: [`docs/FINETUNE_SFT.md`](../../docs/FINETUNE_SFT.md)
- RL post-training: [`finetune/rl/README.md`](../rl/README.md)
- Public CUDA infer / PAI types: `src/alpamayo_r1/`
  (`AlpamayoR1.sample_trajectories_from_data_with_vlm_rollout`,
  `PAIDataset`)
