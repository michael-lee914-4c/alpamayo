# Repository scripts

Helpers under `scripts/` prepare **Physical AI AV (PAI)** data and convert
checkpoints between the HuggingFace release layout, Cosmos-RL training, and
HuggingFace-loadable exports. They are not the CUDA SFT/RL trainers and not
the MLX port.

| Script | Intent |
| --- | --- |
| [`download_pai.py`](download_pai.py) | Snapshot a PAI slice from HuggingFace (or the full ~97 TB set) |
| [`curate_pai_samples.py`](curate_pai_samples.py) | Sample a smaller `clip_index` parquet from already-downloaded chunks |
| [`convert_release_config_to_training.py`](convert_release_config_to_training.py) | HF release → Cosmos-RL / `ReasoningVLA` training directory |
| [`convert_cosmos_rl_checkpoint.py`](convert_cosmos_rl_checkpoint.py) | Cosmos-RL `model_rank_*.pth` shards → HF `*.safetensors` |
| [`apply_mlx_vlm_patches.sh`](apply_mlx_vlm_patches.sh) | Apply a local `mlx_vlm` patch (see [pitfalls](#pitfalls)) |

Full CUDA recipes: [SFT](../docs/FINETUNE_SFT.md) and [RL](../finetune/rl/README.md).
MLX inference and Stage-1/2 smokes live under [`mlx_port/scripts/`](../mlx_port/scripts/)
(stage map: [`mlx_port/PORT.md`](../mlx_port/PORT.md)).

## Workflow

```
HuggingFace release (Alpamayo-R1-10B or Alpamayo-1.5-10B)
        │
        ▼  convert_release_config_to_training.py
training-ready dir  ($ALPAMAYO_MODEL_DIR)  ← used by Cosmos-RL
        │
        ▼  cosmos-rl  …/checkpoints/step_<N>/policy/model_rank_*.pth
        │
        ▼  convert_cosmos_rl_checkpoint.py  --base-hf-ckpt $ALPAMAYO_MODEL_DIR
HF export  →  RLWrapperReasoningVLA.from_pretrained(...)

PAI Hub  →  download_pai.py  →  curate_pai_samples.py  →  $ALPAMAYO_PAI_LOCAL_DIR
```

## Download PAI

Gated dataset: [nvidia/PhysicalAI-Autonomous-Vehicles](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles).
Authenticate first (`hf auth login` or `HF_TOKEN`).

`--chunk-ids` is **end-exclusive**: `0-10` downloads chunks `0…9`. A single id
(`3116`) or space-separated list (`0 1`) also work. Omit `--chunk-ids` to
request every chunk.

The script always pulls `features.csv`, `clip_index.parquet`, and `metadata/**`.
Cameras, calibration, labels, lidar, and radar are opt-in via their flags.
Default `--output-dir` is `nvidia/PhysicalAI-Autonomous-Vehicles`.

SFT-style four-camera + egomotion slice (chunks 0–9):

```bash
export HF_TOKEN=<your Hugging Face token>
python scripts/download_pai.py --chunk-ids 0-10 \
  --camera camera_front_wide_120fov camera_cross_left_120fov \
           camera_cross_right_120fov camera_front_tele_30fov \
  --calibration camera_intrinsics sensor_extrinsics \
  --labels egomotion \
  --output-dir /path/to/pai_dataset
```

RL local-test example (chunk 3116) is in the [RL guide](../finetune/rl/README.md#5-download-a-subset-of-the-physical-ai-dataset).

Requires `huggingface_hub`. Missing it exits with an install hint.

## Curate a mini clip index

Does **not** copy media. It filters the existing parquet and writes a new one.
`--chunk` uses the same exclusive-range grammar as `--chunk-ids`. The column
is `chunk_id` if present, otherwise `chunk`. Sample size is
`min(--num-samples, matching rows)` with `random_state=11`. Default
`--clip-index-path` is `dataset/alpamayo/PAI_mini/clip_index.parquet`.

```bash
python scripts/curate_pai_samples.py \
  --clip-index-path /path/to/pai_dataset/clip_index.parquet \
  --chunk 3116 \
  --num-samples 16 \
  --output-path /path/to/pai_dataset/clip_index_mini.parquet
```

Point `PAIDataset` / RL Hydra at the curated parquet (`clip_index_metadata`)
so training does not walk the full downloaded index.

## Convert a release checkpoint for RL

Default `--alpamayo-model` is **`nvidia/Alpamayo-1.5-10B`**, not this repo's
Alpamayo 1 release. For Alpamayo-R1-10B pass `--alpamayo-model` explicitly.
Default `--output-dir` is `/tmp/<model-name>_training`.

The script:

1. Downloads the Alpamayo snapshot and tokenizer/preprocessor files from
   `--vlm-name-or-path` (default `nvidia/Cosmos-Reason2-8B`).
2. **Symlinks** `*.safetensors` and `model.safetensors.index.json` into the
   output directory (they resolve into the HuggingFace cache).
3. Rewrites `config.json`: `model_type=alpamayo_reasoning_vla`,
   `architectures=["ReasoningVLA"]`, sets `vlm_name_or_path`, and remaps
   Alpamayo 1.5 Hydra `_target_` prefixes (`alpamayo1_5.*` → `alpamayo_r1.*`).
   Alpamayo 1 (`alpamayo_r1.*`) needs no remap.
4. Loads `RLWrapperReasoningVLA` (adds `src/`, `finetune/`, and
   `finetune/rl/models` to `sys.path`) and writes the expanded processor.

```bash
# This repo (Alpamayo 1)
python scripts/convert_release_config_to_training.py \
  --alpamayo-model nvidia/Alpamayo-R1-10B \
  --output-dir "$ALPAMAYO_MODEL_DIR"

# RL guide default (Alpamayo 1.5)
python scripts/convert_release_config_to_training.py \
  --output-dir "$ALPAMAYO_MODEL_DIR"
```

## Export a Cosmos-RL policy checkpoint

Cosmos-RL writes per-rank `model_rank_<r>.pth` DTensor shards under
`<output_dir>/checkpoints/step_<N>/policy/`. Those files are for resume, not
`ReasoningVLA.from_pretrained()`.

This converter merges shards on CPU (1-D mesh `Shard` / `Replicate` only),
strips a `reasoning_vla.` key prefix, copies non-weight files from
`--base-hf-ckpt` (the directory from the previous step), and writes sharded
`model-XXXXX.safetensors` plus `model.safetensors.index.json` and
`cosmos_export_provenance.json`. Default `--max-shard-size` is `4GB`.
Existing config/tokenizer files are left alone unless you pass `--overwrite`.

```bash
python scripts/convert_cosmos_rl_checkpoint.py \
  --cosmos-policy-ckpt /path/to/checkpoints/step_<N>/policy \
  --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
  --output-dir /path/to/exported/step_<N>_hf
```

Load path used by the [RL export notebook](../finetune/rl/notebooks/rl_checkpoint_inference.ipynb):

```python
from reasoning_vla.base_model import RLWrapperReasoningVLA

model = RLWrapperReasoningVLA.from_pretrained("/path/to/exported/step_<N>_hf")
```

## Pitfalls

- **End-exclusive chunk ranges.** `0-10` is chunks `0…9` in
  `download_pai.py`, `curate_pai_samples.py`, and `PAIDataset(chunk_ids="0-10")`.
- **Full PAI is huge.** Omitting `--chunk-ids` asks for the entire dataset.
- **Release convert defaults to 1.5.** This tree is Alpamayo 1; pass
  `--alpamayo-model nvidia/Alpamayo-R1-10B` unless you intend 1.5.
- **Symlinked weights.** Deleting or evicting the HuggingFace cache breaks a
  training dir produced by `convert_release_config_to_training.py`.
- **Export needs the training-ready base.** `--base-hf-ckpt` must contain
  `config.json` plus tokenizer/processor files; the Cosmos policy dir is
  weights only.
- **1-D shard mesh only.** Multi-dimensional DTensor placements raise
  `NotImplementedError`.
- **`apply_mlx_vlm_patches.sh` is inert here.** It looks for
  `patches/mlx_vlm_generate.patch`, which is not in this tree, and exits
  if `mlx_vlm` is not importable. Do not treat it as a required setup step.
  MLX work uses [`mlx_port/`](../mlx_port/).
