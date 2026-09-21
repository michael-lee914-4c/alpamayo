# Inference and PAI data contracts

Public CUDA interfaces used by `src/alpamayo_r1/test_inference.py`,
[SFT](FINETUNE_SFT.md), and [RL](../finetune/rl/README.md). Verified against
source on main (`4f4c049`, PR #16). Download/curate CLIs live in
[`scripts/`](../scripts/).

## Load one clip

```python
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset

data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
# optional: local_dir="/path/to/pai_dataset" (sets maybe_stream=False yourself)
```

`t0_us` defaults to **5.1 s**. The call raises when
`t0_us <= num_history_steps * time_step` (1.6 s at the defaults). The earliest
history stamp is `t0 − (n−1)·dt` (1.5 s); tightening the inequality would
admit clips the event filter already dropped.

### Time axes (defaults: 16 hist / 64 future / 4 frames / 0.1 s)

| Tensor | Stamps | Includes t0? |
| ------ | ------ | ------------ |
| `ego_history_*` | `[t0−1.5s, …, t0]` | yes (last step) |
| `ego_future_*` | `[t0+0.1s, …, t0+6.4s]` | no |
| `image_frames` | `[t0−0.3s, t0−0.2s, t0−0.1s, t0]` | yes (last frame) |

Ego xyz/rot are rewritten into the **t0 local frame**:
`xyz_local = R_t0^{-1} @ (xyz_world − xyz_t0)`. The last history point is the
origin; t0 orientation is identity. Relative image timestamps are
`(abs − global_tmin) * 1e-6` across cameras (not per-camera).

### Cameras

Default cameras (then **argsort by index**):

| Name | Index |
| ---- | ----- |
| `camera_cross_left_120fov` | 0 |
| `camera_front_wide_120fov` | 1 |
| `camera_cross_right_120fov` | 2 |
| `camera_front_tele_30fov` | 6 |

Unknown names map to **0**, then sort. Rear cameras 3–5 exist in the map but
are not loaded by default.

### Returned shapes

| Key | Shape |
| --- | ----- |
| `image_frames` | `(N_cam, num_frames, 3, H, W)` uint8 NCHW |
| `camera_indices` | `(N_cam,)` |
| `ego_history_xyz` / `ego_future_xyz` | `(1, 1, T, 3)` |
| `ego_history_rot` / `ego_future_rot` | `(1, 1, T, 3, 3)` |
| `relative_timestamps` / `absolute_timestamps` | `(N_cam, num_frames)` |

## `PAIDataset` (SFT / RL / notebooks)

[`alpamayo_r1.data.pai.PAIDataset`](../src/alpamayo_r1/data/pai.py) wraps a
local tree (`features.csv`, `clip_index.parquet`, `metadata/**`).

| Knob | Bundled SFT (`sft_base.yaml`) | Meaning |
| ---- | ----------------------------- | ------- |
| `chunk_ids` | train `"0-99"`, val `"99-100"` | **End-exclusive** `range(start, end)`. `"0-10"` is chunks 0–9. A hyphen-less string such as `"0,1"` is iterated as characters (`list("0,1")`) and typically matches nothing. `None` = all chunks. |
| `use_default_keyframe` | `true` | `true` → every clip at `DEFAULT_T0_US` (5.1 s). `false` → `event_t0s[0]` after the safety filter. |
| `include_extr_intr` | off | Adds `extr`, `intr`, `ego_lwh`, `ego_length_offset`. Needs `sensor_extrinsics`, `camera_intrinsics`, and `vehicle_dimensions` on disk. |
| `reshape_tensors_for_rl` | off | Flattens cameras-then-frames to `(N_cam*n_frame, 1, C, H, W)` and `repeat_interleave`s camera ids. Ego tensors are squeezed, not flattened. |
| `vla_preprocess_args` | Qwen processor | Instantiated via Hydra; result stored as `tokenized_data`. |

`__getitem__` squeezes the leading batch dim on every `ego_*` key, so
`(1, 1, T, 3)` becomes `(1, T, 3)` — the traj-group axis is kept. If
`vla_preprocess_args` is set, preprocess runs **again after** extrinsics / RL
reshape; the first pass is discarded. Inspect a sample with
[`notebooks/inspect_dataset.ipynb`](../notebooks/inspect_dataset.ipynb).

### Event-t0 filter

`PhysicalAIAVDatasetLocalInterface` keeps events in
`[1.6 s, end_timestamp − 6.4 s]` and drops rows whose `event_t0s` become
empty. `get_clip_feature` returns `None` for an unknown feature name; a
`.bin` / unexpected extension raises.

## Run CUDA inference

```python
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# image_frames is (N_cam, n_frame, C, H, W) — flatten to rank-4 (N, C, H, W)
messages = helper.create_message(data["image_frames"].flatten(0, 1))
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=False,
    continue_final_message=True, return_dict=True, return_tensors="pt",
)
pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
    data={
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    },
    top_p=0.98, temperature=0.6, num_traj_samples=1,
    max_generation_length=256, return_extra=True,
)
```

Full script: [`src/alpamayo_r1/test_inference.py`](../src/alpamayo_r1/test_inference.py).
Interactive: [`notebooks/inference.ipynb`](../notebooks/inference.ipynb)
(directory is `notebooks/`, not `notebook/`).

### `helper.create_message` / `get_processor`

- Frames must be rank-4 `(N, C, H, W)`. 4 cameras × 4 frames → 16 images.
- Inserts **48** history pads between `<|traj_history_start|>` / `_end`.
- Assistant content is `<|cot_start|>` only (generation continues that span).
- `get_processor` loads `Qwen/Qwen3-VL-2B-Instruct` and binds
  `min_pixels=163840`, `max_pixels=196608` (same values as SFT
  `model_config`). Do not skip the bind — a raw Qwen processor keeps
  `size.longest_edge=16777216` and leaves a ~32k-token 1080p prefix.

### `sample_trajectories_from_data_with_vlm_rollout`

| Contract | Value in source |
| -------- | --------------- |
| `n_traj_group` | must be **1** |
| API default `num_traj_samples` | **6** (the test script passes `1` for 24 GB) |
| VLM `num_return_sequences` | `num_traj_samples` (not `× num_traj_sets`) |
| Diffusion batch | `B * num_traj_samples * num_traj_sets` |
| Pred layout | `(B, num_traj_sets, num_traj_samples, T, …)` |
| Sampling | `T=0.6`, `top_p=0.98`, `do_sample=True` |
| Stop | `StopAfterEOS` on `<|traj_future_start|>` **plus one token** |
| Traj-bin mask | `ExpertLogitsProcessor` sets `[start, start+vocab)` to `-inf` |
| Expert crop | **first** `<|traj_future_start|>` (`argmax`) + 1; missing → last token + 1 |
| Expert attn | non-causal (`expert_non_causal_attention=True`); pad span `offset:-n_diff` is `-inf` |
| Hist for unicycle | last traj group `ego_history[:, -1]` |

`return_extra=True` yields CoC text from `extract_text_tokens`, reshaped to
`[B, num_traj_sets, num_traj_samples]`.

## SFT preprocess vs this helper

Training does **not** use `helper.create_message`.
`QwenProcessor._preprocess_data` builds the conversation from
`components_order` / `components_prompt`, then:

- `continue_final_message=generation_mode` (train `False`, val/gen `True`)
- always `add_generation_prompt=False`
- uint8 images are `/255` with `do_rescale=False`

Bundled processor (`vla_processor.yaml`): train labels `traj_future` only;
val `generation_mode: true` asks for the last assistant span.

## Eval keys

`DistanceMetrics` uses the **last** traj group (`ego_future_xyz[:, -1]`).
Horizons `[0.5, 1, 3, 5]` s become steps via `int(t / 0.1)` →
`[5, 10, 30, 50]`. Keys are `min_ade`, `min_ade/by_t=0.5`, … .
`compute_ade` **raises** when the horizon exceeds `T`; `compute_minade`
skips those horizons. XY only unless `only_xy=False`.

`MetricRunner` prefixes `metric/`. `evaluate_hf` keeps only those keys and
rewrites them to `val/metric/…` (mean of `sum/numel`). Missing `pred_xyz`
returns `{}` (no crash). Dummy-zero logprob ADE therefore picks sample 0.

## Pitfalls

- **Chunk ranges are half-open.** SFT `"0-99"` is chunks 0–98; val `"99-100"`
  is chunk 99 only. The same split is used by `download_pai.py --chunk-ids`.
- **SFT t0 is 5.1 s, not the clip event**, because `use_default_keyframe: true`.
  RL YAML defaults to the clip event (`false`); the shipped
  `cosmos-rl` entry scripts override it back to `True`.
  `notebooks/inspect_dataset.ipynb` uses the event (class default).
- **Helper tokenize defaults are the infer path**
  (`continue_final_message=True`). Flipping that on a train conversation
  appends or drops the assistant suffix and retargets every CE label.
- **`get_processor` is required.** A raw `AutoProcessor.from_pretrained`
  without the pixel-budget kwargs leaves the 32k-token prefix.
- **API `num_traj_samples=6` OOM.** The test script and FAQ assume `1` on
  24 GB. Raising it also multiplies CoC traces (`return_extra`).
- **Unknown camera → index 0** (same as cross-left), then argsort. A typo
  therefore shares the left slot. `construct_image` only asserts ids are
  non-decreasing, so this does not raise.
- **MLX: do not reuse one loaded model across clips on current main.**
  `AlpamayoLanguageModel.get_rope_index` caches 3-row T/H/W positions and
  `rope_deltas` behind `_pos_valid` and returns them for every later call.
  `mlx_port/scripts/run_local_traj_sample.py` (and Stage-1/2 smokes) load
  once and loop clips, so clips 1–N prefill with clip 0's mRoPE when token
  counts match. Open fix: PR #37. Decode (no `pixel_values`) is unaffected.

## Related

- CUDA SFT how-to: [FINETUNE_SFT.md](FINETUNE_SFT.md)
- RL post-training: [finetune/rl/README.md](../finetune/rl/README.md)
- MLX port lab log (not a how-to): [mlx_port/PORT.md](../mlx_port/PORT.md)
