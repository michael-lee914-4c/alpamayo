# `alpamayo_r1` (CUDA package)

Public inference, PAI loading, SFT preprocess, and eval metrics for the
NVIDIA Alpamayo 1 release. Setup and FAQ live in the [root README](../../README.md).
Contracts, shapes, and pitfalls: **[docs/INFERENCE_AND_DATA.md](../../docs/INFERENCE_AND_DATA.md)**.

| Module | Role |
| ------ | ---- |
| `models.alpamayo_r1.AlpamayoR1` | VLM + action expert. `from_pretrained` + `sample_trajectories_from_data_with_vlm_rollout` |
| `load_physical_aiavdataset` | One-clip PAI → images, t0-local ego hist/future, camera ids |
| `data.pai.PAIDataset` | Local-tree Dataset used by SFT/RL (`chunk_ids` end-exclusive, optional 5.1 s keyframe) |
| `helper` | Infer chat (`create_message`, 48 hist pads) and pixel-budget `get_processor` |
| `processor.qwen_processor` | SFT/eval conversation + collate (`generation_mode` switches `continue_final_message`) |
| `metrics` | `DistanceMetrics` / `MetricRunner` (`metric/` → `val/metric/` in `evaluate_hf`) |
| `test_inference.py` | End-to-end smoke: load clip → generate CoC + traj → minADE |

SFT entry points are under [`finetune/sft/`](../../finetune/sft/)
([guide](../../docs/FINETUNE_SFT.md)). RL is
[`finetune/rl/`](../../finetune/rl/README.md). The Apple-Silicon port is
[`mlx_port/`](../../mlx_port/) (lab log: `PORT.md`).
