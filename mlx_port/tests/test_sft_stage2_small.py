"""Small-scale Stage-2 split: freeze LoRA VLM, dense expert CFM, same 8-clip pool."""

import subprocess
import sys

import pytest

from mlx_port.gt_eval import REASONING_PATH
from mlx_port.scripts.sft_stage1_small import select_non_coc_clips


def test_stage2_split_matches_stage1_seed0():
    if not REASONING_PATH.exists():
        pytest.skip(f"PAI-CoC not on this machine ({REASONING_PATH})")
    train, ev = select_non_coc_clips(REASONING_PATH.parents[1], n_clips=8, seed=0)
    assert train == [
        "77447940-31f4-4230-a4e8-ad106de5ee5c",
        "b1195c93-50fb-45a9-8511-9afde3b778ed",
        "de3c23c3-8742-4ffe-b613-bfc2cb6e33cc",
        "2532fb2e-5068-44f6-9f64-2256d852d51d",
    ]
    assert ev == [
        "e55eeb5e-c957-408f-a640-a0466ae766b4",
        "76865f45-5f9e-432c-b29c-038111a20aad",
        "f52e773d-09ef-4abf-8399-735216e21387",
        "6707ab59-3e79-4c37-b41d-9c4baf28b292",
    ]


def test_sft_stage2_small_help_mentions_frozen_vlm():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.sft_stage2_small", "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    for token in (
        "Stage-2",
        "freeze",
        "flow-matching",
        "--lora-adapter-dir",
        "--expert-lr",
        "--expert-lora",
        "--train-action-proj",
        "--epochs",
        "Expert stays dense",
    ):
        if token not in proc.stdout:
            raise AssertionError(f"{token!r} missing from help:\n{proc.stdout}")


def test_sft_stage2_small_rejects_missing_adapters(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--lora-adapter-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when adapters are missing")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "adapters missing" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_expert_lora_rank_requires_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--expert-lora-rank",
            "16",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit when --expert-lora-rank has no --expert-lora")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "expert-lora" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_train_action_proj_requires_expert_lora():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--train-action-proj",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError(
            "expected non-zero exit when --train-action-proj has no --expert-lora"
        )
    err = (proc.stderr or "") + (proc.stdout or "")
    if "expert-lora" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_rejects_renamed_expert_dense():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--expert-dense",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for renamed --expert-dense")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "--train-action-proj" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_rejects_epochs_with_steps():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--epochs",
            "2",
            "--steps",
            "8",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --epochs + --steps")
    if "exclusive" not in (proc.stderr + proc.stdout):
        raise AssertionError(proc.stderr)


def test_sft_stage2_small_rejects_n_clips_below_two():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.sft_stage2_small", "--n-clips", "1"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --n-clips 1")
    if "n-clips" not in (proc.stderr + proc.stdout):
        raise AssertionError(proc.stderr)


def test_sft_stage2_small_no_expert_lora_save_requires_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--no-expert-lora-save",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --no-expert-lora-save without --expert-lora")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "expert-lora" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_expert_lora_save_dir_requires_flag():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--expert-lora-save-dir",
            "/tmp/expert-lora-missing-flag",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --expert-lora-save-dir without --expert-lora")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "expert-lora" not in err:
        raise AssertionError(err)


def test_sft_stage2_small_expert_lora_save_every_requires_positive():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage2_small",
            "--expert-lora",
            "--expert-lora-save-every",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --expert-lora-save-every 0")
    err = (proc.stderr or "") + (proc.stdout or "")
    if "expert-lora-save-every" not in err:
        raise AssertionError(err)
