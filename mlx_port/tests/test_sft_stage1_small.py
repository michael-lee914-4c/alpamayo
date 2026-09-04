"""Small-scale Stage-1 split: no CoC clips, 50/50, language-only QLoRA CLI."""

import subprocess
import sys

import pandas as pd
import pytest

from mlx_port.gt_eval import (
    DEFAULT_EVAL_CLIP_ID,
    REASONING_PATH,
    list_local_traj_clips,
    split_train_eval,
)
from mlx_port.scripts.sft_stage1_small import resolve_train_steps, select_non_coc_clips
from mlx_port.scripts.time_train_step import build_pai_train_batch


def test_split_train_eval_is_half_and_disjoint():
    ids = [f"c{i}" for i in range(20)]
    a, b = split_train_eval(ids, seed=0, train_frac=0.5)
    assert len(a) == 10
    assert len(b) == 10
    assert set(a) | set(b) == set(ids)
    assert not (set(a) & set(b))
    c, d = split_train_eval(ids, seed=0, train_frac=0.5)
    assert a == c and b == d
    e, f = split_train_eval(ids, seed=1, train_frac=0.5)
    assert (e, f) != (a, b)


def test_split_train_eval_rejects_tiny_or_dupes():
    try:
        split_train_eval(["only"], seed=0)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:
        raise AssertionError("expected ValueError for one clip")
    try:
        split_train_eval(["a", "a"], seed=0)
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate ids")


def test_build_stage1_rejects_t0_when_recipe_is_coc():
    try:
        build_pai_train_batch(
            None, "unused", "/tmp", recipe="coc", t0_us=5_100_000
        )
    except ValueError as exc:
        assert "t0_us" in str(exc)
    else:
        raise AssertionError("expected ValueError when recipe=coc and t0_us is set")


@pytest.mark.skipif(not REASONING_PATH.exists(), reason="PAI-CoC not on this machine")
def test_traj_pool_excludes_coc_and_default_eval_clip():
    table = list_local_traj_clips(exclude_coc=True)
    assert DEFAULT_EVAL_CLIP_ID not in table.index
    train, ev = select_non_coc_clips(
        REASONING_PATH.parents[1], n_clips=8, seed=0
    )
    assert len(train) == 4
    assert len(ev) == 4
    assert not (set(train) & set(ev))
    assert DEFAULT_EVAL_CLIP_ID not in train + ev
    train30, ev30 = select_non_coc_clips(
        REASONING_PATH.parents[1], n_clips=30, seed=0
    )
    assert len(train30) == 15
    assert len(ev30) == 15
    assert not (set(train30) & set(ev30))
    assert DEFAULT_EVAL_CLIP_ID not in train30 + ev30


def test_sft_stage1_small_help_mentions_no_coc():
    proc = subprocess.run(
        [sys.executable, "-m", "mlx_port.scripts.sft_stage1_small", "--help"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr)
    for token in (
        "no CoC",
        "50/50",
        "language QLoRA",
        "--lora-save-every",
        "--lora-save-dir",
        "--no-lora-save",
        "--epochs",
    ):
        if token not in proc.stdout:
            raise AssertionError(f"{token!r} missing from help:\n{proc.stdout}")


def test_sft_stage1_small_rejects_save_every_with_no_save():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage1_small",
            "--no-lora-save",
            "--lora-save-every",
            "5",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        raise AssertionError("expected non-zero exit for --no-lora-save + --lora-save-every")
    if "exclusive" not in (proc.stderr + proc.stdout):
        raise AssertionError(proc.stderr)


def test_resolve_train_steps_epochs_is_n_train_times_epochs():
    steps, epochs = resolve_train_steps(steps=None, epochs=2, n_train=4)
    assert steps == 8
    assert epochs == 2
    steps, epochs = resolve_train_steps(steps=None, epochs=None, n_train=4)
    assert steps == 10
    assert epochs is None
    steps, epochs = resolve_train_steps(steps=7, epochs=None, n_train=4)
    assert steps == 7
    assert epochs is None


def test_resolve_train_steps_rejects_both_and_bad_values():
    try:
        resolve_train_steps(steps=10, epochs=2, n_train=4)
    except ValueError as exc:
        assert "exclusive" in str(exc)
    else:
        raise AssertionError("expected exclusive error")
    try:
        resolve_train_steps(steps=None, epochs=0, n_train=4)
    except ValueError as exc:
        assert "epochs" in str(exc)
    else:
        raise AssertionError("expected epochs >= 1")


def test_sft_stage1_small_rejects_epochs_with_steps():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_port.scripts.sft_stage1_small",
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


def _write_pai(tmp_path, index: dict, coc: dict | None = None):
    root = tmp_path / "pai"
    (root / "reasoning").mkdir(parents=True)
    pd.DataFrame.from_dict(index, orient="index").to_parquet(root / "clip_index.parquet")
    if coc is not None:
        pd.DataFrame.from_dict(coc, orient="index").to_parquet(
            root / "reasoning" / "ood_reasoning.parquet"
        )
    return root


def test_select_non_coc_clips_rejects_default_eval_in_traj_pool(tmp_path):
    root = _write_pai(
        tmp_path,
        index={
            DEFAULT_EVAL_CLIP_ID: {"clip_is_valid": True, "chunk": 0},
            "other": {"clip_is_valid": True, "chunk": 1},
        },
        coc={"unrelated": {"split": "val", "event_cluster": "x", "events": []}},
    )
    try:
        select_non_coc_clips(root, n_clips=2, seed=0)
    except RuntimeError as exc:
        assert DEFAULT_EVAL_CLIP_ID in str(exc)
    else:
        raise AssertionError("expected RuntimeError when default CoC eval clip is in the traj pool")


def test_select_non_coc_clips_rejects_short_local_pool(tmp_path):
    root = _write_pai(
        tmp_path,
        index={
            "a": {"clip_is_valid": True, "chunk": 0},
            "b": {"clip_is_valid": True, "chunk": 1},
            "c": {"clip_is_valid": True, "chunk": 2},
        },
        coc={"unrelated": {"split": "val", "event_cluster": "x", "events": []}},
    )
    try:
        select_non_coc_clips(root, n_clips=8, seed=0)
    except RuntimeError as exc:
        assert "only 3 non-CoC" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when the local traj pool is smaller than n_clips")
