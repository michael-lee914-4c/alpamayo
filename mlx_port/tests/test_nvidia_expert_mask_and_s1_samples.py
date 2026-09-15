"""NVIDIA ExpertLogitsProcessor and Stage-1 sample count.

Weight-free. Complementary to open #30 (infer
``num_return_sequences = num_traj_samples``) and the MLX
``ExpertLogitsProcessor`` test — NVIDIA masks
``scores[:, offset:offset+vocab]`` to ``-inf``, Stage-1 discrete
sample sets ``num_return_sequences = samples * sets``, and infer
stops on ``<|traj_future_start|>`` (not ``im_end``). Swapping those
would leak traj bins into CoC or request the wrong generate width.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_ALPAMAYO = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"
_SFT_BASE = _REPO / "finetune" / "sft" / "models" / "sft_base_model.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _load_expert_processor():
    tree = ast.parse(_ALPAMAYO.read_text())
    cls_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ExpertLogitsProcessor":
            cls_node = node
            break
    if cls_node is None:
        pytest.skip("ExpertLogitsProcessor missing")
    # Drop the LogitsProcessor base so we do not need transformers.
    cls_node.bases = []
    mod = ast.Module(body=[cls_node], type_ignores=[])
    ast.fix_missing_locations(mod)
    ns: dict[str, object] = {"__name__": "extracted", "torch": pytest.importorskip("torch")}
    exec(compile(mod, str(_ALPAMAYO), "exec"), ns)
    return ns["ExpertLogitsProcessor"]


def test_expert_logits_processor_masks_traj_slice_only(torch):
    ExpertLogitsProcessor = _load_expert_processor()
    proc = ExpertLogitsProcessor(traj_token_offset=100, traj_vocab_size=50)
    scores = torch.zeros(2, 200)
    scores[0, 99] = 1.5
    scores[0, 150] = 2.5
    out = proc(torch.zeros(2, 3, dtype=torch.long), scores)
    assert torch.all(torch.isinf(out[0, 100:150]) & (out[0, 100:150] < 0))
    assert float(out[0, 99].item()) == 1.5
    assert float(out[0, 150].item()) == 2.5
    assert torch.all(out[1, 100:150] == float("-inf"))
    src = _ALPAMAYO.read_text()
    assert "scores[:, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size]" in src


def test_stage1_num_return_sequences_is_samples_times_sets():
    sft = _SFT_BASE.read_text()
    infer = _ALPAMAYO.read_text()
    assert "n_samples_total = num_traj_samples * num_traj_sets" in sft
    assert "generation_config.num_return_sequences = n_samples_total" in sft
    # Infer (#30) stays samples-only — Stage-1 must not copy that.
    assert "generation_config.num_return_sequences = num_traj_samples" in infer
    assert "generation_config.num_return_sequences = n_samples_total" not in infer
    num_traj_samples, num_traj_sets = 6, 2
    n_samples_total = num_traj_samples * num_traj_sets
    assert n_samples_total == 12
    assert n_samples_total != num_traj_samples


def test_infer_eos_is_traj_future_start_not_im_end():
    src = _ALPAMAYO.read_text()
    assert 'to_special_token("traj_future_start")' in src
    assert "StopAfterEOS(eos_token_id=eos_token_id)" in src
    assert 'to_special_token("im_end")' not in src
    # Helper used to wrap the name.
    token_utils = (_REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py").read_text()
    assert 'return "<|" + token + "|>"' in token_utils
    assert "<|traj_future_start|>" == "<|" + "traj_future_start" + "|>"
