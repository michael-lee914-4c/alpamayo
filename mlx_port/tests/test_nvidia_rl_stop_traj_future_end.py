"""RL vLLM rollout stops on ``traj_future_end``, not ``traj_future_start``.

Weight-free. Complementary to open #32 (SFT infer EOS is
``<|traj_future_start|>``) and #26 (``StopAfterEOS`` one-token-after).
Cosmos-RL overwrites ``sampling_params.stop_token_ids`` with the *end*
id so the completion includes the waypoint span. Stopping on start
would yield an empty rollout and a zero reward.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_ROLLOUT = _REPO / "finetune" / "rl" / "models" / "reasoning_vla" / "rollout.py"
_SFT_INFER = _REPO / "src" / "alpamayo_r1" / "models" / "alpamayo_r1.py"
_BASE = _REPO / "src" / "alpamayo_r1" / "models" / "base_model.py"


def test_source_rl_overrides_stop_to_future_end():
    rollout = _ROLLOUT.read_text()
    infer = _SFT_INFER.read_text()
    base = _BASE.read_text()
    assert 'SPECIAL_TOKENS["traj_future_end"]' in rollout
    assert "sampling_params.stop_token_ids = [self._traj_future_end_token_id]" in rollout
    assert 'to_special_token("traj_future_start")' in infer
    assert '"future_end": "<|traj_future_end|>"' in base
    assert '"future_start": "<|traj_future_start|>"' in base


def test_stop_list_is_end_id_not_start():
    class _Params:
        stop_token_ids = [1, 2]

    start_id, end_id = 11, 22
    sampling_params = _Params()
    sampling_params.stop_token_ids = [end_id]
    assert sampling_params.stop_token_ids == [end_id]
    assert start_id not in sampling_params.stop_token_ids
