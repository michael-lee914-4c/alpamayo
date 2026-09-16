"""NVIDIA extract contracts when a start marker is missing.

Weight-free. Complementary to open #24 (missing-*end* → empty / last
start). NVIDIA ``extract_between_special_tokens`` returns the prefix
before the first end when start is absent — not ``""`` and not the
whole string (the MLX port's fallback). ``extract_traj_tokens`` sets
``start_positions = -1``, so the prefix before end is scooped as
traj bins instead of staying zeros.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TOKEN_UTILS = _REPO / "src" / "alpamayo_r1" / "models" / "token_utils.py"
_MLX_TOKEN_UTILS = _REPO / "mlx_port" / "models" / "token_utils_mlx.py"


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _nv_extract_between(decoded_batch: list[str], token: str) -> list[str]:
    start_token = "<|" + token + "_start|>"
    end_token = "<|" + token + "_end|>"
    out: list[str] = []
    for s in decoded_batch:
        before_end, sep, _ = s.partition(end_token)
        if not sep:
            out.append("")
            continue
        i = before_end.rfind(start_token)
        if i != -1:
            out.append(before_end[i + len(start_token) :].strip())
        else:
            out.append(before_end.strip())
    return out


def test_extract_between_missing_end_is_empty():
    src = _TOKEN_UTILS.read_text()
    assert "before_end, sep, _ = s.partition(end_token)" in src
    assert "if not sep:" in src
    assert 'apnd("")' in src

    got = _nv_extract_between(["user text with no markers"], "cot")
    assert got == [""]


def test_extract_between_missing_start_returns_prefix_not_whole_text():
    src = _TOKEN_UTILS.read_text()
    assert "i = before_end.rfind(start_token)" in src
    assert "apnd(before_end.strip())" in src

    text = "SYSTEM leak <|cot_end|> trailing should drop"
    got = _nv_extract_between([text], "cot")
    assert got == ["SYSTEM leak"]

    # MLX returns the whole stripped string when either marker is missing.
    mlx_src = _MLX_TOKEN_UTILS.read_text()
    assert "results.append(text.strip())" in mlx_src
    mlx_fallback = text.strip()
    assert mlx_fallback != got[0]
    assert "trailing should drop" in mlx_fallback


def test_extract_between_uses_last_start_via_rfind():
    text = "<|cot_start|>first<|cot_start|>second <|cot_end|>tail"
    got = _nv_extract_between([text], "cot")
    assert got == ["second"]
    first_start = text.split("<|cot_start|>", 1)[-1].split("<|cot_end|>", 1)[0].strip()
    assert first_start.startswith("first")


def test_extract_traj_tokens_missing_start_scoops_prefix(torch):
    src = _TOKEN_UTILS.read_text()
    assert 'torch.full((batch_size,), -1, dtype=torch.long, device=device)' in src
    assert "range_tensor > start_positions.unsqueeze(1)" in src

    start_id, end_id, band0 = 100, 101, 50
    # No start. End at col 4. Prefix [10, 11, 12, 13] is treated as traj.
    tokens = torch.tensor([[10, 11, 12, 13, end_id, 99]], dtype=torch.long)
    seq_len = tokens.shape[1]
    end_mask = tokens == end_id
    end_positions = torch.where(
        end_mask.any(dim=1),
        end_mask.int().argmax(dim=1),
        torch.full((1,), seq_len, dtype=torch.long),
    )
    start_positions = torch.full((1,), -1, dtype=torch.long)
    range_tensor = torch.arange(seq_len).unsqueeze(0)
    valid_mask = (range_tensor > start_positions.unsqueeze(1)) & (
        range_tensor < end_positions.unsqueeze(1)
    )
    assert valid_mask.tolist() == [[True, True, True, True, False, False]]

    n_out = 4
    traj = torch.zeros((1, n_out), dtype=torch.long)
    cumsum_indices = torch.cumsum(valid_mask.int(), dim=1) - 1
    output_mask = valid_mask & (cumsum_indices < n_out)
    extracted = torch.where(valid_mask, tokens, torch.zeros_like(tokens))
    batch_ids = torch.zeros(int(output_mask.sum()), dtype=torch.long)
    pos = cumsum_indices[output_mask]
    values = torch.clamp(extracted[output_mask] - band0, min=0, max=9)
    traj[batch_ids, pos] = values
    # 10-50…13-50 clamp to 0 — garbage prefix, not an empty skip.
    assert traj.tolist() == [[0, 0, 0, 0]]
    assert int(valid_mask.sum()) == 4
