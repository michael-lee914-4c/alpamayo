"""Skip weight / SSD tests when those assets are not on the machine (CI)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_QWEN = _REPO_ROOT / "pre-trained" / "Qwen3-VL-8B-Instruct"
_ALPAMAYO = _REPO_ROOT / "pre-trained" / "Alpamayo-R1-10B"
_PAI_COC = Path("/Volumes/MicronSSD/pai_coc")

collect_ignore: list[str] = []

try:
    import physical_ai_av  # noqa: F401
except ImportError:
    collect_ignore += [
        "test_end_to_end_inference.py",
        "test_single_image_coc.py",
        "test_data_loading.py",
    ]

if os.environ.get("ALPAMAYO_CI_NO_WEIGHTS"):
    for name in (
        "test_end_to_end_inference.py",
        "test_single_image_coc.py",
        "test_data_loading.py",
        "test_vlm_weight_loading.py",
        "test_expert_weight_loading.py",
        "test_vlm_vision_encoder.py",
        "test_conv3d_layout.py",
        "test_tokenizer_id_parity.py",
    ):
        if name not in collect_ignore:
            collect_ignore.append(name)

if not _ALPAMAYO.exists():
    for name in (
        "test_conv3d_layout.py",
        "test_tokenizer_id_parity.py",
        "test_vlm_weight_loading.py",
        "test_expert_weight_loading.py",
    ):
        if name not in collect_ignore:
            collect_ignore.append(name)

if not _QWEN.exists():
    for name in ("test_vlm_vision_encoder.py", "test_tokenizer_id_parity.py"):
        if name not in collect_ignore:
            collect_ignore.append(name)

if not _PAI_COC.exists() and "test_data_loading.py" not in collect_ignore:
    collect_ignore.append("test_data_loading.py")
