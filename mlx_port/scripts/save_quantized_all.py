"""Live-pack all4 (full VLM + diffusion expert) and write ``mlx_all4/``.

Action-in/out stay dense. Use ``--force`` to replace an existing trio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.models.quantize_all import (
    ALL4_CONFIG_NAME,
    ALL4_EXPERT_WEIGHTS_NAME,
    ALL4_VLM_WEIGHTS_NAME,
    resolve_all4_dir,
)

CHECKPOINT = Path("/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save all4 packed VLM + expert.")
    parser.add_argument("--alpamayo-path", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Packed dir. Default is {alpamayo}/mlx_all4.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing mlx_all4 trio and live-pack again.",
    )
    args = parser.parse_args()
    dest = Path(
        resolve_all4_dir(str(args.alpamayo_path), str(args.out_dir) if args.out_dir else None)
    )
    if args.force and dest.exists():
        for name in (ALL4_VLM_WEIGHTS_NAME, ALL4_EXPERT_WEIGHTS_NAME, ALL4_CONFIG_NAME):
            p = dest / name
            if p.is_file():
                p.unlink()
        print(f"[QUANT] --force removed existing files in {dest}")
    print(f"[QUANT] saving all4 VLM + expert → {dest}")
    AlpamayoR1MLX.from_pretrained(
        str(args.alpamayo_path),
        load_expert=True,
        dtype=mx.bfloat16,
        quantize_all=True,
        all4_path=str(dest),
    )
    vlm = dest / ALL4_VLM_WEIGHTS_NAME
    expert = dest / ALL4_EXPERT_WEIGHTS_NAME
    cfg = dest / ALL4_CONFIG_NAME
    for path in (vlm, expert, cfg):
        if not path.is_file():
            raise RuntimeError(f"save did not write {path}")
    print(
        f"[QUANT] done {vlm} ({vlm.stat().st_size / (1024 ** 3):.2f} GiB) "
        f"+ {expert} ({expert.stat().st_size / (1024 ** 3):.2f} GiB)"
    )


if __name__ == "__main__":
    main()
