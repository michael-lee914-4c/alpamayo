"""Live-pack the T3.1 language tower and write ``mlx_lm4/`` for disk load.

Expert is not loaded. Use ``--force`` to replace an existing checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.models.quantize_lm import (
    LM4_CONFIG_NAME,
    LM4_WEIGHTS_NAME,
    resolve_lm4_dir,
)

CHECKPOINT = Path("/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B")


def main() -> None:
    parser = argparse.ArgumentParser(description="Save T3.1 packed language tower.")
    parser.add_argument("--alpamayo-path", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Packed dir. Default is {alpamayo}/mlx_lm4.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing mlx_lm4 pair and live-pack again.",
    )
    args = parser.parse_args()
    dest = Path(resolve_lm4_dir(str(args.alpamayo_path), str(args.out_dir) if args.out_dir else None))
    if args.force and dest.exists():
        for name in (LM4_WEIGHTS_NAME, LM4_CONFIG_NAME):
            p = dest / name
            if p.is_file():
                p.unlink()
        print(f"[QUANT] --force removed existing files in {dest}")
    print(f"[QUANT] saving T3.1 language tower → {dest}")
    AlpamayoR1MLX.from_pretrained(
        str(args.alpamayo_path),
        load_expert=False,
        dtype=mx.bfloat16,
        quantize_lm=True,
        lm4_path=str(dest),
    )
    weights = dest / LM4_WEIGHTS_NAME
    if not weights.is_file():
        raise RuntimeError(f"save did not write {weights}")
    print(f"[QUANT] done {weights} ({weights.stat().st_size / (1024 ** 3):.2f} GiB)")


if __name__ == "__main__":
    main()
