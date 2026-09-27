"""
Generates a tiny synthetic dataset in ImageFolder layout (train/val/test, NORMAL/PNEUMONIA)
so CI can run train.py -> evaluate.py end-to-end without a real dataset.

This does not touch data/ - it writes to --out_dir (default: dummy_data), which is
disposable and gitignored. It never reads or modifies the real notebooks/training code.

Usage:
    python scripts/make_dummy_data.py
    python scripts/make_dummy_data.py --out_dir /tmp/dummy_data --per_class 6
"""

import argparse
import os

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny synthetic ImageFolder dataset for CI smoke tests.")
    parser.add_argument("--out_dir", type=str, default="dummy_data", help="Output root directory.")
    parser.add_argument("--per_class", type=int, default=6, help="Images per class per split.")
    parser.add_argument("--image_size", type=int, default=300, help="Generated image side length (pixels).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    splits = {"train": args.per_class, "val": max(2, args.per_class // 2), "test": max(2, args.per_class // 2)}
    classes = ["NORMAL", "PNEUMONIA"]

    total = 0
    for split, n in splits.items():
        for cls in classes:
            out_path = os.path.join(args.out_dir, split, cls)
            os.makedirs(out_path, exist_ok=True)
            for i in range(n):
                pixels = rng.integers(0, 255, (args.image_size, args.image_size), dtype=np.uint8)
                Image.fromarray(pixels, mode="L").convert("RGB").save(os.path.join(out_path, f"{i}.png"))
                total += 1

    print(f"[INFO] Wrote {total} synthetic images under '{args.out_dir}' "
          f"({', '.join(f'{k}={v}/class' for k, v in splits.items())}).")


if __name__ == "__main__":
    main()
