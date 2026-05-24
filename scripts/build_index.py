#!/usr/bin/env python
"""Embed every image under data/coco/ and write a FAISS index to index/."""

from __future__ import annotations

import os

# PyTorch and FAISS both ship libomp.dylib on macOS; without this the second
# one to load aborts the process. Must be set before importing torch/faiss.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BATCH_SIZE, INDEX_PATH  # noqa: E402
from src.data import iter_image_paths  # noqa: E402
from src.embed import embed_images_batched  # noqa: E402
from src.index import add, new_index, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild even if an index already exists.",
    )
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    if INDEX_PATH.exists() and not args.rebuild:
        print(f"Index already exists at {INDEX_PATH}. Pass --rebuild to overwrite.")
        return

    paths = list(iter_image_paths())
    if not paths:
        print("No images found under data/coco/. Run scripts/download_coco.py first.")
        sys.exit(1)
    print(f"Found {len(paths)} images. Embedding...")

    index = new_index()
    kept: list[Path] = []
    with tqdm(total=len(paths), unit="img") as bar:
        for batch_paths, vecs in embed_images_batched(paths, batch_size=args.batch_size):
            add(index, vecs)
            kept.extend(batch_paths)
            bar.update(len(batch_paths))

    save(index, kept)
    dropped = len(paths) - len(kept)
    print(
        f"Indexed {len(kept)} vectors -> {INDEX_PATH}"
        + (f" ({dropped} unreadable images skipped)" if dropped else "")
    )


if __name__ == "__main__":
    main()
