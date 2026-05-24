from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from .config import EMBED_DIM, INDEX_DIR, INDEX_PATH, PATHS_PATH


def new_index() -> faiss.Index:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Inner product over L2-normalized vectors == cosine similarity.
    return faiss.IndexFlatIP(EMBED_DIM)


def save(index: faiss.Index, paths: list[Path]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    PATHS_PATH.write_text("\n".join(str(p) for p in paths))


def load() -> tuple[faiss.Index, list[Path]]:
    if not INDEX_PATH.exists() or not PATHS_PATH.exists():
        raise FileNotFoundError(
            f"Index not found at {INDEX_PATH}. Run scripts/build_index.py first."
        )
    index = faiss.read_index(str(INDEX_PATH))
    paths = [Path(line) for line in PATHS_PATH.read_text().splitlines() if line]
    if index.ntotal != len(paths):
        raise RuntimeError(
            f"Index/paths mismatch: {index.ntotal} vectors vs {len(paths)} paths"
        )
    return index, paths


def add(index: faiss.Index, vectors: np.ndarray) -> None:
    index.add(vectors)
