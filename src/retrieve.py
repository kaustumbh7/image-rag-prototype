from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import DEFAULT_TOP_K
from .embed import embed_text
from .index import load


@lru_cache(maxsize=1)
def _resources():
    return load()


def search(query: str, k: int = DEFAULT_TOP_K) -> list[tuple[Path, float]]:
    index, paths = _resources()
    qv = embed_text(query)
    scores, ids = index.search(qv, k)
    results: list[tuple[Path, float]] = []
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
        if idx < 0:
            continue
        results.append((paths[idx], float(score)))
    return results
