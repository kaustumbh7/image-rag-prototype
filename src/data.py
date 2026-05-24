from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .config import DATA_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def iter_image_paths(root: Path = DATA_DIR) -> Iterator[Path]:
    if not root.exists():
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            yield p
