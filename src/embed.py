from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import open_clip
import torch
from PIL import Image

from .config import BATCH_SIZE, MODEL_NAME, PRETRAINED, get_device


_model = None
_preprocess = None
_tokenizer = None
_device = None


def _load() -> None:
    global _model, _preprocess, _tokenizer, _device
    if _model is not None:
        return
    _device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED
    )
    model.eval().to(_device)
    _model = model
    _preprocess = preprocess
    _tokenizer = open_clip.get_tokenizer(MODEL_NAME)


def _open_rgb(path: Path) -> Image.Image | None:
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception:
        return None


@torch.inference_mode()
def embed_images_batched(
    paths: Iterable[Path], batch_size: int = BATCH_SIZE
) -> Iterator[tuple[list[Path], np.ndarray]]:
    """Yield (paths, vectors) pairs in batches. Unreadable images are dropped.

    Pairs are emitted as batches complete, so the caller can stream-write to
    disk and keep the path↔vector alignment without buffering all of COCO.
    """
    _load()
    batch_imgs: list[torch.Tensor] = []
    batch_paths: list[Path] = []

    def flush() -> tuple[list[Path], np.ndarray] | None:
        if not batch_imgs:
            return None
        x = torch.stack(batch_imgs).to(_device)
        feats = _model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vecs = feats.cpu().float().numpy().astype(np.float32)
        paired = (list(batch_paths), vecs)
        batch_imgs.clear()
        batch_paths.clear()
        return paired

    for p in paths:
        img = _open_rgb(p)
        if img is None:
            continue
        batch_imgs.append(_preprocess(img))
        batch_paths.append(p)
        if len(batch_imgs) >= batch_size:
            result = flush()
            if result is not None:
                yield result

    result = flush()
    if result is not None:
        yield result


@torch.inference_mode()
def embed_text(query: str) -> np.ndarray:
    _load()
    tokens = _tokenizer([query]).to(_device)
    feats = _model.encode_text(tokens)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().float().numpy().astype(np.float32)
