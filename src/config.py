from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "coco"
INDEX_DIR = ROOT / "index"
INDEX_PATH = INDEX_DIR / "coco.faiss"
PATHS_PATH = INDEX_DIR / "paths.txt"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512

DEFAULT_TOP_K = 5
BATCH_SIZE = 64

RAG_MODEL = "claude-sonnet-4-6"
RAG_MAX_IMAGE_PX = 1024


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
