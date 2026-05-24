from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from PIL import Image

from .config import RAG_MAX_IMAGE_PX, RAG_MODEL

load_dotenv()

SYSTEM = (
    "You are answering the user's question using ONLY the provided retrieved "
    "images. Refer to images by their 1-based index (Image 1, Image 2, ...) "
    "when describing what you see. If the images do not support an answer, "
    "say so directly rather than guessing."
)


def _encode_image(path: Path, max_px: int = RAG_MAX_IMAGE_PX) -> tuple[str, str]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def answer(query: str, image_paths: list[Path]) -> str:
    """Send the retrieved images + query to Claude and return the answer text."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    client = Anthropic(api_key=api_key)

    content: list[dict] = []
    for i, p in enumerate(image_paths, start=1):
        data, media_type = _encode_image(p)
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            }
        )
        content.append({"type": "text", "text": f"(Image {i})"})

    content.append(
        {
            "type": "text",
            "text": (
                f"User question: {query}\n\n"
                "Answer the question using only the images above."
            ),
        }
    )

    resp = client.messages.create(
        model=RAG_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
