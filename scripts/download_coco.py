#!/usr/bin/env python
"""Download a COCO 2017 image split into data/coco/<split>/."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR  # noqa: E402

URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[skip] {dest.name} already downloaded")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def extract(zip_path: Path, out_dir: Path) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[skip] {out_dir} already extracted")
        return
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        for m in tqdm(members, desc=f"extracting {zip_path.name}"):
            zf.extract(m, out_dir.parent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split",
        choices=list(URLS),
        default="val2017",
        help="Which COCO 2017 split to download (default: val2017, ~1GB)",
    )
    args = ap.parse_args()

    url = URLS[args.split]
    zip_path = DATA_DIR / f"{args.split}.zip"
    out_dir = DATA_DIR / args.split

    download(url, zip_path)
    extract(zip_path, out_dir)
    print(f"Done. Images in {out_dir}")


if __name__ == "__main__":
    main()
