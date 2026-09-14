#!/usr/bin/env python
"""Download the Le2i fall-detection dataset.

Source: Kaggle dataset "tuyenldvn/falldataset-imvia" (a mirror of the
original Le2i UMR6306 release: Coffee_room_01/02, Home_01/02, Lecture_room,
Office - ~17GB of .avi video across 190 recordings).

Requires Kaggle API credentials: place kaggle.json at ~/.kaggle/kaggle.json
(or set KAGGLE_USERNAME / KAGGLE_KEY env vars) before running. In Colab:

    from google.colab import files
    files.upload()  # upload kaggle.json
    !mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_le2i.py --out_dir data/raw
    python scripts/download_le2i.py --out_dir data/raw --annotations_only
"""
import argparse
import urllib.parse
import zipfile
from pathlib import Path

DATASET = "tuyenldvn/falldataset-imvia"


def _download_one_file(api, remote_name: str, dest_dir: Path) -> None:
    """Download a single dataset file, tolerating two Kaggle API quirks
    seen in practice: it sometimes wraps the file in a same-named .zip,
    and it sometimes saves the local filename percent-encoded (e.g.
    "video%20(1).txt" instead of "video (1).txt").
    """
    before = set(dest_dir.iterdir()) if dest_dir.exists() else set()
    dest_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_file(DATASET, remote_name, path=str(dest_dir))
    new_files = [p for p in dest_dir.iterdir() if p not in before]

    for p in new_files:
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(dest_dir)
            p.unlink()

    for p in list(dest_dir.iterdir()):
        if "%" in p.name:
            p.rename(p.with_name(urllib.parse.unquote(p.name)))


def download_full(out_dir: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    out_dir.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET, path=str(out_dir), unzip=True, quiet=False)


def download_annotations_only(out_dir: Path) -> None:
    """Fetch just the per-frame ground-truth .txt files (a few hundred KB
    total) plus nothing else. Useful for developing/testing the inventory
    and windowing logic without pulling ~17GB of video.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    all_files = []
    token = None
    while True:
        resp = api.dataset_list_files(DATASET, page_token=token)
        all_files.extend(resp.files)
        token = getattr(resp, "next_page_token", None) or getattr(resp, "nextPageToken", None)
        if not token:
            break

    annotation_files = [f for f in all_files if f.name.endswith(".txt") and "Annotation" in f.name]
    print(f"downloading {len(annotation_files)} annotation files to {out_dir} ...")
    for f in annotation_files:
        dest = out_dir / f.name
        if dest.exists():
            continue
        _download_one_file(api, f.name, dest.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument(
        "--annotations_only",
        action="store_true",
        help="download only the ground-truth .txt files, skip all video",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.annotations_only:
        download_annotations_only(out_dir)
    else:
        download_full(out_dir)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
