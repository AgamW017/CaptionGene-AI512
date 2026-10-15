"""
Download TACO dataset images.

TACO images are hosted on Flickr. The annotations.json file is fetched
from the official TACO GitHub repo and contains Flickr URLs for each image.

Usage:
    python scripts/download_taco.py --limit 50 --dry-run
    python scripts/download_taco.py --limit 50
    python scripts/download_taco.py          # download all (>1500 images)

Output layout:
    data/raw/taco/
        annotations.json
        data/
            batch_1/
                000001.jpg
                ...

License: CC BY 4.0
Source:  https://github.com/pedropro/TACO
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)
RAW_DIR = Path("data/raw/taco")
ANNOTATIONS_FILE = RAW_DIR / "annotations.json"

# Polite delay between requests (seconds)
REQUEST_DELAY = 0.5


def _fetch_annotations(dry_run: bool) -> dict | None:
    """Download or load annotations.json."""
    if ANNOTATIONS_FILE.exists():
        logger.info("annotations.json already present — skipping download.")
        with ANNOTATIONS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        logger.info("[dry-run] Would download: %s", TACO_ANNOTATIONS_URL)
        logger.info("[dry-run] → %s", ANNOTATIONS_FILE)
        return None

    logger.info("Downloading annotations.json from TACO GitHub…")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(TACO_ANNOTATIONS_URL, ANNOTATIONS_FILE)
    logger.info("Saved %s", ANNOTATIONS_FILE)
    with ANNOTATIONS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def _image_dest(img: dict) -> Path:
    """Return the local destination path for a TACO image dict."""
    file_name: str = img.get("file_name", "")
    # file_name is like "batch_1/000001.jpg" or "data/batch_1/000001.jpg"
    rel = file_name if file_name.startswith("data/") else f"data/{file_name}"
    return RAW_DIR / rel


def _download_image(img: dict, dry_run: bool) -> bool:
    """Download one image. Returns True on success or if already present."""
    url = img.get("flickr_url") or img.get("coco_url") or ""
    if not url:
        logger.warning("No URL for image %s — skipping.", img.get("id"))
        return False

    dest = _image_dest(img)

    if dry_run:
        logger.info("[dry-run] Would download %s → %s", url, dest)
        return True

    if dest.exists():
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TACO dataset images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of images to download (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without doing it.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help="Seconds to wait between image downloads.",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    if args.dry_run:
        logger.info("=== DRY RUN — nothing will be written ===")

    coco = _fetch_annotations(args.dry_run)
    if coco is None:
        logger.info("[dry-run] Cannot proceed without annotations — exiting.")
        return

    images = coco.get("images", [])
    if limit is not None:
        images = images[:limit]

    total = len(images)
    logger.info("Planning to download %d images to %s", total, RAW_DIR)

    ok = failed = skipped = 0
    for i, img in enumerate(images, 1):
        dest = _image_dest(img)
        if not args.dry_run and dest.exists():
            skipped += 1
            continue

        success = _download_image(img, args.dry_run)
        if success:
            ok += 1
        else:
            failed += 1

        if not args.dry_run and i % 10 == 0:
            logger.info(
                "Progress: %d/%d (ok=%d failed=%d skipped=%d)",
                i,
                total,
                ok,
                failed,
                skipped,
            )
            time.sleep(args.delay)

    logger.info(
        "Done. ok=%d failed=%d skipped=%d  Output: %s",
        ok,
        failed,
        skipped,
        RAW_DIR,
    )
    if failed > 0:
        logger.warning("%d images failed — re-run to retry.", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
