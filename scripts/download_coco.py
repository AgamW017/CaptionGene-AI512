"""
Download COCO 2017 subset — waste-relevant classes only.

Strategy: download the full annotations file (small, ~241 MB for train,
~1 MB for val), then download only the images that contain at least one
waste-relevant annotation. This avoids downloading the full 18 GB image set.

Usage:
    python scripts/download_coco.py --dry-run
    python scripts/download_coco.py --limit 50          # 50 images from val2017
    python scripts/download_coco.py --limit 50 --split train
    python scripts/download_coco.py --limit 0            # all waste images

Output layout:
    data/raw/coco/
        annotations/
            instances_val2017.json
            instances_train2017.json   (if --split train)
        images/
            val2017/
                *.jpg
            train2017/
                *.jpg

License: CC BY 4.0
Source:  https://cocodataset.org/
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

RAW_DIR = Path("data/raw/coco")

ANNOTATION_URLS = {
    "val": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
# Both val and train annotations are in the same zip.
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP = RAW_DIR / "annotations_trainval2017.zip"

COCO_IMAGE_BASE = "http://images.cocodataset.org"

# Waste-relevant COCO class names (must match taxonomy.yaml coco section)
WASTE_CLASSES: frozenset[str] = frozenset(
    [
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "backpack",
        "handbag",
        "suitcase",
        "cell phone",
        "book",
        "scissors",
    ]
)

REQUEST_DELAY = 0.3


def _get_annotations(split: str, dry_run: bool) -> dict | None:
    """Ensure annotations JSON is available; return parsed dict."""
    ann_file = RAW_DIR / "annotations" / f"instances_{split}2017.json"

    if ann_file.exists():
        logger.info("Annotations already present: %s", ann_file)
        with ann_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    if dry_run:
        logger.info("[dry-run] Would download annotations zip: %s", ANNOTATIONS_ZIP_URL)
        logger.info("[dry-run] → %s", ann_file)
        return None

    import zipfile

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "annotations").mkdir(parents=True, exist_ok=True)

    if not ANNOTATIONS_ZIP.exists():
        logger.info("Downloading COCO annotations zip (~241 MB)…\n  %s", ANNOTATIONS_ZIP_URL)
        logger.info("This may take a few minutes.")
        urllib.request.urlretrieve(ANNOTATIONS_ZIP_URL, ANNOTATIONS_ZIP)
        logger.info("Downloaded annotations zip.")

    logger.info("Extracting %s from zip…", ann_file.name)
    with zipfile.ZipFile(ANNOTATIONS_ZIP, "r") as zf:
        target = f"annotations/instances_{split}2017.json"
        if target in zf.namelist():
            zf.extract(target, RAW_DIR)
        else:
            logger.error("Expected %s in zip but not found.", target)
            return None

    with ann_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def _waste_images(coco: dict) -> list[dict]:
    """Return list of image dicts that contain at least one waste annotation."""
    cat_by_id = {c["id"]: c["name"] for c in coco.get("categories", [])}
    waste_ids = {cid for cid, name in cat_by_id.items() if name in WASTE_CLASSES}
    waste_image_ids = {
        ann["image_id"] for ann in coco.get("annotations", []) if ann["category_id"] in waste_ids
    }
    return [img for img in coco.get("images", []) if img["id"] in waste_image_ids]


def _download_image(img: dict, split: str, dry_run: bool) -> bool:
    file_name: str = img.get("file_name", "")
    dest = RAW_DIR / "images" / f"{split}2017" / file_name
    url = f"{COCO_IMAGE_BASE}/{split}2017/{file_name}"

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
        logger.warning("Failed: %s — %s", url, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download COCO 2017 waste-class subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=["val", "train"],
        default="val",
        help="Which COCO split to use. Start with 'val' (smaller, ~5000 waste images).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max images to download (0 = all waste images in split).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing.",
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

    coco = _get_annotations(args.split, args.dry_run)
    if coco is None:
        if args.dry_run:
            logger.info(
                "[dry-run] With real data: would download ~%d waste images from %s2017.",
                limit or "all",
                args.split,
            )
        return

    waste_imgs = _waste_images(coco)
    logger.info(
        "Found %d waste-class images in %s2017 (waste classes: %d categories)",
        len(waste_imgs),
        args.split,
        len(WASTE_CLASSES),
    )

    if limit is not None:
        waste_imgs = waste_imgs[:limit]

    ok = failed = skipped = 0
    for i, img in enumerate(waste_imgs, 1):
        dest = RAW_DIR / "images" / f"{args.split}2017" / img.get("file_name", "")
        if not args.dry_run and dest.exists():
            skipped += 1
            continue

        success = _download_image(img, args.split, args.dry_run)
        if success:
            ok += 1
        else:
            failed += 1

        if not args.dry_run and i % 10 == 0:
            logger.info("Progress: %d/%d", i, len(waste_imgs))
            time.sleep(args.delay)

    logger.info(
        "Done. ok=%d failed=%d skipped=%d  Output: %s",
        ok,
        failed,
        skipped,
        RAW_DIR,
    )
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
