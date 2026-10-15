"""
Download ZeroWaste-F dataset.

ZeroWaste-F is hosted on Google Drive. This script downloads it via
the gdown library (pip install gdown).

Usage:
    python scripts/download_zerowaste.py --dry-run
    python scripts/download_zerowaste.py --limit 50
    python scripts/download_zerowaste.py          # full dataset (~1 GB)

The full dataset is a single zip (~1 GB). With --limit, the zip is still
downloaded but only the first N images are copied to the output directory,
keeping disk usage low during development.

Output layout:
    data/raw/zerowaste/
        zerowaste-f/
            train/
                data/   (images)
                labels.json
            val/
                data/
                labels.json

License: Research use only — non-commercial.
Source:  https://github.com/dbash/zerowaste
Paper:   Bashkirova et al. CVPR 2022, arXiv:2106.02740
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Google Drive file ID for ZeroWaste-F
GDRIVE_FILE_ID = "1o7GJJ4KCAA3bD_VBdNqrRTRBOXaknRXr"
RAW_DIR = Path("data/raw/zerowaste")
ZIP_PATH = RAW_DIR / "zerowaste-f.zip"
EXTRACT_DIR = RAW_DIR


def _check_gdown() -> bool:
    try:
        import gdown  # noqa: F401

        return True
    except ImportError:
        return False


def _download_zip(dry_run: bool) -> bool:
    """Download the ZeroWaste-F zip from Google Drive."""
    if ZIP_PATH.exists():
        logger.info("Zip already present at %s — skipping download.", ZIP_PATH)
        return True

    if dry_run:
        logger.info("[dry-run] Would download Google Drive file ID: %s", GDRIVE_FILE_ID)
        logger.info("[dry-run] → %s", ZIP_PATH)
        return True

    if not _check_gdown():
        logger.error(
            "gdown is not installed. Install it with:\n"
            "  .venv/bin/pip install gdown\n"
            "Then re-run this script."
        )
        return False

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ZeroWaste-F from Google Drive (file ID: %s)…", GDRIVE_FILE_ID)
    logger.info("This is ~1 GB — may take several minutes.")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
            "-O",
            str(ZIP_PATH),
        ],
        check=False,
    )
    if result.returncode != 0:
        logger.error("gdown failed. Check your network or try downloading manually:")
        logger.error("  https://drive.google.com/file/d/%s/view", GDRIVE_FILE_ID)
        logger.error("  Save to: %s", ZIP_PATH)
        return False

    logger.info("Download complete: %s", ZIP_PATH)
    return True


def _extract(limit: int | None, dry_run: bool) -> None:
    """Extract zip, optionally limiting to the first N images per split."""
    if dry_run:
        logger.info("[dry-run] Would extract %s → %s", ZIP_PATH, EXTRACT_DIR)
        if limit:
            logger.info("[dry-run] Would keep only first %d images per split.", limit)
        return

    if not ZIP_PATH.exists():
        logger.error("Zip not found at %s — run download step first.", ZIP_PATH)
        return

    logger.info("Extracting %s…", ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        members = zf.namelist()

        if limit is None:
            zf.extractall(EXTRACT_DIR)
            logger.info("Extracted all %d files.", len(members))
            return

        # Selective extraction: always keep labels.json files; limit images
        image_counts: dict[str, int] = {}  # split_name -> count
        for member in members:
            path = Path(member)
            parts = path.parts

            # Always extract non-image files (labels.json, etc.)
            is_image = (
                len(parts) >= 2
                and parts[-1].lower().endswith((".jpg", ".jpeg", ".png"))
                and "data" in parts
            )
            if not is_image:
                zf.extract(member, EXTRACT_DIR)
                continue

            # Determine split from path (zerowaste-f/train/data/*.png)
            split_name = parts[1] if len(parts) > 2 else "train"
            count = image_counts.get(split_name, 0)
            if count < limit:
                zf.extract(member, EXTRACT_DIR)
                image_counts[split_name] = count + 1

        logger.info(
            "Extracted with limit=%d: %s",
            limit,
            {k: v for k, v in image_counts.items()},
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ZeroWaste-F dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max images per split to extract (0 = all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing.",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    if args.dry_run:
        logger.info("=== DRY RUN — nothing will be written ===")

    ok = _download_zip(args.dry_run)
    if not ok:
        sys.exit(1)

    _extract(limit, args.dry_run)
    logger.info("ZeroWaste output: %s", RAW_DIR)


if __name__ == "__main__":
    main()
