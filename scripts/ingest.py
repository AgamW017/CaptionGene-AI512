"""
Ingest pipeline: raw data → manifest → splits → SQLite DB.

Runs all three loaders (or a subset), merges records into a manifest,
assigns canonical image_ids, generates train/val/test splits, and
writes everything to the database.

Usage:
    # Dry run — shows what would be ingested
    python scripts/ingest.py --dry-run

    # Ingest all sources (must have downloaded data first)
    python scripts/ingest.py

    # Ingest one source only, limit to 50 images
    python scripts/ingest.py --source taco --limit 50

    # Ingest all sources, 50 images each
    python scripts/ingest.py --limit 50

Output:
    data/db/wastecap.db      (SQLite, gitignored)
    data/processed/splits.json  (committed to git)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make sure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=os.environ.get("WASTECAP_LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_TACO = Path("data/raw/taco")
RAW_ZW = Path("data/raw/zerowaste")
RAW_COCO = Path("data/raw/coco")
SPLITS_OUT = Path("data/processed/splits.json")


def _load_source(
    source: str,
    limit: int | None,
    dry_run: bool,
) -> list:
    """Load records from a single source. Returns [] on failure."""
    from wastecap.data.coco_subset import load_coco_subset
    from wastecap.data.taco import load_taco
    from wastecap.data.zerowaste import load_zerowaste

    loaders = {
        "taco": (load_taco, RAW_TACO),
        "zerowaste": (load_zerowaste, RAW_ZW),
        "coco": (load_coco_subset, RAW_COCO),
    }

    fn, raw_dir = loaders[source]

    if not raw_dir.exists():
        logger.warning(
            "Raw data directory %s not found — skipping %s. "
            "Run: python scripts/download_%s.py --limit %s",
            raw_dir,
            source,
            source,
            limit or 50,
        )
        return []

    try:
        records = fn(raw_dir, limit=limit)
        logger.info("Loaded %d records from %s", len(records), source)
        return records
    except FileNotFoundError as exc:
        logger.warning("Skipping %s: %s", source, exc)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest raw data into the WasteCap database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["taco", "zerowaste", "coco", "all"],
        default="all",
        help="Source to ingest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max images per source (None = all available).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate records but do not write to DB.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
    )
    args = parser.parse_args()

    sources = ["taco", "zerowaste", "coco"] if args.source == "all" else [args.source]

    # 1. Load
    all_records = []
    for src in sources:
        recs = _load_source(src, args.limit, args.dry_run)
        all_records.extend(recs)

    if not all_records:
        logger.error(
            "No records loaded. Download data first:\n"
            "  python scripts/download_taco.py --limit 50\n"
            "  python scripts/download_zerowaste.py --limit 50\n"
            "  python scripts/download_coco.py --limit 50"
        )
        sys.exit(1)

    logger.info("Total records before manifest: %d", len(all_records))

    # 2. Manifest: dedup + quality filter + reindex
    from wastecap.data.manifest import build_manifest

    records = build_manifest(
        all_records,
        deduplicate=True,
        quality_filter=True,
        reindex=True,
    )
    logger.info("After manifest: %d records", len(records))

    # 3. Splits
    from wastecap.data.splits import generate_splits

    splits = generate_splits(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_path=SPLITS_OUT if not args.dry_run else None,
    )
    logger.info(
        "Splits: train=%d val=%d test=%d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    if args.dry_run:
        logger.info("[dry-run] Would write %d records to DB. Exiting.", len(records))
        for src in sources:
            src_recs = [r for r in records if r.source.value == src]
            logger.info("  %s: %d records", src, len(src_recs))
        return

    # 4. Write to DB
    from wastecap.store import init_db, upsert_record

    db_path = init_db()
    written = 0
    for rec in records:
        upsert_record(rec, db_path)
        written += 1

    logger.info("Wrote %d records to %s", written, db_path)
    logger.info("splits.json: %s", SPLITS_OUT)


if __name__ == "__main__":
    main()
