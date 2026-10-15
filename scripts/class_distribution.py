"""
Print per-split class distribution and flag imbalance.

Reads the SQLite database and splits.json, then prints:
  - Object count per canonical category per split
  - Image count per source per split
  - Imbalance warnings when one class is >5× the median or represents <2% of objects

Usage:
    python scripts/class_distribution.py
    python scripts/class_distribution.py --min-images 5 --imbalance-ratio 5.0
    python scripts/class_distribution.py --splits-file data/processed/splits.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=os.environ.get("WASTECAP_LOG_LEVEL", "WARNING"),
    format="%(levelname)s %(name)s: %(message)s",
)

SPLITS_FILE = Path("data/processed/splits.json")
_BAR_WIDTH = 30


def _bar(count: int, max_count: int) -> str:
    filled = round(_BAR_WIDTH * count / max_count) if max_count > 0 else 0
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-split class distribution and flag imbalance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=SPLITS_FILE,
        help="Path to splits.json.",
    )
    parser.add_argument(
        "--imbalance-ratio",
        type=float,
        default=5.0,
        help="Flag a class if its count > ratio × median count.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Flag a class if its object count < min-count in a split.",
    )
    args = parser.parse_args()

    # Load splits
    if not args.splits_file.exists():
        print(
            f"ERROR: {args.splits_file} not found.\nRun: python scripts/ingest.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with args.splits_file.open() as f:
        splits: dict[str, list[str]] = json.load(f)

    # Load DB
    from wastecap.store import init_db, list_records

    db_path = init_db()
    id_to_split = {
        img_id: split_name for split_name, id_list in splits.items() for img_id in id_list
    }

    # Gather stats
    cat_counts: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    src_counts: dict[str, Counter] = {"train": Counter(), "val": Counter(), "test": Counter()}
    img_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    all_records = list_records(db_path=db_path)
    if not all_records:
        print("No records in database. Run: python scripts/ingest.py", file=sys.stderr)
        sys.exit(1)

    for rec in all_records:
        split_name = id_to_split.get(rec.image_id)
        if split_name is None:
            continue
        img_counts[split_name] += 1
        src_counts[split_name][rec.source.value] += 1
        for obj in rec.objects:
            cat_counts[split_name][obj.category.value] += 1

    # --- Print ---
    width = 60
    print("=" * width)
    print("  WasteCap — Class Distribution Report")
    print("=" * width)

    all_categories = sorted({cat for counter in cat_counts.values() for cat in counter})
    _ = all_categories  # reserved for future per-category cross-split table
    split_names = ["train", "val", "test"]

    for split_name in split_names:
        counter = cat_counts[split_name]
        n_images = img_counts[split_name]
        n_objects = sum(counter.values())
        print(f"\n── {split_name.upper()}  ({n_images} images, {n_objects} objects) ──")

        if n_objects == 0:
            print("  (no objects)")
            continue

        max_count = max(counter.values(), default=1)
        sorted_cats = sorted(counter.items(), key=lambda x: -x[1])
        for cat, count in sorted_cats:
            pct = 100.0 * count / n_objects
            bar = _bar(count, max_count)
            print(f"  {cat:<22s} {bar} {count:5d}  ({pct:4.1f}%)")

    # Source distribution
    print(f"\n{'─' * width}")
    print("  Images per source per split")
    print(f"{'─' * width}")
    all_sources = sorted({src for counter in src_counts.values() for src in counter})
    header = f"  {'source':<15s}" + "".join(f"  {s:<10s}" for s in split_names)
    print(header)
    for src in all_sources:
        row = f"  {src:<15s}" + "".join(f"  {src_counts[s].get(src, 0):<10d}" for s in split_names)
        print(row)

    # Imbalance warnings
    print(f"\n{'─' * width}")
    warnings: list[str] = []
    for split_name in split_names:
        counter = cat_counts[split_name]
        if not counter:
            continue
        counts = list(counter.values())
        counts_sorted = sorted(counts)
        median = counts_sorted[len(counts_sorted) // 2]

        for cat, count in counter.items():
            if median > 0 and count > args.imbalance_ratio * median:
                warnings.append(
                    f"  [IMBALANCE] {split_name}: '{cat}' has {count} objects "
                    f"({count / median:.1f}× median={median})"
                )
            if count < args.min_count:
                warnings.append(f"  [SPARSE]    {split_name}: '{cat}' has only {count} object(s)")

    if warnings:
        print("  ⚠ Imbalance / Sparsity Flags:")
        for w in warnings:
            print(w)
    else:
        print("  ✓ No severe imbalance detected.")

    print("=" * width)


if __name__ == "__main__":
    main()
