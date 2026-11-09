"""Validate the committed split manifest against the local annotation DB."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wastecap.data.splits import load_splits, validate_splits
from wastecap.store import list_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=Path("data/processed/splits.json"))
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()
    splits = load_splits(args.splits)
    records = list_records(db_path=args.db_path)
    validate_splits(splits, records)
    print(
        "validated "
        + ", ".join(f"{name}={len(ids)}" for name, ids in splits.items())
        + f" records={len(records)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
