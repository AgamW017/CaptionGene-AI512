"""Ingest raw images into the DB — stub (M1)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw images into DB (M1)")
    parser.add_argument("--source", choices=["taco", "zerowaste", "coco", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] Would ingest source={args.source} limit={args.limit}")
        return

    raise NotImplementedError("Ingest pipeline is implemented in M1.")


if __name__ == "__main__":
    main()
