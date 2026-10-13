"""Download COCO waste-relevant subset — stub (M1)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download COCO subset (M1)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        limit_str = f" (limit={args.limit})" if args.limit else ""
        print(f"[dry-run] Would download COCO waste subset{limit_str} to data/raw/coco/")
        return

    raise NotImplementedError("COCO download is implemented in M1.")


if __name__ == "__main__":
    main()
