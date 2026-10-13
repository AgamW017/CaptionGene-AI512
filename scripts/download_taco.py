"""Download TACO dataset — stub (M1)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TACO dataset (M1)")
    parser.add_argument("--limit", type=int, default=None, help="Max images to download")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done")
    args = parser.parse_args()

    if args.dry_run:
        limit_str = f" (limit={args.limit})" if args.limit else ""
        print(f"[dry-run] Would download TACO dataset{limit_str} to data/raw/taco/")
        return

    raise NotImplementedError("TACO download is implemented in M1.")


if __name__ == "__main__":
    main()
