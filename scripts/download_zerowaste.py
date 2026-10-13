"""Download ZeroWaste dataset — stub (M1)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ZeroWaste dataset (M1)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        limit_str = f" (limit={args.limit})" if args.limit else ""
        print(f"[dry-run] Would download ZeroWaste dataset{limit_str} to data/raw/zerowaste/")
        return

    raise NotImplementedError("ZeroWaste download is implemented in M1.")


if __name__ == "__main__":
    main()
