"""Batch CLIPScore computation — stub (M4)."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CLIPScore for all candidates (M4)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--model", default=None, help="Filter by model name")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] Would compute CLIPScore for split={args.split} model={args.model}")
        return

    raise NotImplementedError("CLIPScore computation is implemented in M4.")


if __name__ == "__main__":
    main()
