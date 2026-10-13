"""CLI entry point for WasteCap — stub (M2)."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Main CLI entry point. Subcommands are implemented in M2."""
    parser = argparse.ArgumentParser(
        prog="wastecap",
        description="WasteCap dataset curation pipeline",
    )
    parser.add_argument("--version", action="version", version="wastecap 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand stubs — logic added in M2
    subparsers.add_parser("generate", help="Generate captions (M2)")
    subparsers.add_parser("export", help="Export annotations (M6)")
    subparsers.add_parser("stats", help="Print dataset statistics (M6)")
    subparsers.add_parser("eval", help="Run baseline evaluation (M7)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    print(f"Subcommand '{args.command}' is not yet implemented (see milestone map).")
    sys.exit(1)


if __name__ == "__main__":
    main()
