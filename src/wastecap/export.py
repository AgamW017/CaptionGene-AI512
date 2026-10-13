"""Export approved annotations to JSON / JSONL / CSV — stub (M6)."""

from __future__ import annotations

from pathlib import Path


def export_json(output_path: Path, include_all: bool = False) -> int:
    """Export approved records to JSON.

    Returns number of records written. Not implemented until M6.
    """
    raise NotImplementedError("Export is implemented in M6.")


def export_jsonl(output_path: Path, include_all: bool = False) -> int:
    """Export approved records to JSONL. Not implemented until M6."""
    raise NotImplementedError("Export is implemented in M6.")


def export_csv(output_path: Path, include_all: bool = False) -> int:
    """Export approved records to CSV. Not implemented until M6."""
    raise NotImplementedError("Export is implemented in M6.")
