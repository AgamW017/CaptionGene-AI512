"""TACO dataset loader — stub (implemented in M1)."""

from __future__ import annotations

from pathlib import Path

from wastecap.schema import AnnotationRecord


def load_taco(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load TACO annotations from raw_dir and return AnnotationRecord stubs.

    Not implemented until M1.
    """
    raise NotImplementedError("TACO loader is implemented in M1.")
