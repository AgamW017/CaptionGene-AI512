"""ZeroWaste dataset loader — stub (implemented in M1)."""

from __future__ import annotations

from pathlib import Path

from wastecap.schema import AnnotationRecord


def load_zerowaste(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load ZeroWaste annotations and return AnnotationRecord stubs.

    Not implemented until M1.
    """
    raise NotImplementedError("ZeroWaste loader is implemented in M1.")
