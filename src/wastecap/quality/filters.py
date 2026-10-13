"""Quality filter helpers — stub (M4)."""

from __future__ import annotations

from wastecap.schema import AnnotationRecord


def filter_by_clipscore(
    records: list[AnnotationRecord],
    threshold: float,
) -> list[AnnotationRecord]:
    """Return records whose best candidate CLIPScore >= threshold. Not implemented until M4."""
    raise NotImplementedError("Filters are implemented in M4.")
