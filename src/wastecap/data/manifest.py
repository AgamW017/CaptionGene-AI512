"""Image manifest: unified registry across all sources — stub (M1)."""

from __future__ import annotations

from wastecap.schema import AnnotationRecord


def build_manifest(records: list[AnnotationRecord]) -> list[AnnotationRecord]:
    """Deduplicate records by content hash and assign canonical image_ids.

    Not implemented until M1.
    """
    raise NotImplementedError("Manifest is implemented in M1.")
