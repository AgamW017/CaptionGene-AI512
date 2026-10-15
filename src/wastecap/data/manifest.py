"""
Unified image manifest for WasteCap.

Responsibilities:
- Merge AnnotationRecord lists from multiple sources into one registry.
- Deduplicate by SHA-256 content hash (same image bytes from different sources).
- Apply basic quality filters (minimum resolution, non-zero dimensions).
- Assign globally unique image_ids in the form '<source>_<zero-padded-index>'.

The manifest does NOT copy or resize images.  That is the job of the
pre-processing pipeline (scripts/ingest.py).  It operates on records
that already have valid `file` paths.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from wastecap.schema import AnnotationRecord, DataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality filter thresholds (tunable, not hard-coded in callers)
# ---------------------------------------------------------------------------

MIN_WIDTH: int = 100  # pixels
MIN_HEIGHT: int = 100  # pixels
MIN_AREA: int = MIN_WIDTH * MIN_HEIGHT


def _sha256(path: Path) -> str | None:
    """Return hex SHA-256 of the file at *path*, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as exc:
        logger.warning("Cannot hash %s: %s", path, exc)
        return None


def _quality_ok(rec: AnnotationRecord) -> bool:
    """Return True if the record passes the minimum quality threshold."""
    if rec.width < MIN_WIDTH or rec.height < MIN_HEIGHT:
        logger.debug(
            "Skipping %s: resolution %dx%d below threshold %dx%d",
            rec.image_id,
            rec.width,
            rec.height,
            MIN_WIDTH,
            MIN_HEIGHT,
        )
        return False
    return True


def _canonical_id(source: DataSource, index: int) -> str:
    """Return canonical image_id like 'taco_000042'."""
    return f"{source.value}_{index:06d}"


def build_manifest(
    records: list[AnnotationRecord],
    *,
    deduplicate: bool = True,
    quality_filter: bool = True,
    reindex: bool = True,
) -> list[AnnotationRecord]:
    """Merge, deduplicate and quality-filter a list of AnnotationRecords.

    Args:
        records:        Combined list from all loaders.
        deduplicate:    If True, drop images whose file content hash has already
                        been seen (keeps the first occurrence).
        quality_filter: If True, drop images below MIN_WIDTH × MIN_HEIGHT.
        reindex:        If True, assign canonical image_ids (<source>_<index>)
                        within each source, resetting any loader-assigned ids.

    Returns:
        Filtered, deduplicated list with canonical image_ids.
    """
    seen_hashes: dict[str, str] = {}  # hash -> image_id of first occurrence
    kept: list[AnnotationRecord] = []
    dropped_quality = 0
    dropped_duplicate = 0

    # Per-source counters for reindexing
    source_counters: dict[str, int] = {}

    for rec in records:
        # Quality filter first (cheap, no I/O)
        if quality_filter and not _quality_ok(rec):
            dropped_quality += 1
            continue

        # Content-hash deduplication (requires file access)
        if deduplicate:
            file_path = Path(rec.file)
            if file_path.exists():
                digest = _sha256(file_path)
                if digest is not None:
                    if digest in seen_hashes:
                        logger.debug(
                            "Duplicate of %s — dropping %s",
                            seen_hashes[digest],
                            rec.image_id,
                        )
                        dropped_duplicate += 1
                        continue
                    seen_hashes[digest] = rec.image_id
            # If the file doesn't exist yet (dry-run / pre-download) skip hash check

        if reindex:
            src = rec.source.value
            idx = source_counters.get(src, 0)
            source_counters[src] = idx + 1
            rec = rec.model_copy(update={"image_id": _canonical_id(rec.source, idx)})

        kept.append(rec)

    logger.info(
        "Manifest: %d kept, %d dropped (quality), %d dropped (duplicate)",
        len(kept),
        dropped_quality,
        dropped_duplicate,
    )
    return kept
