"""Dataset statistics derived from the SQLite source of truth."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from wastecap.schema import RecordStatus
from wastecap.store import list_records


def compute_statistics(
    records: list | None = None,
    *,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Return image, class, approval, caption, and review-progress statistics."""
    items = records if records is not None else list_records(db_path=db_path)
    status = Counter(record.status.value for record in items)
    splits = Counter(record.split.value for record in items if record.split)
    classes = Counter(obj.category.value for record in items for obj in record.objects)
    captions_per_class = Counter(
        obj.category.value
        for record in items
        if record.status == RecordStatus.approved and record.caption and record.caption.strip()
        for obj in record.objects
    )
    clip_scores = [
        candidate.clip_score
        for record in items
        for candidate in record.candidates
        if candidate.clip_score is not None
    ]
    clip_scores_by_model: dict[str, list[float]] = {}
    for record in items:
        for candidate in record.candidates:
            if candidate.clip_score is not None:
                clip_scores_by_model.setdefault(candidate.model, []).append(candidate.clip_score)
    reviewed = sum(record.review is not None for record in items)
    approved = status[RecordStatus.approved.value]
    return {
        "images": len(items),
        "images_by_split": dict(sorted(splits.items())),
        "class_distribution": dict(sorted(classes.items())),
        "approval_rate": approved / len(items) if items else 0.0,
        "approved_images": approved,
        "captions_per_class": dict(sorted(captions_per_class.items())),
        "clipscore_distribution": sorted(clip_scores),
        "clipscore_by_model": {
            model: sorted(scores) for model, scores in sorted(clip_scores_by_model.items())
        },
        "review_progress": {
            "reviewed_images": reviewed,
            "unreviewed_images": len(items) - reviewed,
            "review_rate": reviewed / len(items) if items else 0.0,
            "status_counts": dict(sorted(status.items())),
        },
    }
