"""COCO subset loader — stub (implemented in M1)."""

from __future__ import annotations

from pathlib import Path

from wastecap.schema import AnnotationRecord

# Canonical COCO class names that are waste-relevant (from taxonomy.yaml mapping).
COCO_WASTE_CLASSES: frozenset[str] = frozenset(
    [
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "backpack",
        "handbag",
        "suitcase",
        "cell phone",
        "book",
        "scissors",
    ]
)


def load_coco_subset(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load COCO images containing waste-relevant classes and return stubs.

    Not implemented until M1.
    """
    raise NotImplementedError("COCO subset loader is implemented in M1.")
