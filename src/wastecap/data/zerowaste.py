"""
ZeroWaste dataset loader.

Expected raw_dir layout (after download_zerowaste.py):
    data/raw/zerowaste/
        zerowaste-f/
            train/
                data/
                    *.png
                labels.json     # COCO-format instance segmentation
            val/
                data/
                    *.png
                labels.json
            test/                   # may not be present in all releases
                data/
                    *.png
                labels.json

ZeroWaste annotation format: COCO JSON with 4 classes:
  rigid_plastic, cardboard, metal, soft_plastic

License: Research use only (non-commercial).
Reference: Bashkirova et al. (CVPR 2022), arXiv:2106.02740
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from wastecap.schema import (
    AnnotationRecord,
    DataSource,
    ObjectAnnotation,
    PolygonMask,
    RLEMask,
)
from wastecap.taxonomy import default_sorting_action, map_source_label

logger = logging.getLogger(__name__)

ZEROWASTE_LICENSE = "Research use only (non-commercial)"
_SPLITS_DIRS = ["train", "val", "test"]


def _load_split_dir(
    split_dir: Path,
    limit: int | None,
    running_count: int,
) -> tuple[list[AnnotationRecord], int]:
    """Load one ZeroWaste split directory (train/val/test).

    Returns (records, new_running_count).
    """
    labels_file = split_dir / "labels.json"
    if not labels_file.exists():
        logger.debug("No labels.json in %s — skipping.", split_dir)
        return [], running_count

    with labels_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_by_id: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco.get("images", [])
    records: list[AnnotationRecord] = []

    for img in images:
        if limit is not None and running_count >= limit:
            break

        img_id_raw: int = img["id"]
        _file_name: str = img.get("file_name", "")
        width: int = img.get("width", 0)
        height: int = img.get("height", 0)

        tmp_id = f"zerowaste_raw_{img_id_raw}"
        processed_path = f"data/processed/images/{tmp_id}.jpg"

        objects: list[ObjectAnnotation] = []
        for i, ann in enumerate(anns_by_image.get(img_id_raw, [])):
            zw_label = cat_by_id.get(ann["category_id"], "soft_plastic")
            canonical = map_source_label(DataSource.zerowaste, zw_label)
            sorting = default_sorting_action(canonical.category, canonical.material)

            bbox = ann.get("bbox")
            bbox_clean: list[float] | None = None
            if bbox and len(bbox) == 4 and all(v >= 0 for v in bbox):
                bbox_clean = [float(v) for v in bbox]

            mask: PolygonMask | RLEMask | None = None
            segs = ann.get("segmentation")
            if isinstance(segs, dict) and "counts" in segs:
                # RLE format
                counts = segs["counts"]
                size = segs.get("size", [height, width])
                if isinstance(counts, list):
                    # Uncompressed RLE — store as string for portability
                    counts_str = ",".join(str(c) for c in counts)
                else:
                    counts_str = str(counts)
                mask = RLEMask(counts=counts_str, size=size)
            elif isinstance(segs, list) and len(segs) > 0:
                flat_rings = [
                    [float(v) for v in ring]
                    for ring in segs
                    if isinstance(ring, list) and len(ring) >= 6
                ]
                if flat_rings:
                    mask = PolygonMask(data=flat_rings)

            objects.append(
                ObjectAnnotation(
                    object_id=f"o{i + 1}",
                    category=canonical.category,
                    material=canonical.material,
                    bbox=bbox_clean,
                    mask=mask,
                    sorting_action=sorting,
                )
            )

        records.append(
            AnnotationRecord(
                image_id=tmp_id,
                file=processed_path,
                source=DataSource.zerowaste,
                license=ZEROWASTE_LICENSE,
                width=width,
                height=height,
                objects=objects,
            )
        )
        running_count += 1

    return records, running_count


def load_zerowaste(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load ZeroWaste annotations from all available splits.

    Args:
        raw_dir: Path to data/raw/zerowaste/ (contains zerowaste-f/ subdirectory).
        limit:   Total cap on images loaded across all splits (None = all).

    Returns:
        List of AnnotationRecord stubs.

    Raises:
        FileNotFoundError: if the zerowaste-f directory is not found.
    """
    # Support both data/raw/zerowaste/ and data/raw/zerowaste/zerowaste-f/ layouts
    base = raw_dir / "zerowaste-f"
    if not base.exists():
        base = raw_dir
        # Confirm we have at least one split dir
        if not any((base / s).exists() for s in _SPLITS_DIRS):
            raise FileNotFoundError(
                f"ZeroWaste data not found under {raw_dir}. "
                "Run: python scripts/download_zerowaste.py"
            )

    records: list[AnnotationRecord] = []
    count = 0

    for split_name in _SPLITS_DIRS:
        split_dir = base / split_name
        if not split_dir.exists():
            continue
        new_recs, count = _load_split_dir(split_dir, limit, count)
        records.extend(new_recs)
        if limit is not None and count >= limit:
            break

    logger.info("ZeroWaste: loaded %d image records from %s", len(records), raw_dir)
    return records
