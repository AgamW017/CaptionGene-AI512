"""
COCO subset loader — waste-relevant classes only.

Expected raw_dir layout (after download_coco.py):
    data/raw/coco/
        annotations/
            instances_train2017.json   # or instances_val2017.json
        images/
            train2017/
                *.jpg
            val2017/
                *.jpg

Only images that contain at least one annotation from COCO_WASTE_CLASSES
are loaded. Objects from other classes in those images are silently ignored.

License: CC BY 4.0
Reference: Lin et al. (ECCV 2014), arXiv:1405.0312
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

COCO_LICENSE = "CC BY 4.0"

# Canonical COCO class names that map to our waste taxonomy (taxonomy.yaml: coco section)
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

_ANN_FILES = [
    "annotations/instances_train2017.json",
    "annotations/instances_val2017.json",
]


def _load_annotation_file(
    ann_file: Path,
    images_base: Path,
    limit: int | None,
    running_count: int,
) -> tuple[list[AnnotationRecord], int]:
    """Load one COCO annotations file, keeping only waste-class images."""
    with ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_by_id: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # Filter category ids to waste classes only
    waste_cat_ids: set[int] = {cid for cid, name in cat_by_id.items() if name in COCO_WASTE_CLASSES}

    # Only keep images that have at least one waste annotation
    waste_image_ids: set[int] = set()
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        if ann["category_id"] in waste_cat_ids:
            waste_image_ids.add(ann["image_id"])
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = [img for img in coco.get("images", []) if img["id"] in waste_image_ids]

    records: list[AnnotationRecord] = []
    for img in images:
        if limit is not None and running_count >= limit:
            break

        img_id_raw: int = img["id"]
        _file_name: str = img.get("file_name", "")
        width: int = img.get("width", 0)
        height: int = img.get("height", 0)

        tmp_id = f"coco_raw_{img_id_raw}"
        processed_path = f"data/processed/images/{tmp_id}.jpg"

        objects: list[ObjectAnnotation] = []
        obj_counter = 0
        for ann in anns_by_image.get(img_id_raw, []):
            if ann["category_id"] not in waste_cat_ids:
                continue
            coco_label = cat_by_id[ann["category_id"]]
            canonical = map_source_label(DataSource.coco, coco_label)
            sorting = default_sorting_action(canonical.category, canonical.material)

            bbox = ann.get("bbox")
            bbox_clean: list[float] | None = None
            if bbox and len(bbox) == 4 and all(v >= 0 for v in bbox):
                bbox_clean = [float(v) for v in bbox]

            mask: PolygonMask | RLEMask | None = None
            segs = ann.get("segmentation")
            if isinstance(segs, dict) and "counts" in segs:
                counts = segs["counts"]
                size = segs.get("size", [height, width])
                if isinstance(counts, list):
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

            obj_counter += 1
            objects.append(
                ObjectAnnotation(
                    object_id=f"o{obj_counter}",
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
                source=DataSource.coco,
                license=COCO_LICENSE,
                width=width,
                height=height,
                objects=objects,
            )
        )
        running_count += 1

    return records, running_count


def load_coco_subset(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load COCO images that contain waste-relevant objects.

    Args:
        raw_dir: Path to data/raw/coco/ (contains annotations/ and images/).
        limit:   Total cap on images loaded across train2017 + val2017.

    Returns:
        List of AnnotationRecord stubs with only waste-class objects.

    Raises:
        FileNotFoundError: if no COCO annotation files are found.
    """
    found_any = False
    records: list[AnnotationRecord] = []
    count = 0

    for ann_rel in _ANN_FILES:
        ann_file = raw_dir / ann_rel
        if not ann_file.exists():
            continue
        found_any = True

        # Determine images base dir from the annotation filename
        split_name = ann_file.stem.replace("instances_", "")  # e.g. "train2017"
        images_base = raw_dir / "images" / split_name

        new_recs, count = _load_annotation_file(ann_file, images_base, limit, count)
        records.extend(new_recs)

        if limit is not None and count >= limit:
            break

    if not found_any:
        raise FileNotFoundError(
            f"No COCO annotation files found under {raw_dir}/annotations/. "
            "Run: python scripts/download_coco.py"
        )

    logger.info("COCO subset: loaded %d image records from %s", len(records), raw_dir)
    return records
