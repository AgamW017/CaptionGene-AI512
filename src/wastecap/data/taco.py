"""
TACO dataset loader.

Expected raw_dir layout (after download_taco.py):
    data/raw/taco/
        annotations.json        # COCO-format annotations file
        data/
            batch_1/
                000001.jpg
                ...
            batch_2/
                ...

TACO annotation format: standard COCO JSON with:
  - images[].id, file_name, width, height, flickr_url, license
  - annotations[].id, image_id, category_id, bbox [x,y,w,h], segmentation (polygon)
  - categories[].id, name, supercategory

License: CC BY 4.0
Reference: Proença & Simões (2020), arXiv:2003.06975
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
)
from wastecap.taxonomy import default_sorting_action, map_source_label

logger = logging.getLogger(__name__)

TACO_LICENSE = "CC BY 4.0"
TACO_ANNOTATIONS_FILE = "annotations.json"


def load_taco(
    raw_dir: Path,
    limit: int | None = None,
) -> list[AnnotationRecord]:
    """Load TACO annotations and return normalised AnnotationRecord list.

    Args:
        raw_dir: Path to data/raw/taco/ (must contain annotations.json).
        limit:   Cap on number of images to load (None = all).

    Returns:
        List of AnnotationRecord stubs with objects populated from TACO annotations.
        Images not yet generated / downloaded will have the original flickr_url
        stored; `file` points to the expected processed path.

    Raises:
        FileNotFoundError: if annotations.json is not present.
    """
    ann_file = raw_dir / TACO_ANNOTATIONS_FILE
    if not ann_file.exists():
        raise FileNotFoundError(
            f"TACO annotations not found at {ann_file}. Run: python scripts/download_taco.py"
        )

    with ann_file.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    # Build lookup tables
    cat_by_id: dict[int, str] = {c["id"]: c["name"] for c in coco.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco.get("images", [])
    if limit is not None:
        images = images[:limit]

    records: list[AnnotationRecord] = []
    for img in images:
        img_id_raw: int = img["id"]
        _file_name: str = img.get("file_name", "")
        width: int = img.get("width", 0)
        height: int = img.get("height", 0)
        license_str: str = img.get("license", TACO_LICENSE)
        if isinstance(license_str, int):
            # TACO stores license as an integer id in some versions
            license_str = TACO_LICENSE

        # Canonical image_id assigned later by manifest.build_manifest();
        # use a stable temporary id here so annotation cross-refs work.
        tmp_id = f"taco_raw_{img_id_raw}"
        processed_path = f"data/processed/images/{tmp_id}.jpg"

        # Build ObjectAnnotations from TACO annotations
        objects: list[ObjectAnnotation] = []
        for i, ann in enumerate(anns_by_image.get(img_id_raw, [])):
            taco_label = cat_by_id.get(ann["category_id"], "Other")
            canonical = map_source_label(DataSource.taco, taco_label)
            sorting = default_sorting_action(canonical.category, canonical.material)

            bbox = ann.get("bbox")  # [x, y, w, h]
            bbox_clean: list[float] | None = None
            if bbox and len(bbox) == 4 and all(v >= 0 for v in bbox):
                bbox_clean = [float(v) for v in bbox]

            mask: PolygonMask | None = None
            segs = ann.get("segmentation")
            if segs and isinstance(segs, list) and len(segs) > 0:
                # TACO segmentation: list of polygon rings (flat [x,y,...] lists)
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
                source=DataSource.taco,
                license=TACO_LICENSE,
                width=width,
                height=height,
                objects=objects,
            )
        )

    logger.info("TACO: loaded %d image records from %s", len(records), ann_file)
    return records
