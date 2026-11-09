"""
Mask utilities for WasteCap.

Two mask formats are used (matching the typed schema union):

  PolygonMask  data: list of flat [x, y, x, y, ...] rings
  RLEMask      counts: COCO compressed-RLE string, size: [height, width]

Public functions convert masks to bounding boxes and compute polygon areas.
We deliberately do NOT decode full pixel masks at annotation time — masks
are stored as-is and only decoded when needed for IoU evaluation (M7).

RLE decoding
------------
The RLEMask.counts field may contain:
  - A COCO compact string (e.g. "Sfk25Z")  → use pycocotools if available
  - A comma-separated integer string        → manual decode
Both are handled; pycocotools is imported lazily and silently skipped if
not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polygon → bbox
# ---------------------------------------------------------------------------


def polygon_to_bbox(rings: list[list[float]]) -> list[float]:
    """Compute the tight [x, y, w, h] bounding box of polygon rings.

    Args:
        rings: List of flat coordinate rings, each [x0, y0, x1, y1, ...].
               The polygon may have multiple rings (holes are ignored for bbox).

    Returns:
        [x, y, w, h] tight bounding box.

    Raises:
        ValueError: if rings is empty or contains no valid coordinates.
    """
    xs: list[float] = []
    ys: list[float] = []

    for ring in rings:
        if len(ring) < 4 or len(ring) % 2 != 0:
            logger.debug("Skipping malformed polygon ring (len=%d)", len(ring))
            continue
        xs.extend(ring[0::2])  # every even index
        ys.extend(ring[1::2])  # every odd index

    if not xs:
        raise ValueError("No valid polygon coordinates found in rings")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def polygon_area(rings: list[list[float]]) -> float:
    """Compute the signed area of the first polygon ring using the shoelace formula.

    Returns the absolute area in square pixels. Holes (additional rings) are
    not subtracted — this is an approximation sufficient for sorting by size.
    """
    if not rings:
        return 0.0
    ring = rings[0]
    if len(ring) < 6 or len(ring) % 2 != 0:
        return 0.0

    xs = ring[0::2]
    ys = ring[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    return abs(area) / 2.0


# ---------------------------------------------------------------------------
# RLE → bbox
# ---------------------------------------------------------------------------


def _decode_rle_counts(counts_str: str, height: int, width: int) -> list[int] | None:
    """Decode a comma-separated uncompressed RLE into a flat count list.

    Returns None if decoding fails (pycocotools path attempted first).
    """
    # Try pycocotools first (handles compact COCO RLE strings)
    try:
        from pycocotools import mask as coco_mask

        rle = {"counts": counts_str, "size": [height, width]}
        decoded = coco_mask.decode(rle)  # returns H×W uint8 array
        return decoded
    except (ImportError, Exception):
        pass

    # Manual decode of comma-separated integer string
    try:
        counts = [int(c) for c in counts_str.split(",") if c.strip()]
        return counts
    except (ValueError, AttributeError):
        return None


def rle_to_bbox(counts_str: str, height: int, width: int) -> list[float] | None:
    """Compute the tight [x, y, w, h] bounding box from a COCO RLE mask.

    Tries pycocotools (compact RLE) first, then falls back to manual
    decoding of comma-separated integer counts.

    Returns None if decoding fails.
    """
    # Fast path via pycocotools
    try:
        from pycocotools import mask as coco_mask

        rle = {"counts": counts_str, "size": [height, width]}
        bbox_xyxy = coco_mask.toBbox(rle).tolist()  # returns [x, y, w, h]
        return bbox_xyxy
    except (ImportError, Exception):
        pass

    # Manual path: decode comma-separated counts to find pixel bounding box
    try:
        counts = [int(c) for c in counts_str.split(",") if c.strip()]
    except (ValueError, AttributeError):
        logger.debug("Failed to parse RLE counts string")
        return None

    # Reconstruct column-major binary mask from run-length counts
    total_pixels = height * width
    pixel_idx = 0
    xs_on: list[int] = []
    ys_on: list[int] = []
    value = 0  # RLE alternates between 0 and 1, starting with 0

    for run_len in counts:
        if value == 1:
            for k in range(run_len):
                if pixel_idx + k < total_pixels:
                    col = (pixel_idx + k) // height
                    row = (pixel_idx + k) % height
                    xs_on.append(col)
                    ys_on.append(row)
        pixel_idx += run_len
        value = 1 - value

    if not xs_on:
        return None

    x_min, x_max = min(xs_on), max(xs_on)
    y_min, y_max = min(ys_on), max(ys_on)
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


# ---------------------------------------------------------------------------
# Unified mask_to_bbox
# ---------------------------------------------------------------------------


def mask_to_bbox(mask: object) -> list[float] | None:
    """Compute [x, y, w, h] bounding box from either a PolygonMask or RLEMask.

    Args:
        mask: PolygonMask or RLEMask instance from wastecap.schema.

    Returns:
        [x, y, w, h] tight bounding box, or None if computation fails.
    """
    from wastecap.schema import PolygonMask, RLEMask

    if isinstance(mask, PolygonMask):
        try:
            return polygon_to_bbox(mask.data)
        except ValueError as exc:
            logger.warning("polygon_to_bbox failed: %s", exc)
            return None

    if isinstance(mask, RLEMask):
        if len(mask.size) < 2:
            logger.warning("RLEMask.size must have 2 elements: %s", mask.size)
            return None
        return rle_to_bbox(mask.counts, mask.size[0], mask.size[1])

    logger.warning("Unknown mask type: %s", type(mask))
    return None
