"""
Bounding-box utilities for WasteCap.

All functions work in pixel coordinates. Two formats are used throughout
the codebase:

  xywh  [x, y, w, h]   — COCO format, stored in AnnotationRecord
  xyxy  [x1, y1, x2, y2] — used internally for IoU and clipping

Coordinate convention: origin (0, 0) is top-left; x increases right,
y increases down. This matches PIL, COCO, and Qwen2.5-VL conventions.

Qwen2.5-VL note
---------------
Qwen2.5-VL returns grounding boxes in a *normalised 0-999* space.
Use qwen_to_pixel(box, width, height) to convert to pixel coordinates,
then xyxy_to_xywh() to match the stored schema format.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Minimum dimension (pixels) for a box to be considered valid
_MIN_DIMENSION = 1


# ---------------------------------------------------------------------------
# Format conversions
# ---------------------------------------------------------------------------


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        bbox: [x, y, width, height] in pixels.

    Returns:
        [x1, y1, x2, y2] where (x1,y1) is top-left, (x2,y2) is bottom-right.

    Raises:
        ValueError: if bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox}")
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(bbox: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        bbox: [x1, y1, x2, y2] in pixels.

    Returns:
        [x, y, width, height].

    Raises:
        ValueError: if bbox does not have exactly 4 elements.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox}")
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def qwen_to_pixel(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert Qwen2.5-VL normalised [x1, y1, x2, y2] (0-999 space) to pixel xyxy.

    Qwen2.5-VL returns box coordinates in [0, 999] × [0, 999] regardless of
    image size. This function rescales to actual pixel coordinates.

    Returns:
        [x1, y1, x2, y2] in pixel space (xyxy format).
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox}")
    x1, y1, x2, y2 = bbox
    sx = image_width / 1000.0
    sy = image_height / 1000.0
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_bbox(
    bbox: list[float],
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[bool, str]:
    """Validate a [x, y, w, h] bounding box.

    Args:
        bbox:         [x, y, w, h] in pixels.
        image_width:  If provided, check that the box is within the image.
        image_height: If provided, check that the box is within the image.

    Returns:
        (is_valid, reason). reason is "" when valid.
    """
    if len(bbox) != 4:
        return False, f"Expected 4 elements, got {len(bbox)}"

    x, y, w, h = bbox

    if any(math.isnan(v) or math.isinf(v) for v in bbox):
        return False, "bbox contains NaN or Inf values"

    if x < 0 or y < 0:
        return False, f"Negative coordinates: x={x}, y={y}"

    if w < _MIN_DIMENSION or h < _MIN_DIMENSION:
        return False, f"Degenerate box: w={w}, h={h} (must be ≥ {_MIN_DIMENSION})"

    if image_width is not None and x + w > image_width:
        return False, f"Box extends beyond image width: x+w={x + w} > {image_width}"

    if image_height is not None and y + h > image_height:
        return False, f"Box extends beyond image height: y+h={y + h} > {image_height}"

    return True, ""


def is_valid_bbox(
    bbox: list[float],
    image_width: int | None = None,
    image_height: int | None = None,
) -> bool:
    """Return True if bbox passes all validation checks."""
    valid, _ = validate_bbox(bbox, image_width, image_height)
    return valid


# ---------------------------------------------------------------------------
# Clipping and normalisation
# ---------------------------------------------------------------------------


def clip_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Clip a [x, y, w, h] bbox to the image boundaries.

    Coordinates are clamped to [0, image_width] × [0, image_height].
    Returns [x, y, w, h] after clipping (w or h may become 0 for
    out-of-bounds boxes — check is_valid_bbox after clipping).
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox}")
    x, y, w, h = bbox
    x1 = max(0.0, min(float(x), float(image_width)))
    y1 = max(0.0, min(float(y), float(image_height)))
    x2 = max(0.0, min(float(x + w), float(image_width)))
    y2 = max(0.0, min(float(y + h), float(image_height)))
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Normalise a [x, y, w, h] bbox to [0, 1] relative coordinates.

    Args:
        bbox:         [x, y, w, h] in pixels.
        image_width:  Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        [x_rel, y_rel, w_rel, h_rel] in [0, 1] space.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image dimensions must be positive: {image_width}×{image_height}")
    x, y, w, h = bbox
    return [x / image_width, y / image_height, w / image_width, h / image_height]


def denormalize_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """Convert normalised [x_rel, y_rel, w_rel, h_rel] back to pixel [x, y, w, h]."""
    x, y, w, h = bbox
    return [x * image_width, y * image_height, w * image_width, h * image_height]


# ---------------------------------------------------------------------------
# Area and centroid
# ---------------------------------------------------------------------------


def bbox_area(bbox: list[float]) -> float:
    """Return the area of a [x, y, w, h] bounding box in square pixels."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox}")
    return max(0.0, float(bbox[2])) * max(0.0, float(bbox[3]))


def bbox_centroid(bbox: list[float]) -> tuple[float, float]:
    """Return the (cx, cy) centroid of a [x, y, w, h] bounding box."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


# ---------------------------------------------------------------------------
# Spatial description helpers (used by instruction generator)
# ---------------------------------------------------------------------------

_X_THIRDS = (1 / 3, 2 / 3)
_Y_THIRDS = (1 / 3, 2 / 3)


def spatial_description(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> str:
    """Return a coarse spatial description of where the box sits in the image.

    Divides the image into a 3×3 grid and returns a phrase like
    "upper left", "centre", "lower right", etc.

    Args:
        bbox:         [x, y, w, h] in pixels.
        image_width:  Image width.
        image_height: Image height.

    Returns:
        Human-readable spatial description string.
    """
    cx, cy = bbox_centroid(bbox)

    if cx < image_width * _X_THIRDS[0]:
        h_label = "left"
    elif cx < image_width * _X_THIRDS[1]:
        h_label = "centre"
    else:
        h_label = "right"

    if cy < image_height * _Y_THIRDS[0]:
        v_label = "upper"
    elif cy < image_height * _Y_THIRDS[1]:
        v_label = "middle"
    else:
        v_label = "lower"

    if h_label == "centre" and v_label == "middle":
        return "centre"
    if h_label == "centre":
        return v_label
    if v_label == "middle":
        return h_label
    return f"{v_label} {h_label}"
