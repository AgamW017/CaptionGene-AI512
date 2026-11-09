"""
Intersection-over-Union (IoU) utilities for WasteCap grounding evaluation.

All boxes are in [x, y, w, h] format (COCO convention) matching the schema.

Grounding accuracy definition (from CLAUDE.md):
    IoU ≥ 0.5 between a predicted box and the reference box counts as a
    correct grounding. This threshold is the standard PASCAL VOC criterion.

Qwen2.5-VL predicted boxes (stored in Candidate.grounding_boxes after M2)
are evaluated against the reference boxes in ObjectAnnotation.bbox using
these utilities in M7's grounding_eval.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Standard VOC/COCO grounding accuracy threshold
DEFAULT_IOU_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Core IoU
# ---------------------------------------------------------------------------


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union between two [x, y, w, h] boxes.

    Args:
        box_a: [x, y, w, h] in pixels.
        box_b: [x, y, w, h] in pixels.

    Returns:
        IoU in [0.0, 1.0]. Returns 0.0 for degenerate boxes.

    Raises:
        ValueError: if either box does not have exactly 4 elements.
    """
    if len(box_a) != 4:
        raise ValueError(f"box_a must have 4 elements, got {len(box_a)}")
    if len(box_b) != 4:
        raise ValueError(f"box_b must have 4 elements, got {len(box_b)}")

    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    # Degenerate boxes
    if wa <= 0 or ha <= 0 or wb <= 0 or hb <= 0:
        return 0.0

    # Convert to xyxy for intersection calculation
    ax1, ay1, ax2, ay2 = xa, ya, xa + wa, ya + ha
    bx1, by1, bx2, by2 = xb, yb, xb + wb, yb + hb

    # Intersection rectangle
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    intersection = inter_w * inter_h

    if intersection == 0.0:
        return 0.0

    area_a = wa * ha
    area_b = wb * hb
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compute_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes.

    Convenience wrapper for callers already holding xyxy boxes
    (e.g. Qwen grounding boxes after conversion).
    """
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("Both boxes must have 4 elements")
    from wastecap.grounding.boxes import xyxy_to_xywh

    return compute_iou(xyxy_to_xywh(box_a), xyxy_to_xywh(box_b))


# ---------------------------------------------------------------------------
# Batch IoU
# ---------------------------------------------------------------------------


def batch_iou(
    predicted: list[list[float]],
    reference: list[list[float]],
) -> list[list[float]]:
    """Compute an IoU matrix between predicted and reference boxes.

    Args:
        predicted:  List of N predicted [x, y, w, h] boxes.
        reference:  List of M reference [x, y, w, h] boxes.

    Returns:
        N×M matrix (list of lists) of IoU values.
        result[i][j] = IoU(predicted[i], reference[j]).
    """
    return [[compute_iou(p, r) for r in reference] for p in predicted]


def best_match(
    predicted: list[list[float]],
    reference: list[list[float]],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Find the best reference match for each predicted box (greedy).

    Each reference box can only be matched once. Predicted boxes are
    processed in order; the highest-IoU unmatched reference is assigned.

    Args:
        predicted:  List of predicted [x, y, w, h] boxes.
        reference:  List of reference [x, y, w, h] boxes.
        threshold:  Minimum IoU to count as a match.

    Returns:
        List of (pred_idx, ref_idx, iou) tuples for matched pairs.
        Unmatched predictions (IoU < threshold) are omitted.
    """
    if not predicted or not reference:
        return []

    iou_matrix = batch_iou(predicted, reference)
    matched_refs: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for pred_idx, ious in enumerate(iou_matrix):
        best_iou = -1.0
        best_ref = -1
        for ref_idx, iou_val in enumerate(ious):
            if ref_idx not in matched_refs and iou_val > best_iou:
                best_iou = iou_val
                best_ref = ref_idx

        if best_ref >= 0 and best_iou >= threshold:
            matches.append((pred_idx, best_ref, best_iou))
            matched_refs.add(best_ref)

    return matches


def grounding_accuracy(
    predicted: list[list[float]],
    reference: list[list[float]],
    threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Compute grounding accuracy: fraction of reference boxes correctly matched.

    A reference box is "correctly grounded" if at least one predicted box
    has IoU ≥ threshold with it.

    Returns:
        Float in [0.0, 1.0]. Returns 0.0 if reference is empty.
    """
    if not reference:
        return 0.0
    if not predicted:
        return 0.0

    matched = best_match(predicted, reference, threshold)
    matched_refs = {ref_idx for _, ref_idx, _ in matched}
    return len(matched_refs) / len(reference)
