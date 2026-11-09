"""Grounding accuracy evaluation for predicted and reference boxes."""

from __future__ import annotations

from wastecap.grounding.iou import DEFAULT_IOU_THRESHOLD, grounding_accuracy


def compute_grounding_accuracy(
    predicted_boxes: list[list[float]],
    reference_boxes: list[list[float]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> float:
    """Compute the fraction of reference boxes matched at the IoU threshold."""
    if iou_threshold < 0 or iou_threshold > 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    return grounding_accuracy(predicted_boxes, reference_boxes, iou_threshold)


def grounding_summary(
    predicted_boxes: list[list[float]],
    reference_boxes: list[list[float]],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[str, float | int]:
    """Return counts and accuracy for one image."""
    accuracy = compute_grounding_accuracy(predicted_boxes, reference_boxes, iou_threshold)
    matched = round(accuracy * len(reference_boxes))
    return {
        "predicted_boxes": len(predicted_boxes),
        "reference_boxes": len(reference_boxes),
        "matched_boxes": matched,
        "accuracy": accuracy,
    }
