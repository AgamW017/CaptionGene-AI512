"""Grounding accuracy evaluation — stub (M7)."""

from __future__ import annotations


def compute_grounding_accuracy(
    predicted_boxes: list[list[float]],
    reference_boxes: list[list[float]],
    iou_threshold: float = 0.5,
) -> float:
    """Compute fraction of predicted boxes with IoU >= threshold. Not implemented until M7."""
    raise NotImplementedError("Grounding eval is implemented in M7.")
