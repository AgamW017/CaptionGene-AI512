"""Bounding box utilities — stub (M5)."""

from __future__ import annotations


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]. Not implemented until M5."""
    raise NotImplementedError("Box utilities are implemented in M5.")


def clip_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    """Clip bbox to image bounds. Not implemented until M5."""
    raise NotImplementedError("Box utilities are implemented in M5.")
