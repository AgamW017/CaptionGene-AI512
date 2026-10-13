"""Mask utilities — stub (M5)."""

from __future__ import annotations

from wastecap.schema import PolygonMask, RLEMask


def rle_to_polygon(mask: RLEMask) -> PolygonMask:
    """Convert RLE mask to polygon. Not implemented until M5."""
    raise NotImplementedError("Mask utilities are implemented in M5.")


def mask_to_bbox(mask: RLEMask | PolygonMask) -> list[float]:
    """Compute tight bounding box from mask. Not implemented until M5."""
    raise NotImplementedError("Mask utilities are implemented in M5.")
