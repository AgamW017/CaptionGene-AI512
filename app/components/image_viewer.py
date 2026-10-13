"""Image viewer with bounding box overlay — stub (M3)."""

from __future__ import annotations

from PIL import Image


def image_viewer(
    image: Image.Image,
    bboxes: list[list[float]] | None = None,
    labels: list[str] | None = None,
) -> None:
    """Render image with optional bbox overlays. Not implemented until M3."""
    raise NotImplementedError("image_viewer is implemented in M3.")
