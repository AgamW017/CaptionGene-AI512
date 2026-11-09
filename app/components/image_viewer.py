"""
Image viewer component: renders a PIL image with optional bounding-box overlays.

Uses ImageDraw to paint boxes directly onto a copy of the image, then hands
the result to st.image so no extra dependencies are needed.

Public API
----------
image_viewer(image, bboxes, labels, caption, use_container_width)
load_image(path) -> PIL.Image | None
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# Colours for up to 8 distinct objects; cycles if more.
_BOX_COLOURS = [
    "#FF4B4B",  # red
    "#4B9CFF",  # blue
    "#4BFF7A",  # green
    "#FFD24B",  # yellow
    "#FF4BF0",  # magenta
    "#4BFFF0",  # cyan
    "#FF974B",  # orange
    "#C14BFF",  # purple
]
_BOX_WIDTH = 3


def _draw_boxes(
    image: Image.Image,
    bboxes: list[list[float]],
    labels: list[str] | None,
) -> Image.Image:
    """Return a copy of *image* with bboxes drawn on it.

    Args:
        image:  Source PIL image (RGB).
        bboxes: List of [x, y, w, h] in pixel coordinates.
        labels: Optional label per bbox. Shown as a small tag above the box.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    for i, bbox in enumerate(bboxes):
        if len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        x1 = max(0, min(int(x), img.width - 1))
        y1 = max(0, min(int(y), img.height - 1))
        x2 = max(x1 + 1, min(int(x + w), img.width - 1))
        y2 = max(y1 + 1, min(int(y + h), img.height - 1))
        colour = _BOX_COLOURS[i % len(_BOX_COLOURS)]
        width = max(_BOX_WIDTH, round(min(img.width, img.height) / 180))
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)

        if labels and i < len(labels):
            label = labels[i]
            # Small label background at top-left corner of box
            try:
                font = ImageFont.load_default()
                bbox_txt = draw.textbbox((x1, max(0, y1 - 16)), label, font=font)
                draw.rectangle(bbox_txt, fill=colour)
                draw.text((x1, max(0, y1 - 16)), label, fill="white", font=font)
            except Exception:
                pass  # label rendering is best-effort

    return img


def image_viewer(
    image: Image.Image,
    bboxes: list[list[float]] | None = None,
    labels: list[str] | None = None,
    caption: str | None = None,
    use_container_width: bool = True,
) -> None:
    """Render a PIL image, optionally with bounding-box overlays.

    Args:
        image:               PIL Image (any mode; converted to RGB internally).
        bboxes:              List of [x, y, w, h] boxes to draw.
        labels:              Optional label per box (e.g. object_id or category).
        caption:             Caption shown below the image.
        use_container_width: Passed to st.image.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    display = _draw_boxes(image, bboxes, labels) if bboxes else image

    st.image(display, caption=caption, use_container_width=use_container_width)
    if bboxes:
        valid = sum(1 for bbox in bboxes if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0)
        st.caption(f"Showing {valid} reference box(es) · source annotation overlay")


def load_image(path: str | Path) -> Image.Image | None:
    """Load a PIL Image from *path*. Returns None (with a warning) on failure.

    Centralised here so all pages use the same error-handling pattern.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Image not found: %s", p)
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception as exc:
        logger.warning("Cannot open image %s: %s", p, exc)
        return None
