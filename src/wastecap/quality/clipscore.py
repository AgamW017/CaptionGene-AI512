"""CLIPScore computation — stub (M4)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def compute_clipscore(image: Image.Image, text: str) -> float:
    """Compute CLIPScore for a single (image, text) pair. Not implemented until M4."""
    raise NotImplementedError("CLIPScore is implemented in M4.")


def batch_clipscore(
    image_paths: list[Path],
    texts: list[str],
) -> list[float]:
    """Compute CLIPScore for a batch. Not implemented until M4."""
    raise NotImplementedError("Batch CLIPScore is implemented in M4.")
