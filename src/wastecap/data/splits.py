"""Train/val/test split generator — stub (M1)."""

from __future__ import annotations

from pathlib import Path

from wastecap.schema import AnnotationRecord


def generate_splits(
    records: list[AnnotationRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    """Assign stratified train/val/test splits; write data/splits.json.

    Not implemented until M1.
    """
    raise NotImplementedError("Splits generator is implemented in M1.")
