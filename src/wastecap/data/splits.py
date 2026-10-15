"""
Train / val / test split generator for WasteCap.

Design constraints (from CLAUDE.md):
- Split by *source image* — no leakage between splits.
- Source-stratified: each source (taco, zerowaste, coco) contributes to all
  three splits in proportion to its size, preserving dataset balance.
- Seeded: reproducible with the same seed.
- Output: data/processed/splits.json (committed to git).
  Format: {"train": [image_id, ...], "val": [...], "test": [...]}
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from wastecap.schema import AnnotationRecord, Split

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("data/processed/splits.json")


def generate_splits(
    records: list[AnnotationRecord],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    output_path: Path | None = None,
) -> dict[str, list[str]]:
    """Assign stratified train/val/test splits and persist splits.json.

    Args:
        records:     All records from the manifest (split field may be None).
        train_ratio: Fraction for training (default 0.70).
        val_ratio:   Fraction for validation (default 0.15).
                     test_ratio = 1 - train_ratio - val_ratio (default 0.15).
        seed:        Random seed for reproducibility.
        output_path: Where to write splits.json.
                     Defaults to data/processed/splits.json.

    Returns:
        {"train": [image_id, ...], "val": [...], "test": [...]}

    Raises:
        ValueError: if ratios don't sum to ≤ 1.0 or records is empty.
    """
    if not records:
        raise ValueError("Cannot split an empty record list.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0")

    # Group by source for stratified splitting
    by_source: dict[str, list[str]] = {}
    for rec in records:
        src = rec.source.value
        by_source.setdefault(src, []).append(rec.image_id)

    rng = random.Random(seed)

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for src, ids in sorted(by_source.items()):  # sorted for determinism
        ids_copy = ids.copy()
        rng.shuffle(ids_copy)
        n = len(ids_copy)

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Give test whatever remains; ensures every source has ≥1 in each split
        # when n >= 3, otherwise train absorbs extras.
        if n < 3:
            # Edge case: tiny source — all go to train
            splits["train"].extend(ids_copy)
            logger.warning("Source %r has only %d image(s) — all assigned to train.", src, n)
            continue

        n_test = n - n_train - n_val
        if n_test < 1:
            # Shrink val by 1 to give test at least 1
            n_val -= 1
            n_test = 1

        splits["train"].extend(ids_copy[:n_train])
        splits["val"].extend(ids_copy[n_train : n_train + n_val])
        splits["test"].extend(ids_copy[n_train + n_val :])

        logger.info(
            "Source %r: %d total → train=%d val=%d test=%d",
            src,
            n,
            n_train,
            n_val,
            n - n_train - n_val,
        )

    # Sanity check: no id in two splits
    all_ids = splits["train"] + splits["val"] + splits["test"]
    assert len(all_ids) == len(set(all_ids)), "BUG: duplicate image_id across splits"
    assert len(all_ids) == len(records), "BUG: some records were lost during splitting"

    # Mutate records in-place to set the split field
    id_to_split = {
        img_id: Split(split_name) for split_name, id_list in splits.items() for img_id in id_list
    }
    for rec in records:
        rec.split = id_to_split.get(rec.image_id)

    # Write splits.json
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    logger.info(
        "Splits written to %s: train=%d val=%d test=%d",
        out,
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )

    return splits


def load_splits(path: Path | None = None) -> dict[str, list[str]]:
    """Load splits.json from disk.

    Returns:
        {"train": [...], "val": [...], "test": [...]}

    Raises:
        FileNotFoundError: if splits.json does not exist.
    """
    p = path or DEFAULT_OUTPUT
    if not p.exists():
        raise FileNotFoundError(f"splits.json not found at {p}. Run scripts/ingest.py first.")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)
