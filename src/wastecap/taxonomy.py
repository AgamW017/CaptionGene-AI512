"""
Taxonomy loader for WasteCap.

Reads configs/taxonomy.yaml and provides helper functions for:
- Mapping source dataset labels to canonical (category, material) pairs.
- Listing canonical values for validation and UI dropdowns.
- Looking up the default sorting action for a (category, material) pair.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import yaml

from wastecap.schema import DataSource, Material, SortingAction, WasteCategory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class CanonicalLabel(NamedTuple):
    category: WasteCategory
    material: Material


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _taxonomy_path() -> Path:
    """Resolve taxonomy.yaml from WASTECAP_CONFIGS_DIR env var or project root."""
    configs_dir = os.environ.get("WASTECAP_CONFIGS_DIR", "configs")
    path = Path(configs_dir) / "taxonomy.yaml"
    if not path.exists():
        # Try relative to this file's location (src/wastecap/)
        fallback = Path(__file__).parent.parent.parent / "configs" / "taxonomy.yaml"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f"taxonomy.yaml not found at {path} or {fallback}. "
            "Set WASTECAP_CONFIGS_DIR to the configs/ directory."
        )
    return path


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict:
    """Load and cache the raw taxonomy YAML."""
    path = _taxonomy_path()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Taxonomy loaded from %s", path)
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_categories() -> list[str]:
    """Return canonical category names in canonical order."""
    tax = _load_taxonomy()
    return [c["name"] for c in tax["categories"]]


def list_materials() -> list[str]:
    """Return canonical material names."""
    return _load_taxonomy()["materials"]


def list_conditions() -> list[str]:
    """Return canonical condition names."""
    return _load_taxonomy()["conditions"]


def list_sorting_actions() -> list[str]:
    """Return canonical sorting action names."""
    return _load_taxonomy()["sorting_actions"]


def map_source_label(source: str | DataSource, label: str) -> CanonicalLabel:
    """Map a source-dataset label to a canonical (category, material) pair.

    Args:
        source: One of 'taco', 'zerowaste', 'coco' (or DataSource enum).
        label:  The label string as it appears in the source dataset.

    Returns:
        CanonicalLabel(category, material).  Falls back to
        (unknown, unknown) if the label is not in the taxonomy, and logs a warning.
    """
    source_key = source.value if isinstance(source, DataSource) else str(source)
    tax = _load_taxonomy()
    mapping: dict = tax.get(source_key, {})

    if label not in mapping:
        logger.warning(
            "Unknown label %r for source %r — mapping to (unknown, unknown)", label, source_key
        )
        return CanonicalLabel(
            category=WasteCategory.unknown,
            material=Material.unknown,
        )

    entry = mapping[label]
    try:
        category = WasteCategory(entry["category"])
    except ValueError:
        logger.warning("Invalid category %r for label %r — using unknown", entry["category"], label)
        category = WasteCategory.unknown

    try:
        material = Material(entry["material"])
    except ValueError:
        logger.warning("Invalid material %r for label %r — using unknown", entry["material"], label)
        material = Material.unknown

    return CanonicalLabel(category=category, material=material)


def default_sorting_action(
    category: str | WasteCategory, material: str | Material
) -> SortingAction:
    """Look up the default sorting action for a (category, material) pair.

    Falls back to 'landfill' if no specific mapping exists.
    """
    cat_key = category.value if isinstance(category, WasteCategory) else str(category)
    mat_key = material.value if isinstance(material, Material) else str(material)
    compound_key = f"{cat_key}/{mat_key}"

    tax = _load_taxonomy()
    defaults: dict = tax.get("default_sorting_action", {})

    action_str = defaults.get(compound_key)
    if action_str is None:
        logger.debug("No default sorting action for %r — falling back to landfill", compound_key)
        return SortingAction.landfill

    try:
        return SortingAction(action_str)
    except ValueError:
        logger.warning(
            "Invalid sorting action %r in taxonomy for key %r — using landfill",
            action_str,
            compound_key,
        )
        return SortingAction.landfill


def category_info(name: str) -> dict | None:
    """Return full category dict (id, name, description) for a canonical name."""
    tax = _load_taxonomy()
    for entry in tax["categories"]:
        if entry["name"] == name:
            return entry
    return None


def reload_taxonomy() -> None:
    """Clear the LRU cache and force a reload on the next call.

    Useful in tests when WASTECAP_CONFIGS_DIR is patched.
    """
    _load_taxonomy.cache_clear()
    _taxonomy_path.cache_clear() if hasattr(_taxonomy_path, "cache_clear") else None  # type: ignore[attr-defined]
