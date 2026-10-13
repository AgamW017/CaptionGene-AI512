"""Tests for wastecap.taxonomy — label mapping, defaults, list functions."""

from __future__ import annotations

from wastecap.schema import DataSource, Material, SortingAction, WasteCategory
from wastecap.taxonomy import (
    CanonicalLabel,
    category_info,
    default_sorting_action,
    list_categories,
    list_conditions,
    list_materials,
    list_sorting_actions,
    map_source_label,
)

# ---------------------------------------------------------------------------
# List functions
# ---------------------------------------------------------------------------


def test_list_categories_length() -> None:
    cats = list_categories()
    assert len(cats) == 16


def test_list_categories_contains_expected() -> None:
    cats = list_categories()
    for expected in ["bottle", "can", "cup", "bag", "food_waste", "electronic", "unknown"]:
        assert expected in cats


def test_list_materials() -> None:
    mats = list_materials()
    expected = {
        "plastic",
        "metal",
        "paper",
        "cardboard",
        "glass",
        "organic",
        "textile",
        "mixed",
        "unknown",
    }
    assert set(mats) == expected


def test_list_conditions() -> None:
    conds = list_conditions()
    assert "crushed" in conds
    assert "partially_occluded" in conds
    assert len(conds) == 6


def test_list_sorting_actions_has_cardboard() -> None:
    """recycle_cardboard must be a separate action."""
    actions = list_sorting_actions()
    assert "recycle_cardboard" in actions
    assert "recycle_paper" in actions
    assert len(actions) == 8


# ---------------------------------------------------------------------------
# TACO label mapping
# ---------------------------------------------------------------------------


def test_taco_clear_plastic_bottle() -> None:
    label = map_source_label("taco", "Clear plastic bottle")
    assert label == CanonicalLabel(WasteCategory.bottle, Material.plastic)


def test_taco_glass_bottle() -> None:
    label = map_source_label("taco", "Glass bottle")
    assert label == CanonicalLabel(WasteCategory.bottle, Material.glass)


def test_taco_drink_can() -> None:
    label = map_source_label("taco", "Drink can")
    assert label == CanonicalLabel(WasteCategory.can, Material.metal)


def test_taco_crisp_packet() -> None:
    label = map_source_label("taco", "Crisp packet")
    assert label == CanonicalLabel(WasteCategory.wrapper, Material.plastic)


def test_taco_unknown_label_falls_back() -> None:
    label = map_source_label("taco", "Alien debris")
    assert label.category == WasteCategory.unknown
    assert label.material == Material.unknown


def test_taco_with_datasource_enum() -> None:
    label = map_source_label(DataSource.taco, "Straw")
    assert label == CanonicalLabel(WasteCategory.straw, Material.plastic)


# ---------------------------------------------------------------------------
# ZeroWaste label mapping
# ---------------------------------------------------------------------------


def test_zerowaste_rigid_plastic() -> None:
    label = map_source_label("zerowaste", "rigid_plastic")
    assert label == CanonicalLabel(WasteCategory.container, Material.plastic)


def test_zerowaste_cardboard() -> None:
    label = map_source_label("zerowaste", "cardboard")
    assert label == CanonicalLabel(WasteCategory.box, Material.cardboard)


def test_zerowaste_metal() -> None:
    label = map_source_label("zerowaste", "metal")
    assert label == CanonicalLabel(WasteCategory.can, Material.metal)


def test_zerowaste_soft_plastic() -> None:
    label = map_source_label("zerowaste", "soft_plastic")
    assert label == CanonicalLabel(WasteCategory.wrapper, Material.plastic)


# ---------------------------------------------------------------------------
# COCO label mapping
# ---------------------------------------------------------------------------


def test_coco_bottle() -> None:
    label = map_source_label("coco", "bottle")
    assert label == CanonicalLabel(WasteCategory.bottle, Material.unknown)


def test_coco_wine_glass() -> None:
    label = map_source_label("coco", "wine glass")
    assert label == CanonicalLabel(WasteCategory.bottle, Material.glass)


def test_coco_banana() -> None:
    label = map_source_label("coco", "banana")
    assert label == CanonicalLabel(WasteCategory.food_waste, Material.organic)


def test_coco_cell_phone() -> None:
    label = map_source_label("coco", "cell phone")
    assert label == CanonicalLabel(WasteCategory.electronic, Material.mixed)


# ---------------------------------------------------------------------------
# Default sorting actions
# ---------------------------------------------------------------------------


def test_default_sorting_bottle_plastic() -> None:
    assert default_sorting_action("bottle", "plastic") == SortingAction.recycle_plastic


def test_default_sorting_bottle_glass() -> None:
    assert default_sorting_action("bottle", "glass") == SortingAction.recycle_glass


def test_default_sorting_can_metal() -> None:
    assert default_sorting_action("can", "metal") == SortingAction.recycle_metal


def test_default_sorting_carton_cardboard() -> None:
    assert default_sorting_action("carton", "cardboard") == SortingAction.recycle_cardboard


def test_default_sorting_food_waste_organic() -> None:
    assert default_sorting_action("food_waste", "organic") == SortingAction.compost


def test_default_sorting_electronic() -> None:
    assert default_sorting_action("electronic", "mixed") == SortingAction.hazardous


def test_default_sorting_bag_plastic() -> None:
    """Soft plastic bags go to landfill (not kerbside recyclable)."""
    assert default_sorting_action("bag", "plastic") == SortingAction.landfill


def test_default_sorting_unknown_pair_fallback() -> None:
    """Unknown (category, material) pairs fall back to landfill."""
    assert default_sorting_action("unknown", "unknown") == SortingAction.landfill


def test_default_sorting_with_enums() -> None:
    result = default_sorting_action(WasteCategory.box, Material.cardboard)
    assert result == SortingAction.recycle_cardboard


# ---------------------------------------------------------------------------
# Category info
# ---------------------------------------------------------------------------


def test_category_info_bottle() -> None:
    info = category_info("bottle")
    assert info is not None
    assert info["id"] == 0
    assert "bottle" in info["description"].lower()


def test_category_info_unknown_category() -> None:
    assert category_info("spaceship") is None
