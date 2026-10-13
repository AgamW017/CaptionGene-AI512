"""Tests for wastecap.schema — Pydantic models and validation rules."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wastecap.schema import (
    AnnotationRecord,
    Candidate,
    Condition,
    DataSource,
    Material,
    ObjectAnnotation,
    PolygonMask,
    RecordStatus,
    Review,
    RLEMask,
    SortingAction,
    WasteCategory,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_waste_category_values() -> None:
    assert WasteCategory("bottle") == WasteCategory.bottle
    assert WasteCategory("unknown") == WasteCategory.unknown
    with pytest.raises(ValueError):
        WasteCategory("spaceship")


def test_sorting_action_has_cardboard() -> None:
    """recycle_cardboard must be separate from recycle_paper (approved plan decision)."""
    assert SortingAction.recycle_cardboard != SortingAction.recycle_paper


def test_all_enums_round_trip() -> None:
    for cat in WasteCategory:
        assert WasteCategory(cat.value) == cat
    for mat in Material:
        assert Material(mat.value) == mat
    for cond in Condition:
        assert Condition(cond.value) == cond
    for action in SortingAction:
        assert SortingAction(action.value) == action


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def test_rle_mask() -> None:
    m = RLEMask(counts="abc123", size=[200, 200])
    assert m.type == "rle"
    assert m.size == [200, 200]


def test_polygon_mask() -> None:
    m = PolygonMask(data=[[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]])
    assert m.type == "polygon"
    assert len(m.data[0]) == 8


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------


def test_candidate_defaults() -> None:
    c = Candidate(model="mock", prompt_id="default", text="a plastic bottle")
    assert c.clip_score is None
    assert not c.parse_failed
    assert c.raw_parsed is None
    assert c.grounding_boxes is None


def test_candidate_with_clip_score() -> None:
    c = Candidate(model="blip2", prompt_id="p1", text="hello", clip_score=0.82)
    assert c.clip_score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# ObjectAnnotation
# ---------------------------------------------------------------------------


def test_object_annotation_defaults() -> None:
    obj = ObjectAnnotation(object_id="o1")
    assert obj.category == WasteCategory.unknown
    assert obj.material == Material.unknown
    assert obj.count == 1
    assert obj.bbox is None


def test_object_annotation_full() -> None:
    obj = ObjectAnnotation(
        object_id="o1",
        category=WasteCategory.bottle,
        material=Material.plastic,
        condition=Condition.crushed,
        bbox=[10.0, 20.0, 50.0, 60.0],
        sorting_action=SortingAction.recycle_plastic,
        instruction="Pick up the crushed plastic bottle and place it in the recycling bin.",
    )
    assert obj.category == WasteCategory.bottle
    assert obj.instruction is not None


def test_object_annotation_negative_bbox_rejected() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        ObjectAnnotation(object_id="o1", bbox=[-1.0, 0.0, 50.0, 50.0])


def test_instruction_without_bbox_rejected() -> None:
    with pytest.raises(ValidationError, match="bbox"):
        ObjectAnnotation(
            object_id="o1",
            instruction="Pick it up.",
            bbox=None,
        )


# ---------------------------------------------------------------------------
# AnnotationRecord
# ---------------------------------------------------------------------------


def test_annotation_record_minimal() -> None:
    rec = AnnotationRecord(
        image_id="taco_000001",
        file="data/processed/images/taco_000001.jpg",
        source=DataSource.taco,
        width=640,
        height=480,
    )
    assert rec.status == RecordStatus.pending
    assert rec.split is None
    assert rec.objects == []
    assert rec.candidates == []


def test_annotation_record_full_round_trip(fixture_records: list) -> None:
    """All fixture records must parse and re-serialise without data loss."""
    for rec in fixture_records:
        json_str = rec.model_dump_json()
        rec2 = AnnotationRecord.model_validate_json(json_str)
        assert rec2.image_id == rec.image_id
        assert rec2.source == rec.source
        assert rec2.width == rec.width
        assert len(rec2.objects) == len(rec.objects)


def test_approved_status_requires_review() -> None:
    with pytest.raises(ValidationError, match="Review"):
        AnnotationRecord(
            image_id="x",
            file="x.jpg",
            source=DataSource.taco,
            width=100,
            height=100,
            status=RecordStatus.approved,
            review=None,
        )


def test_approved_with_review_passes() -> None:
    rev = Review(reviewer="test_reviewer")
    rec = AnnotationRecord(
        image_id="x",
        file="x.jpg",
        source=DataSource.taco,
        width=100,
        height=100,
        status=RecordStatus.approved,
        review=rev,
    )
    assert rec.status == RecordStatus.approved


def test_duplicate_object_ids_rejected() -> None:
    obj1 = ObjectAnnotation(object_id="o1")
    obj2 = ObjectAnnotation(object_id="o1")
    with pytest.raises(ValidationError, match="Duplicate object_ids"):
        AnnotationRecord(
            image_id="x",
            file="x.jpg",
            source=DataSource.taco,
            width=100,
            height=100,
            objects=[obj1, obj2],
        )


def test_polygon_mask_in_record() -> None:
    mask = PolygonMask(data=[[0.0, 0.0, 50.0, 0.0, 50.0, 50.0]])
    obj = ObjectAnnotation(object_id="o1", mask=mask)
    rec = AnnotationRecord(
        image_id="mask_test",
        file="x.jpg",
        source=DataSource.zerowaste,
        width=200,
        height=200,
        objects=[obj],
    )
    # Round-trip preserves mask type
    rec2 = AnnotationRecord.model_validate_json(rec.model_dump_json())
    assert rec2.objects[0].mask is not None
    assert rec2.objects[0].mask.type == "polygon"


def test_rle_mask_in_record() -> None:
    mask = RLEMask(counts="abc123", size=[200, 200])
    obj = ObjectAnnotation(object_id="o1", mask=mask)
    rec = AnnotationRecord(
        image_id="rle_test",
        file="x.jpg",
        source=DataSource.zerowaste,
        width=200,
        height=200,
        objects=[obj],
    )
    rec2 = AnnotationRecord.model_validate_json(rec.model_dump_json())
    assert rec2.objects[0].mask.type == "rle"
