"""Tests for wastecap.store — SQLite CRUD, round-trips, integrity."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from wastecap.schema import (
    AnnotationRecord,
    Candidate,
    DataSource,
    Material,
    ObjectAnnotation,
    RecordStatus,
    Review,
    SortingAction,
    Split,
    WasteCategory,
)
from wastecap.store import (
    candidate_model_counts,
    count_records,
    delete_record,
    get_record,
    ingest_record,
    init_db,
    list_records,
    source_counts,
    status_counts,
    update_record,
    upsert_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    image_id: str = "taco_000001",
    source: DataSource = DataSource.taco,
    split: Split = Split.train,
    status: RecordStatus = RecordStatus.pending,
) -> AnnotationRecord:
    return AnnotationRecord(
        image_id=image_id,
        file=f"data/processed/images/{image_id}.jpg",
        source=source,
        license="CC BY 4.0",
        width=640,
        height=480,
        split=split,
        status=status,
    )


def _make_approved_record(image_id: str = "taco_approved") -> AnnotationRecord:
    rev = Review(reviewer="test_reviewer", notes="looks good")
    return AnnotationRecord(
        image_id=image_id,
        file=f"data/processed/images/{image_id}.jpg",
        source=DataSource.taco,
        width=640,
        height=480,
        split=Split.val,
        status=RecordStatus.approved,
        review=rev,
        caption="crushed plastic bottle on pavement",
    )


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


def test_init_db_creates_file(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "test.db"
    result = init_db(path)
    assert result == path
    assert path.exists()


def test_init_db_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "test.db"
    init_db(path)
    init_db(path)  # should not raise
    assert path.exists()


# ---------------------------------------------------------------------------
# ingest_record + get_record (core round-trip)
# ---------------------------------------------------------------------------


def test_ingest_and_get_round_trip(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)
    fetched = get_record(rec.image_id, db_path)
    assert fetched is not None
    assert fetched.image_id == rec.image_id
    assert fetched.source == rec.source
    assert fetched.width == rec.width
    assert fetched.split == rec.split
    assert fetched.status == rec.status


def test_get_record_missing_returns_none(db_path: Path) -> None:
    assert get_record("nonexistent", db_path) is None


def test_ingest_duplicate_raises(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)
    with pytest.raises(sqlite3.IntegrityError):
        ingest_record(rec, db_path)


# ---------------------------------------------------------------------------
# update_record
# ---------------------------------------------------------------------------


def test_update_record(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)

    rec.split = Split.val
    rec.scene_context = "outdoor, cluttered"
    update_record(rec, db_path)

    fetched = get_record(rec.image_id, db_path)
    assert fetched.split == Split.val
    assert fetched.scene_context == "outdoor, cluttered"


def test_update_nonexistent_raises(db_path: Path) -> None:
    rec = _make_record()
    with pytest.raises(KeyError, match="not found"):
        update_record(rec, db_path)


# ---------------------------------------------------------------------------
# upsert_record
# ---------------------------------------------------------------------------


def test_upsert_inserts_new(db_path: Path) -> None:
    rec = _make_record()
    upsert_record(rec, db_path)
    assert get_record(rec.image_id, db_path) is not None


def test_upsert_updates_existing(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)
    rec.scene_context = "updated context"
    upsert_record(rec, db_path)
    fetched = get_record(rec.image_id, db_path)
    assert fetched.scene_context == "updated context"


# ---------------------------------------------------------------------------
# Candidates persistence
# ---------------------------------------------------------------------------


def test_candidates_round_trip(db_path: Path) -> None:
    cand = Candidate(model="mock", prompt_id="default", text="a crushed bottle", clip_score=0.75)
    rec = _make_record()
    rec.candidates = [cand]
    ingest_record(rec, db_path)

    fetched = get_record(rec.image_id, db_path)
    assert len(fetched.candidates) == 1
    assert fetched.candidates[0].model == "mock"
    assert fetched.candidates[0].clip_score == pytest.approx(0.75)


def test_candidates_updated_on_update(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)

    cand1 = Candidate(model="mock", prompt_id="p1", text="caption one")
    cand2 = Candidate(model="blip2", prompt_id="p1", text="caption two")
    rec.candidates = [cand1, cand2]
    update_record(rec, db_path)

    fetched = get_record(rec.image_id, db_path)
    assert len(fetched.candidates) == 2
    models = {c.model for c in fetched.candidates}
    assert models == {"mock", "blip2"}


# ---------------------------------------------------------------------------
# Object annotations persistence
# ---------------------------------------------------------------------------


def test_objects_round_trip(db_path: Path) -> None:
    obj = ObjectAnnotation(
        object_id="o1",
        category=WasteCategory.bottle,
        material=Material.plastic,
        bbox=[10.0, 20.0, 50.0, 60.0],
        sorting_action=SortingAction.recycle_plastic,
    )
    rec = _make_record()
    rec.objects = [obj]
    ingest_record(rec, db_path)

    fetched = get_record(rec.image_id, db_path)
    assert len(fetched.objects) == 1
    assert fetched.objects[0].category == WasteCategory.bottle
    assert fetched.objects[0].bbox == [10.0, 20.0, 50.0, 60.0]


# ---------------------------------------------------------------------------
# list_records + count_records
# ---------------------------------------------------------------------------


def test_list_records_all(db_path: Path) -> None:
    for i in range(3):
        ingest_record(_make_record(f"img_{i:03d}"), db_path)
    records = list_records(db_path=db_path)
    assert len(records) == 3


def test_list_records_filter_source(db_path: Path) -> None:
    ingest_record(_make_record("t1", source=DataSource.taco), db_path)
    ingest_record(_make_record("z1", source=DataSource.zerowaste), db_path)
    taco_only = list_records(source="taco", db_path=db_path)
    assert len(taco_only) == 1
    assert taco_only[0].image_id == "t1"


def test_list_records_filter_status(db_path: Path) -> None:
    ingest_record(_make_record("p1"), db_path)
    ingest_record(_make_approved_record("a1"), db_path)
    pending = list_records(status="pending", db_path=db_path)
    approved = list_records(status="approved", db_path=db_path)
    assert len(pending) == 1
    assert len(approved) == 1


def test_list_records_limit_offset(db_path: Path) -> None:
    for i in range(5):
        ingest_record(_make_record(f"img_{i:03d}"), db_path)
    page1 = list_records(limit=2, offset=0, db_path=db_path)
    page2 = list_records(limit=2, offset=2, db_path=db_path)
    assert len(page1) == 2
    assert len(page2) == 2
    ids1 = {r.image_id for r in page1}
    ids2 = {r.image_id for r in page2}
    assert ids1.isdisjoint(ids2)


def test_count_records(db_path: Path) -> None:
    for i in range(4):
        ingest_record(_make_record(f"c_{i}"), db_path)
    assert count_records(db_path=db_path) == 4
    assert count_records(source="taco", db_path=db_path) == 4
    assert count_records(source="zerowaste", db_path=db_path) == 0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def test_status_counts(db_path: Path) -> None:
    ingest_record(_make_record("s1"), db_path)
    ingest_record(_make_record("s2"), db_path)
    ingest_record(_make_approved_record("s3"), db_path)
    counts = status_counts(db_path=db_path)
    assert counts.get("pending", 0) == 2
    assert counts.get("approved", 0) == 1


def test_source_counts(db_path: Path) -> None:
    ingest_record(_make_record("t1", source=DataSource.taco), db_path)
    ingest_record(_make_record("z1", source=DataSource.zerowaste), db_path)
    ingest_record(_make_record("z2", source=DataSource.zerowaste), db_path)
    counts = source_counts(db_path=db_path)
    assert counts.get("taco", 0) == 1
    assert counts.get("zerowaste", 0) == 2


def test_candidate_model_counts(db_path: Path) -> None:
    cand_mock = Candidate(model="mock", prompt_id="p1", text="a")
    cand_blip = Candidate(model="blip2", prompt_id="p1", text="b")
    rec = _make_record()
    rec.candidates = [cand_mock, cand_blip]
    ingest_record(rec, db_path)
    counts = candidate_model_counts(db_path=db_path)
    assert counts.get("mock", 0) == 1
    assert counts.get("blip2", 0) == 1


# ---------------------------------------------------------------------------
# delete_record
# ---------------------------------------------------------------------------


def test_delete_record(db_path: Path) -> None:
    rec = _make_record()
    ingest_record(rec, db_path)
    assert get_record(rec.image_id, db_path) is not None
    delete_record(rec.image_id, db_path)
    assert get_record(rec.image_id, db_path) is None


# ---------------------------------------------------------------------------
# Fixture records round-trip
# ---------------------------------------------------------------------------


def test_fixture_records_round_trip(db_path: Path, fixture_records: list) -> None:
    """All 6 fixture records must survive ingest → get without data loss."""
    for rec in fixture_records:
        ingest_record(rec, db_path)

    assert count_records(db_path=db_path) == len(fixture_records)

    for rec in fixture_records:
        fetched = get_record(rec.image_id, db_path)
        assert fetched is not None
        assert fetched.image_id == rec.image_id
        assert fetched.source == rec.source
        assert fetched.width == rec.width
        assert len(fetched.objects) == len(rec.objects)
        assert len(fetched.candidates) == len(rec.candidates)
