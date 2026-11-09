"""M6 export, validation, and statistics tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from wastecap.export import (
    ExportValidationError,
    export_csv,
    export_json,
    export_jsonl,
    export_splits,
)
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
from wastecap.stats import compute_statistics


def _record(image_id: str, split: Split, status: RecordStatus = RecordStatus.approved):
    return AnnotationRecord(
        image_id=image_id,
        file=f"data/processed/images/{image_id}.jpg",
        source=DataSource.taco,
        width=100,
        height=100,
        split=split,
        caption=f"Approved caption for {image_id}.",
        status=status,
        review=Review(reviewer="tester") if status == RecordStatus.approved else None,
        objects=[
            ObjectAnnotation(
                object_id="o1",
                category=WasteCategory.bottle,
                material=Material.plastic,
                bbox=[10, 10, 20, 20],
                sorting_action=SortingAction.recycle_plastic,
            )
        ],
        candidates=[
            Candidate(
                model="mock",
                prompt_id="test",
                text="caption",
                clip_score=0.42,
                raw_parsed={"objects": [{"object_id": "o1"}]},
            )
        ],
    )


def test_exports_round_trip_all_formats(tmp_path: Path) -> None:
    record = _record("train_1", Split.train)
    json_path = tmp_path / "train.json"
    jsonl_path = tmp_path / "train.jsonl"
    csv_path = tmp_path / "train.csv"

    assert export_json(json_path, [record]) == 1
    assert export_jsonl(jsonl_path, [record]) == 1
    assert export_csv(csv_path, [record]) == 1

    loaded_json = json.loads(json_path.read_text())
    loaded_jsonl = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    with csv_path.open(newline="") as handle:
        loaded_csv = list(csv.DictReader(handle))

    assert loaded_json[0]["image_id"] == "train_1"
    assert loaded_jsonl[0]["objects"][0]["object_id"] == "o1"
    assert json.loads(loaded_csv[0]["objects"])[0]["object_id"] == "o1"


def test_export_splits_writes_one_file_per_format(tmp_path: Path) -> None:
    counts = export_splits(
        tmp_path,
        [_record("train_1", Split.train), _record("val_1", Split.val)],
    )
    assert counts["train"] == {"json": 1, "jsonl": 1, "csv": 1}
    assert counts["test"] == {"json": 0, "jsonl": 0, "csv": 0}
    assert (tmp_path / "val.json").exists()
    assert (tmp_path / "test.csv").exists()


def test_export_rejects_empty_caption() -> None:
    record = _record("empty", Split.train)
    record.caption = " "
    with pytest.raises(ExportValidationError, match="empty image caption"):
        export_json(Path("/tmp/unused-export.json"), [record])


def test_export_rejects_dangling_candidate_object_id() -> None:
    record = _record("dangling", Split.train)
    record.candidates[0].raw_parsed = {"_object_id": "missing"}
    with pytest.raises(ExportValidationError, match="dangling object_id"):
        export_json(Path("/tmp/unused-export.json"), [record])


def test_statistics_include_progress_and_class_counts() -> None:
    records = [_record("a", Split.train), _record("b", Split.val, RecordStatus.pending)]
    stats = compute_statistics(records)
    assert stats["images"] == 2
    assert stats["class_distribution"] == {"bottle": 2}
    assert stats["captions_per_class"] == {"bottle": 1}
    assert stats["approved_images"] == 1
    assert stats["review_progress"]["reviewed_images"] == 1
    assert stats["clipscore_distribution"] == [0.42, 0.42]
    assert stats["clipscore_by_model"] == {"mock": [0.42, 0.42]}
