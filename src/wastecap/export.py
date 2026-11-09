"""Validated exports for approved WasteCap annotations."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from wastecap.schema import AnnotationRecord, RecordStatus
from wastecap.store import list_records

logger = logging.getLogger(__name__)

_CSV_FIELDS = (
    "image_id",
    "file",
    "source",
    "license",
    "width",
    "height",
    "split",
    "scene_context",
    "caption",
    "status",
    "review",
    "objects",
    "candidates",
)


class ExportValidationError(ValueError):
    """Raised when records cannot be safely exported."""


def validate_records(
    records: Iterable[AnnotationRecord],
    *,
    require_approved: bool = True,
) -> list[AnnotationRecord]:
    """Validate records and raise with every export-blocking error."""
    validated = list(records)
    errors: list[str] = []
    for index, record in enumerate(validated):
        try:
            record = AnnotationRecord.model_validate(record.model_dump())
        except Exception as exc:
            errors.append(f"{index}:{getattr(record, 'image_id', '?')}: schema error: {exc}")
            continue

        if require_approved and record.status != RecordStatus.approved:
            errors.append(f"{record.image_id}: status is {record.status.value}, not approved")
        if not record.caption or not record.caption.strip():
            errors.append(f"{record.image_id}: empty image caption")

        object_ids = {obj.object_id for obj in record.objects}
        for obj in record.objects:
            if obj.instruction is not None and obj.object_id not in object_ids:
                errors.append(f"{record.image_id}: dangling instruction object_id {obj.object_id}")
        for candidate in record.candidates:
            parsed = candidate.raw_parsed
            if not isinstance(parsed, dict):
                continue
            referenced = parsed.get("_object_id")
            if referenced is not None and referenced not in object_ids:
                errors.append(
                    f"{record.image_id}: candidate {candidate.prompt_id} "
                    f"has dangling object_id {referenced}"
                )
            for parsed_obj in parsed.get("objects", []):
                if isinstance(parsed_obj, dict) and parsed_obj.get("object_id") not in (
                    None,
                    *object_ids,
                ):
                    errors.append(
                        f"{record.image_id}: candidate {candidate.prompt_id} "
                        f"has dangling object_id {parsed_obj['object_id']}"
                    )

    if errors:
        raise ExportValidationError("Export validation failed:\n- " + "\n- ".join(errors))
    return validated


def _record_dict(record: AnnotationRecord) -> dict[str, Any]:
    return record.model_dump(mode="json")


def _approved_records(
    records: Iterable[AnnotationRecord] | None,
    *,
    include_all: bool,
    db_path: Path | None,
    split: str | None,
) -> list[AnnotationRecord]:
    selected = list(records) if records is not None else list_records(split=split, db_path=db_path)
    if split is not None:
        selected = [record for record in selected if record.split and record.split.value == split]
    if not include_all:
        selected = [record for record in selected if record.status == RecordStatus.approved]
    return validate_records(selected, require_approved=not include_all)


def export_json(
    output_path: Path,
    records: Iterable[AnnotationRecord] | None = None,
    *,
    include_all: bool = False,
    db_path: Path | None = None,
    split: str | None = None,
) -> int:
    """Write a JSON array and return the number of records written."""
    selected = _approved_records(records, include_all=include_all, db_path=db_path, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([_record_dict(r) for r in selected], indent=2) + "\n")
    return len(selected)


def export_jsonl(
    output_path: Path,
    records: Iterable[AnnotationRecord] | None = None,
    *,
    include_all: bool = False,
    db_path: Path | None = None,
    split: str | None = None,
) -> int:
    """Write one JSON record per line and return the number written."""
    selected = _approved_records(records, include_all=include_all, db_path=db_path, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            handle.write(json.dumps(_record_dict(record), ensure_ascii=False) + "\n")
    return len(selected)


def export_csv(
    output_path: Path,
    records: Iterable[AnnotationRecord] | None = None,
    *,
    include_all: bool = False,
    db_path: Path | None = None,
    split: str | None = None,
) -> int:
    """Write one schema-preserving record per CSV row."""
    selected = _approved_records(records, include_all=include_all, db_path=db_path, split=split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in selected:
            data = _record_dict(record)
            for field in ("review", "objects", "candidates"):
                data[field] = json.dumps(data[field], ensure_ascii=False)
            writer.writerow({field: data.get(field) for field in _CSV_FIELDS})
    return len(selected)


def export_splits(
    output_dir: Path,
    records: Iterable[AnnotationRecord] | None = None,
    *,
    include_all: bool = False,
    db_path: Path | None = None,
) -> dict[str, dict[str, int]]:
    """Write JSON, JSONL, and CSV files for train/val/test."""
    selected = list(records) if records is not None else list_records(db_path=db_path)
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_records = [r for r in selected if r.split and r.split.value == split]
        result[split] = {
            "json": export_json(
                output_dir / f"{split}.json",
                split_records,
                include_all=include_all,
                db_path=db_path,
                split=split,
            ),
            "jsonl": export_jsonl(
                output_dir / f"{split}.jsonl",
                split_records,
                include_all=include_all,
                db_path=db_path,
                split=split,
            ),
            "csv": export_csv(
                output_dir / f"{split}.csv",
                split_records,
                include_all=include_all,
                db_path=db_path,
                split=split,
            ),
        }
    return result
