"""Focused AppTest coverage for export validation and download controls."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from wastecap.schema import (
    AnnotationRecord,
    DataSource,
    Material,
    ObjectAnnotation,
    RecordStatus,
    Review,
    SortingAction,
    Split,
    WasteCategory,
)
from wastecap.store import ingest_record, init_db


def test_export_page_exposes_download_buttons_after_export(tmp_path: Path, monkeypatch) -> None:
    record = AnnotationRecord(
        image_id="export_app_1",
        file=str(Path(__file__).parent / "fixtures/images/taco_fix001.jpg"),
        source=DataSource.taco,
        width=200,
        height=200,
        split=Split.train,
        caption="A plastic bottle.",
        status=RecordStatus.approved,
        review=Review(reviewer="tester"),
        objects=[
            ObjectAnnotation(
                object_id="o1",
                category=WasteCategory.bottle,
                material=Material.plastic,
                bbox=[10, 10, 30, 30],
                sorting_action=SortingAction.recycle_plastic,
            )
        ],
    )
    db_path = init_db(tmp_path / "test.db")
    ingest_record(record, db_path)
    monkeypatch.setenv("WASTECAP_DB_PATH", str(db_path))
    page = Path(__file__).parent.parent / "app/pages/5_Export.py"
    app = AppTest.from_file(str(page), default_timeout=20).run()
    assert not app.exception
    app.text_input[0].set_value(str(tmp_path / "exports"))
    app.button[0].click().run()
    assert not app.exception
    for extension in ("json", "jsonl", "csv"):
        assert (tmp_path / "exports" / f"train.{extension}").exists()
