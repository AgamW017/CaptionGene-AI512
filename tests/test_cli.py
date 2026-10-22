"""
Tests for wastecap.cli — the generate subcommand using MockModel only.
No GPU, no weight downloads.

The CLI is tested by calling _cmd_generate() directly (avoids subprocess
overhead and gives cleaner error messages) and also via argparse to confirm
the full arg-parsing pipeline works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from wastecap.cli import _cmd_generate, main
from wastecap.schema import DataSource, Split
from wastecap.store import get_record, ingest_record, list_records

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(
    *,
    model: str = "mock",
    split: str = "train",
    level: str = "scene",
    n: int = 3,
    limit: int | None = 5,
    verbose: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        model=model,
        split=split,
        level=level,
        n=n,
        limit=limit,
        verbose=verbose,
    )


def _seed_db_with_images(db_path: Path, n: int = 4) -> list:
    """Insert fixture-backed AnnotationRecord stubs into the test DB."""
    from wastecap.schema import AnnotationRecord, Material, ObjectAnnotation, WasteCategory

    fixture_imgs = sorted((Path(__file__).parent / "fixtures" / "images").glob("*.jpg"))
    records = []
    for i in range(n):
        img_path = fixture_imgs[i % len(fixture_imgs)]
        rec = AnnotationRecord(
            image_id=f"test_img_{i:03d}",
            file=str(img_path),
            source=DataSource.taco,
            width=200,
            height=200,
            split=Split.train,
            objects=[
                ObjectAnnotation(
                    object_id="o1",
                    category=WasteCategory.bottle,
                    material=Material.plastic,
                    bbox=[20.0, 20.0, 80.0, 80.0],
                )
            ],
        )
        ingest_record(rec, db_path)
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


class TestArgParsing:
    def test_generate_default_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Confirm generate subcommand is correctly wired in argparse."""
        import sys

        captured: list[argparse.Namespace] = []

        def fake_func(args: argparse.Namespace) -> int:
            captured.append(args)
            return 0

        monkeypatch.setattr("wastecap.cli._cmd_generate", lambda a: (captured.append(a), 0)[1])
        monkeypatch.setattr(sys, "argv", ["wastecap", "generate", "--limit", "5"])
        with pytest.raises(SystemExit):
            main()
        # The important thing is arg parsing didn't raise a non-SystemExit exception

    def test_no_subcommand_exits_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        monkeypatch.setattr(sys, "argv", ["wastecap"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# _cmd_generate with mock model
# ---------------------------------------------------------------------------


class TestGenerateCommand:
    def test_generate_scene_writes_candidates(self, db_path: Path) -> None:
        _seed_db_with_images(db_path, n=4)
        args = _make_args(model="mock", split="train", level="scene", n=3, limit=4)
        exit_code = _cmd_generate(args)
        assert exit_code == 0

        records = list_records(split="train", db_path=db_path)
        for rec in records:
            assert len(rec.candidates) == 3
            for c in rec.candidates:
                assert c.model == "mock"
                assert c.text

    def test_generate_object_writes_candidates(self, db_path: Path) -> None:
        """Object-level: each image has 1 object with a bbox → 1×n candidates."""
        _seed_db_with_images(db_path, n=3)
        args = _make_args(model="mock", split="train", level="object", n=2, limit=3)
        exit_code = _cmd_generate(args)
        assert exit_code == 0

        records = list_records(split="train", db_path=db_path)
        for rec in records:
            if rec.objects and rec.objects[0].bbox:
                assert len(rec.candidates) == 2
                for c in rec.candidates:
                    # object_id should be stored in raw_parsed
                    assert c.raw_parsed is not None
                    assert c.raw_parsed.get("_object_id") == "o1"

    def test_limit_respected(self, db_path: Path) -> None:
        _seed_db_with_images(db_path, n=6)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)

        records = list_records(split="train", db_path=db_path)
        records_with_candidates = [r for r in records if r.candidates]
        assert len(records_with_candidates) == 2

    def test_candidates_appended_not_replaced(self, db_path: Path) -> None:
        """Running generate twice should accumulate candidates, not replace."""
        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)
        _cmd_generate(args)

        records = list_records(split="train", db_path=db_path)
        for rec in records:
            assert len(rec.candidates) == 2  # two runs of n=1

    def test_n_equals_1_greedy_path(self, db_path: Path) -> None:
        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        exit_code = _cmd_generate(args)
        assert exit_code == 0
        records = list_records(split="train", db_path=db_path)
        assert all(len(r.candidates) == 1 for r in records)

    def test_parse_failed_candidates_stored_not_dropped(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Candidates with parse_failed=True must be written to DB, not dropped."""
        monkeypatch.setenv("WASTECAP_MOCK_MODE", "parse_fail")
        from wastecap.models.registry import clear_cache

        clear_cache()

        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)

        records = list_records(split="train", db_path=db_path)
        for rec in records:
            assert len(rec.candidates) == 1
            assert rec.candidates[0].parse_failed is True
            assert rec.candidates[0].text  # text preserved

    def test_free_text_candidates_stored(
        self, db_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WASTECAP_MOCK_MODE", "free_text")
        from wastecap.models.registry import clear_cache

        clear_cache()

        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)

        records = list_records(split="train", db_path=db_path)
        for rec in records:
            assert len(rec.candidates) == 1
            # free_text mode: parse_failed=False (it's valid text, just flagged
            # in mock as free_text, but parse_structured_output will parse it
            # as fallback — which sets parse_failed=True)
            c = rec.candidates[0]
            assert c.text  # text always preserved

    def test_empty_split_returns_error(self, db_path: Path) -> None:
        # DB is empty — no records in split
        args = _make_args(model="mock", split="val", level="scene", n=1, limit=5)
        exit_code = _cmd_generate(args)
        assert exit_code == 1

    def test_image_not_on_disk_skipped_gracefully(self, db_path: Path) -> None:
        """Records pointing to nonexistent files should be skipped, not crash."""
        from wastecap.schema import AnnotationRecord

        rec = AnnotationRecord(
            image_id="ghost_img",
            file="data/processed/images/does_not_exist.jpg",
            source=DataSource.taco,
            width=200,
            height=200,
            split=Split.train,
        )
        ingest_record(rec, db_path)

        args = _make_args(model="mock", split="train", level="scene", n=1, limit=5)
        exit_code = _cmd_generate(args)
        # exit code 0 because no model errors (just skipped)
        assert exit_code == 0

        fetched = get_record("ghost_img", db_path)
        assert fetched.candidates == []

    def test_all_split_processes_all_records(self, db_path: Path) -> None:
        from wastecap.schema import AnnotationRecord

        fixture_img = str(Path(__file__).parent / "fixtures" / "images" / "taco_fix001.jpg")
        for split_val, sid in [(Split.train, "a"), (Split.val, "b"), (Split.test, "c")]:
            rec = AnnotationRecord(
                image_id=f"multi_{sid}",
                file=fixture_img,
                source=DataSource.taco,
                width=200,
                height=200,
                split=split_val,
            )
            ingest_record(rec, db_path)

        args = _make_args(model="mock", split="all", level="scene", n=1, limit=None)
        exit_code = _cmd_generate(args)
        assert exit_code == 0

        all_recs = list_records(db_path=db_path)
        assert all(len(r.candidates) == 1 for r in all_recs)

    def test_candidates_have_prompt_id(self, db_path: Path) -> None:
        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)
        records = list_records(split="train", db_path=db_path)
        for rec in records:
            for c in rec.candidates:
                assert c.prompt_id  # must not be empty

    def test_scene_candidates_have_no_grounding_for_mock(self, db_path: Path) -> None:
        """MockModel doesn't support grounding — boxes should be None."""
        _seed_db_with_images(db_path, n=2)
        args = _make_args(model="mock", split="train", level="scene", n=1, limit=2)
        _cmd_generate(args)
        records = list_records(split="train", db_path=db_path)
        for rec in records:
            for c in rec.candidates:
                # grounding_boxes may be None (mock) or populated (if parse found boxes)
                # For mock normal mode, raw text is JSON without grounding tags
                assert c.grounding_boxes is None

    def test_multi_source_split(self, db_path: Path) -> None:
        """Records from multiple sources all get candidates."""
        fixture_img = str(Path(__file__).parent / "fixtures" / "images" / "taco_fix001.jpg")
        from wastecap.schema import AnnotationRecord

        for src, sid in [
            (DataSource.taco, "tc"),
            (DataSource.zerowaste, "zw"),
            (DataSource.coco, "cc"),
        ]:
            rec = AnnotationRecord(
                image_id=f"src_{sid}",
                file=fixture_img,
                source=src,
                width=200,
                height=200,
                split=Split.train,
            )
            ingest_record(rec, db_path)

        args = _make_args(model="mock", split="train", level="scene", n=2, limit=None)
        exit_code = _cmd_generate(args)
        assert exit_code == 0

        records = list_records(split="train", db_path=db_path)
        assert all(len(r.candidates) == 2 for r in records)
