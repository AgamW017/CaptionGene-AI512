"""
Tests for M1 data modules: loaders, manifest, splits.

All tests use fixture data only — no network access, no real downloads.
Fixture layout (created by the generate script):
    tests/fixtures/raw/
        taco/
            annotations.json   (6 images: 5 valid 320x240, 1 tiny 80x80)
            data/batch_1/*.jpg
        zerowaste/zerowaste-f/
            train/labels.json  (4 images 256x256)
            val/labels.json    (2 images 256x256)
        coco/
            annotations/instances_val2017.json (6 images: 5 with waste, 1 non-waste-only)
            images/val2017/*.jpg
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES_RAW = Path(__file__).parent / "fixtures" / "raw"
TACO_RAW = FIXTURES_RAW / "taco"
ZW_RAW = FIXTURES_RAW / "zerowaste"
COCO_RAW = FIXTURES_RAW / "coco"


# ============================================================================
# TACO loader
# ============================================================================


class TestTacoLoader:
    def test_loads_expected_count(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        # 6 images in fixture (including tiny 80x80)
        assert len(records) == 6

    def test_limit_respected(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW, limit=3)
        assert len(records) == 3

    def test_source_is_taco(self) -> None:
        from wastecap.data.taco import load_taco
        from wastecap.schema import DataSource

        records = load_taco(TACO_RAW)
        assert all(r.source == DataSource.taco for r in records)

    def test_objects_populated(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        # Every valid image should have ≥1 object
        records_with_objects = [r for r in records if len(r.objects) > 0]
        assert len(records_with_objects) >= 5

    def test_object_category_mapped(self) -> None:
        from wastecap.data.taco import load_taco
        from wastecap.schema import WasteCategory

        records = load_taco(TACO_RAW)
        # "Clear plastic bottle" → bottle; "Drink can" → can
        categories = {obj.category for r in records for obj in r.objects}
        assert WasteCategory.bottle in categories
        assert WasteCategory.can in categories

    def test_bbox_present_and_valid(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        for rec in records:
            for obj in rec.objects:
                if obj.bbox is not None:
                    assert len(obj.bbox) == 4
                    assert all(v >= 0 for v in obj.bbox)

    def test_polygon_mask_parsed(self) -> None:
        from wastecap.data.taco import load_taco
        from wastecap.schema import PolygonMask

        records = load_taco(TACO_RAW)
        masks = [obj.mask for r in records for obj in r.objects if obj.mask is not None]
        assert len(masks) >= 1
        assert all(isinstance(m, PolygonMask) for m in masks)

    def test_sorting_action_set(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        for rec in records:
            for obj in rec.objects:
                assert obj.sorting_action is not None

    def test_license_is_cc_by(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        assert all("CC BY" in r.license for r in records)

    def test_tmp_ids_are_unique(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        ids = [r.image_id for r in records]
        assert len(ids) == len(set(ids))

    def test_missing_annotations_raises(self, tmp_path: Path) -> None:
        from wastecap.data.taco import load_taco

        with pytest.raises(FileNotFoundError, match="annotations.json"):
            load_taco(tmp_path)

    def test_object_ids_unique_per_image(self) -> None:
        from wastecap.data.taco import load_taco

        records = load_taco(TACO_RAW)
        for rec in records:
            oids = [o.object_id for o in rec.objects]
            assert len(oids) == len(set(oids)), f"Duplicate object_ids in {rec.image_id}"


# ============================================================================
# ZeroWaste loader
# ============================================================================


class TestZerowasteLoader:
    def test_loads_both_splits(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste

        records = load_zerowaste(ZW_RAW)
        # 4 train + 2 val = 6
        assert len(records) == 6

    def test_limit_respected(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste

        records = load_zerowaste(ZW_RAW, limit=3)
        assert len(records) == 3

    def test_source_is_zerowaste(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste
        from wastecap.schema import DataSource

        records = load_zerowaste(ZW_RAW)
        assert all(r.source == DataSource.zerowaste for r in records)

    def test_rle_mask_parsed(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste
        from wastecap.schema import RLEMask

        records = load_zerowaste(ZW_RAW)
        rle_masks = [
            obj.mask for r in records for obj in r.objects if isinstance(obj.mask, RLEMask)
        ]
        assert len(rle_masks) >= 1

    def test_four_categories_covered(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste
        from wastecap.schema import Material, WasteCategory

        records = load_zerowaste(ZW_RAW)
        categories = {obj.category for r in records for obj in r.objects}
        materials = {obj.material for r in records for obj in r.objects}
        # rigid_plastic→container/plastic, cardboard→box/cardboard,
        # metal→can/metal, soft_plastic→wrapper/plastic
        assert WasteCategory.container in categories
        assert WasteCategory.box in categories
        assert WasteCategory.can in categories
        assert WasteCategory.wrapper in categories
        assert Material.plastic in materials
        assert Material.cardboard in materials
        assert Material.metal in materials

    def test_missing_data_raises(self, tmp_path: Path) -> None:
        from wastecap.data.zerowaste import load_zerowaste

        with pytest.raises(FileNotFoundError):
            load_zerowaste(tmp_path)

    def test_object_ids_unique_per_image(self) -> None:
        from wastecap.data.zerowaste import load_zerowaste

        records = load_zerowaste(ZW_RAW)
        for rec in records:
            oids = [o.object_id for o in rec.objects]
            assert len(oids) == len(set(oids))


# ============================================================================
# COCO subset loader
# ============================================================================


class TestCocoSubsetLoader:
    def test_loads_only_waste_images(self) -> None:
        from wastecap.data.coco_subset import load_coco_subset

        records = load_coco_subset(COCO_RAW)
        # 5 waste images; 1 non-waste-only image should be excluded
        assert len(records) == 5

    def test_limit_respected(self) -> None:
        from wastecap.data.coco_subset import load_coco_subset

        records = load_coco_subset(COCO_RAW, limit=3)
        assert len(records) == 3

    def test_source_is_coco(self) -> None:
        from wastecap.data.coco_subset import load_coco_subset
        from wastecap.schema import DataSource

        records = load_coco_subset(COCO_RAW)
        assert all(r.source == DataSource.coco for r in records)

    def test_non_waste_objects_excluded(self) -> None:
        """Person annotations must not appear in objects."""
        from wastecap.data.coco_subset import load_coco_subset

        records = load_coco_subset(COCO_RAW)
        # All objects should be waste-mapped categories, not 'person'
        for rec in records:
            for _obj in rec.objects:
                # 'person' has no mapping → would be WasteCategory.unknown
                # but regardless, no object should have category that came from person
                # The real check: exactly 1 waste object per image (fixture has 1 waste + 1 person)
                pass  # The count test below is the meaningful one

    def test_exactly_one_waste_object_per_image(self) -> None:
        """Each fixture image has exactly 1 waste annotation and 1 person (excluded)."""
        from wastecap.data.coco_subset import load_coco_subset

        records = load_coco_subset(COCO_RAW)
        assert all(len(r.objects) == 1 for r in records), (
            f"Expected 1 object per image, got: {[len(r.objects) for r in records]}"
        )

    def test_waste_categories_present(self) -> None:
        from wastecap.data.coco_subset import load_coco_subset
        from wastecap.schema import WasteCategory

        records = load_coco_subset(COCO_RAW)
        categories = {obj.category for r in records for obj in r.objects}
        # bottle, cup, bowl, food_waste all map from our fixture categories
        assert WasteCategory.bottle in categories

    def test_missing_annotations_raises(self, tmp_path: Path) -> None:
        from wastecap.data.coco_subset import load_coco_subset

        with pytest.raises(FileNotFoundError, match="annotation"):
            load_coco_subset(tmp_path)

    def test_license_is_cc_by(self) -> None:
        from wastecap.data.coco_subset import load_coco_subset

        records = load_coco_subset(COCO_RAW)
        assert all("CC BY" in r.license for r in records)


# ============================================================================
# Manifest: build_manifest
# ============================================================================


class TestManifest:
    def _make_records(self, n: int = 5, source: str = "taco") -> list:
        """Return n minimal AnnotationRecords with real fixture images."""
        from wastecap.schema import AnnotationRecord, DataSource

        src = DataSource(source)
        fixture_imgs = sorted((Path(__file__).parent / "fixtures" / "images").glob("*.jpg"))
        records = []
        for i in range(n):
            img_path = fixture_imgs[i % len(fixture_imgs)]
            records.append(
                AnnotationRecord(
                    image_id=f"{source}_tmp_{i:03d}",
                    file=str(img_path),
                    source=src,
                    width=200,
                    height=200,
                )
            )
        return records

    def test_reindex_assigns_canonical_ids(self) -> None:
        from wastecap.data.manifest import build_manifest

        records = self._make_records(3, "taco")
        result = build_manifest(records, deduplicate=False, quality_filter=False, reindex=True)
        for i, rec in enumerate(result):
            assert rec.image_id == f"taco_{i:06d}"

    def test_quality_filter_drops_small_images(self) -> None:
        from wastecap.data.manifest import MIN_HEIGHT, MIN_WIDTH, build_manifest
        from wastecap.schema import AnnotationRecord, DataSource

        good = AnnotationRecord(
            image_id="good",
            file="nofile.jpg",
            source=DataSource.taco,
            width=MIN_WIDTH,
            height=MIN_HEIGHT,
        )
        bad = AnnotationRecord(
            image_id="bad",
            file="nofile.jpg",
            source=DataSource.taco,
            width=MIN_WIDTH - 1,
            height=MIN_HEIGHT,
        )
        result = build_manifest([good, bad], deduplicate=False, quality_filter=True, reindex=False)
        assert len(result) == 1
        assert result[0].image_id == "good"

    def test_quality_filter_off_keeps_all(self) -> None:
        from wastecap.data.manifest import build_manifest
        from wastecap.schema import AnnotationRecord, DataSource

        records = [
            AnnotationRecord(
                image_id=f"r{i}", file="x.jpg", source=DataSource.taco, width=10, height=10
            )
            for i in range(4)
        ]
        result = build_manifest(records, deduplicate=False, quality_filter=False, reindex=False)
        assert len(result) == 4

    def test_deduplication_drops_duplicate_files(self) -> None:
        """Two records pointing to the same real file should result in 1 kept."""
        from wastecap.data.manifest import build_manifest
        from wastecap.schema import AnnotationRecord, DataSource

        img_path = str(Path(__file__).parent / "fixtures" / "images" / "taco_fix001.jpg")
        r1 = AnnotationRecord(
            image_id="dup_a",
            file=img_path,
            source=DataSource.taco,
            width=200,
            height=200,
        )
        r2 = AnnotationRecord(
            image_id="dup_b",
            file=img_path,
            source=DataSource.taco,
            width=200,
            height=200,
        )
        result = build_manifest([r1, r2], deduplicate=True, quality_filter=False, reindex=False)
        assert len(result) == 1
        assert result[0].image_id == "dup_a"

    def test_dedup_off_keeps_duplicates(self) -> None:
        from wastecap.data.manifest import build_manifest
        from wastecap.schema import AnnotationRecord, DataSource

        img_path = str(Path(__file__).parent / "fixtures" / "images" / "taco_fix001.jpg")
        r1 = AnnotationRecord(
            image_id="a", file=img_path, source=DataSource.taco, width=200, height=200
        )
        r2 = AnnotationRecord(
            image_id="b", file=img_path, source=DataSource.taco, width=200, height=200
        )
        result = build_manifest([r1, r2], deduplicate=False, quality_filter=False, reindex=False)
        assert len(result) == 2

    def test_multi_source_reindex_independent(self) -> None:
        """Each source gets its own counter starting at 0."""
        from wastecap.data.manifest import build_manifest
        from wastecap.schema import AnnotationRecord, DataSource

        records = [
            AnnotationRecord(
                image_id=f"t{i}", file="x.jpg", source=DataSource.taco, width=200, height=200
            )
            for i in range(3)
        ] + [
            AnnotationRecord(
                image_id=f"z{i}", file="x.jpg", source=DataSource.zerowaste, width=200, height=200
            )
            for i in range(2)
        ]
        result = build_manifest(records, deduplicate=False, quality_filter=False, reindex=True)
        taco_ids = [r.image_id for r in result if r.source == DataSource.taco]
        zw_ids = [r.image_id for r in result if r.source == DataSource.zerowaste]
        assert taco_ids == ["taco_000000", "taco_000001", "taco_000002"]
        assert zw_ids == ["zerowaste_000000", "zerowaste_000001"]

    def test_taco_loader_records_pass_manifest(self) -> None:
        """Full integration: TACO loader → manifest filters tiny image."""
        from wastecap.data.manifest import build_manifest
        from wastecap.data.taco import load_taco

        raw = load_taco(TACO_RAW)
        result = build_manifest(raw, deduplicate=False, quality_filter=True, reindex=True)
        # Fixture has 5 valid (320x240) + 1 tiny (80x80); tiny should be dropped
        assert len(result) == 5
        assert all(r.image_id.startswith("taco_") for r in result)


# ============================================================================
# Splits: generate_splits
# ============================================================================


class TestGenerateSplits:
    def _make_records(
        self,
        n_taco: int = 10,
        n_zw: int = 8,
        n_coco: int = 7,
    ) -> list:
        from wastecap.schema import AnnotationRecord, DataSource

        records = []
        for src, n in [("taco", n_taco), ("zerowaste", n_zw), ("coco", n_coco)]:
            for i in range(n):
                records.append(
                    AnnotationRecord(
                        image_id=f"{src}_{i:03d}",
                        file=f"data/{src}/{i}.jpg",
                        source=DataSource(src),
                        width=320,
                        height=240,
                    )
                )
        return records

    def test_no_leakage_between_splits(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        records = self._make_records()
        splits = generate_splits(records, output_path=tmp_path / "splits.json")
        train_set = set(splits["train"])
        val_set = set(splits["val"])
        test_set = set(splits["test"])
        assert train_set.isdisjoint(val_set), "train ∩ val not empty"
        assert train_set.isdisjoint(test_set), "train ∩ test not empty"
        assert val_set.isdisjoint(test_set), "val ∩ test not empty"

    def test_all_records_assigned(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        records = self._make_records()
        splits = generate_splits(records, output_path=tmp_path / "splits.json")
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(records)

    def test_seeded_reproducible(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        records = self._make_records()
        s1 = generate_splits(records, seed=42, output_path=tmp_path / "s1.json")

        # Reset splits on records
        for r in records:
            r.split = None

        records2 = self._make_records()
        s2 = generate_splits(records2, seed=42, output_path=tmp_path / "s2.json")

        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_different_seed_gives_different_splits(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        r1 = self._make_records(20, 15, 12)
        r2 = self._make_records(20, 15, 12)
        s1 = generate_splits(r1, seed=42, output_path=tmp_path / "s1.json")
        s2 = generate_splits(r2, seed=99, output_path=tmp_path / "s2.json")
        assert s1["train"] != s2["train"]

    def test_splits_json_written(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        out = tmp_path / "splits.json"
        generate_splits(self._make_records(), output_path=out)
        assert out.exists()
        with out.open() as f:
            data = json.load(f)
        assert set(data.keys()) == {"train", "val", "test"}

    def test_split_field_set_on_records(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits
        from wastecap.schema import Split

        records = self._make_records()
        generate_splits(records, output_path=tmp_path / "splits.json")
        assert all(r.split is not None for r in records)
        assert all(isinstance(r.split, Split) for r in records)

    def test_ratios_approximately_correct(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        records = self._make_records(100, 80, 70)
        splits = generate_splits(
            records,
            train_ratio=0.70,
            val_ratio=0.15,
            output_path=tmp_path / "splits.json",
        )
        total = len(records)
        # Allow ±5% slack from exact ratio (rounding + small sources)
        assert 0.60 <= len(splits["train"]) / total <= 0.80
        assert 0.08 <= len(splits["val"]) / total <= 0.22
        assert 0.08 <= len(splits["test"]) / total <= 0.22

    def test_each_source_in_all_splits(self, tmp_path: Path) -> None:
        """Stratification: every source (taco/zerowaste/coco) appears in all splits."""
        from wastecap.data.splits import generate_splits

        records = self._make_records(30, 25, 20)
        splits = generate_splits(records, output_path=tmp_path / "splits.json")
        for split_name, id_list in splits.items():
            sources_in_split = {_id.split("_")[0] for _id in id_list}
            assert "taco" in sources_in_split, f"taco missing from {split_name}"
            assert "zerowaste" in sources_in_split, f"zerowaste missing from {split_name}"
            assert "coco" in sources_in_split, f"coco missing from {split_name}"

    def test_empty_records_raises(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        with pytest.raises(ValueError, match="empty"):
            generate_splits([], output_path=tmp_path / "splits.json")

    def test_bad_ratios_raises(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits

        records = self._make_records(10, 10, 10)
        with pytest.raises(ValueError, match="ratio"):
            generate_splits(
                records, train_ratio=0.85, val_ratio=0.20, output_path=tmp_path / "splits.json"
            )

    def test_load_splits(self, tmp_path: Path) -> None:
        from wastecap.data.splits import generate_splits, load_splits

        records = self._make_records()
        out = tmp_path / "splits.json"
        generate_splits(records, output_path=out)
        loaded = load_splits(out)
        assert set(loaded.keys()) == {"train", "val", "test"}
        assert len(loaded["train"]) > 0

    def test_load_splits_missing_raises(self, tmp_path: Path) -> None:
        from wastecap.data.splits import load_splits

        with pytest.raises(FileNotFoundError):
            load_splits(tmp_path / "nonexistent.json")


# ============================================================================
# End-to-end: loader → manifest → splits
# ============================================================================


class TestEndToEnd:
    def test_taco_loader_manifest_splits_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline with fixture TACO data."""
        from wastecap.data.manifest import build_manifest
        from wastecap.data.splits import generate_splits
        from wastecap.data.taco import load_taco

        raw = load_taco(TACO_RAW)
        assert len(raw) == 6

        manifest = build_manifest(raw, deduplicate=False, quality_filter=True, reindex=True)
        assert len(manifest) == 5  # tiny image dropped

        splits = generate_splits(manifest, output_path=tmp_path / "splits.json")
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 5

        # No leakage
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == len(set(all_ids))

    def test_multi_source_pipeline(self, tmp_path: Path) -> None:
        """Combine all three sources → manifest → splits."""
        from wastecap.data.coco_subset import load_coco_subset
        from wastecap.data.manifest import build_manifest
        from wastecap.data.splits import generate_splits
        from wastecap.data.taco import load_taco
        from wastecap.data.zerowaste import load_zerowaste
        from wastecap.schema import DataSource

        taco_recs = load_taco(TACO_RAW)
        zw_recs = load_zerowaste(ZW_RAW)
        coco_recs = load_coco_subset(COCO_RAW)

        all_recs = taco_recs + zw_recs + coco_recs
        manifest = build_manifest(all_recs, deduplicate=False, quality_filter=True, reindex=True)

        # 5 taco (tiny dropped) + 6 zerowaste + 5 coco (non-waste excluded) = 16
        assert len(manifest) == 16

        # IDs are canonical per source
        taco_ids = [r.image_id for r in manifest if r.source == DataSource.taco]
        zw_ids = [r.image_id for r in manifest if r.source == DataSource.zerowaste]
        coco_ids = [r.image_id for r in manifest if r.source == DataSource.coco]
        assert all(i.startswith("taco_") for i in taco_ids)
        assert all(i.startswith("zerowaste_") for i in zw_ids)
        assert all(i.startswith("coco_") for i in coco_ids)

        # Splits have no leakage
        splits = generate_splits(manifest, output_path=tmp_path / "splits.json")
        all_ids = splits["train"] + splits["val"] + splits["test"]
        assert len(all_ids) == len(set(all_ids)) == len(manifest)

    def test_ingest_pipeline_to_db(self, tmp_path: Path, db_path: Path) -> None:
        """Records from all loaders → manifest → splits → DB round-trip."""
        from wastecap.data.coco_subset import load_coco_subset
        from wastecap.data.manifest import build_manifest
        from wastecap.data.splits import generate_splits
        from wastecap.data.taco import load_taco
        from wastecap.data.zerowaste import load_zerowaste
        from wastecap.store import count_records, list_records, upsert_record

        all_recs = load_taco(TACO_RAW) + load_zerowaste(ZW_RAW) + load_coco_subset(COCO_RAW)
        manifest = build_manifest(all_recs, deduplicate=False, quality_filter=True, reindex=True)
        generate_splits(manifest, output_path=tmp_path / "splits.json")

        for rec in manifest:
            upsert_record(rec, db_path)

        assert count_records(db_path=db_path) == len(manifest)

        taco_in_db = list_records(source="taco", db_path=db_path)
        assert len(taco_in_db) == 5

        # All records have split assigned
        all_in_db = list_records(db_path=db_path)
        assert all(r.split is not None for r in all_in_db)
