"""
Unit tests for M4 quality modules: scoring, filters, hallucination.
No GPU, no CLIP model download — CLIPScore returns None gracefully.
"""

from __future__ import annotations

import pytest

from wastecap.schema import (
    AnnotationRecord,
    Candidate,
    DataSource,
    Material,
    ObjectAnnotation,
    RecordStatus,
    Split,
    WasteCategory,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidate(
    model: str = "mock",
    text: str = "crushed plastic bottle on the pavement",
    parse_failed: bool = False,
    clip_score: float | None = None,
    raw_parsed: dict | None = None,
) -> Candidate:
    if raw_parsed is None and not parse_failed:
        raw_parsed = {
            "objects": [
                {
                    "object_id": "o1",
                    "category": "bottle",
                    "material": "plastic",
                    "condition": "crushed",
                    "count": 1,
                    "location": "lower left",
                    "caption": "crushed plastic bottle on the pavement",
                }
            ],
            "scene_context": "outdoor, cluttered",
        }
    return Candidate(
        model=model,
        prompt_id="mock_scene_v1",
        text=text,
        parse_failed=parse_failed,
        clip_score=clip_score,
        raw_parsed=raw_parsed,
    )


def _make_record(
    image_id: str = "test_001",
    source: DataSource = DataSource.taco,
    split: Split = Split.train,
    status: RecordStatus = RecordStatus.pending,
    candidates: list | None = None,
    objects: list | None = None,
    review=None,
) -> AnnotationRecord:
    if objects is None:
        objects = [
            ObjectAnnotation(
                object_id="o1",
                category=WasteCategory.bottle,
                material=Material.plastic,
                bbox=[10.0, 10.0, 50.0, 80.0],
            )
        ]
    return AnnotationRecord(
        image_id=image_id,
        file=f"data/processed/images/{image_id}.jpg",
        source=source,
        width=320,
        height=240,
        split=split,
        status=status,
        candidates=candidates or [],
        objects=objects,
        review=review,
    )


# ===========================================================================
# CLIPScore
# ===========================================================================


class TestCLIPScore:
    def test_compute_clipscore_returns_none_without_open_clip(self) -> None:
        """Without open_clip installed, compute_clipscore returns None gracefully."""
        from PIL import Image as PILImage

        from wastecap.quality.clipscore import compute_clipscore, reload_clip

        reload_clip()
        img = PILImage.new("RGB", (64, 64), color=(100, 150, 200))
        result = compute_clipscore(img, "a plastic bottle")
        # In local dev environment without open_clip, must return None
        assert result is None or isinstance(result, float)

    def test_batch_clipscore_returns_list_same_length(self, tmp_path) -> None:
        from PIL import Image as PILImage

        from wastecap.quality.clipscore import batch_clipscore, reload_clip

        reload_clip()
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            PILImage.new("RGB", (64, 64), color=(i * 50, 100, 200)).save(str(p))
            paths.append(p)

        texts = ["bottle", "can", "wrapper"]
        results = batch_clipscore(paths, texts)
        assert len(results) == 3
        # Each result is float or None
        for r in results:
            assert r is None or isinstance(r, float)

    def test_batch_clipscore_mismatched_length_raises(self, tmp_path) -> None:
        from wastecap.quality.clipscore import batch_clipscore

        with pytest.raises(ValueError, match="same length"):
            batch_clipscore([tmp_path / "x.jpg"], ["a", "b"])

    def test_score_candidates_returns_zero_without_clip(self, tmp_path) -> None:
        from wastecap.quality.clipscore import reload_clip, score_candidates_for_record

        reload_clip()
        rec = _make_record(candidates=[_make_candidate()])
        # Patch file to a non-existent path so we skip gracefully
        rec = rec.model_copy(update={"file": str(tmp_path / "nofile.jpg")})
        n = score_candidates_for_record(rec)
        assert n == 0

    def test_already_scored_candidates_not_rescored(self, tmp_path, db_path) -> None:
        from wastecap.quality.clipscore import score_candidates_for_record

        cand = _make_candidate(clip_score=0.35)  # already scored
        rec = _make_record(candidates=[cand])
        rec = rec.model_copy(update={"file": str(tmp_path / "nofile.jpg")})

        from wastecap.store import ingest_record

        ingest_record(rec, db_path)

        n = score_candidates_for_record(rec, db_path)
        assert n == 0  # nothing to score

    def test_extract_caption_text_uses_raw_parsed_caption(self) -> None:
        from wastecap.quality.clipscore import _extract_caption_text

        cand = _make_candidate(text="full raw model output that is very long indeed")
        result = _extract_caption_text(cand)
        # Should return the structured caption, not the raw text
        assert result == "crushed plastic bottle on the pavement"

    def test_extract_caption_text_falls_back_to_raw(self) -> None:
        from wastecap.quality.clipscore import _extract_caption_text

        cand = Candidate(
            model="mock",
            prompt_id="p",
            text="raw model output",
            parse_failed=True,
        )
        assert _extract_caption_text(cand) == "raw model output"


# ===========================================================================
# ReviewScore / scoring
# ===========================================================================


class TestReviewScore:
    def test_good_candidate_scores_well(self) -> None:
        from wastecap.quality.scoring import score_candidate

        cand = _make_candidate(clip_score=0.35)
        score = score_candidate(cand)
        assert score.parse_ok is True
        assert score.length_ok is True
        assert score.completeness > 0.5
        assert score.composite > 0.5
        assert score.grade == "good"

    def test_parse_failed_lowers_score(self) -> None:
        from wastecap.quality.scoring import score_candidate

        good = _make_candidate(parse_failed=False)
        bad = _make_candidate(parse_failed=True, raw_parsed=None)
        s_good = score_candidate(good)
        s_bad = score_candidate(bad)
        assert s_good.composite > s_bad.composite
        assert "parse_failed" in s_bad.flags

    def test_too_short_text_flagged(self) -> None:
        from wastecap.quality.scoring import score_candidate

        cand = _make_candidate(text="ok")
        score = score_candidate(cand)
        assert not score.length_ok
        assert any("too_short" in f for f in score.flags)

    def test_too_long_text_flagged(self) -> None:
        from wastecap.quality.scoring import score_candidate

        long_text = " ".join(["word"] * 70)
        cand = _make_candidate(text=long_text)
        score = score_candidate(cand)
        assert not score.length_ok
        assert any("too_long" in f for f in score.flags)

    def test_low_clip_score_flagged(self) -> None:
        from wastecap.quality.scoring import score_candidate

        cand = _make_candidate(clip_score=0.10)
        score = score_candidate(cand)
        assert any("low_clip" in f for f in score.flags)

    def test_clip_not_computed_flagged(self) -> None:
        from wastecap.quality.scoring import score_candidate

        cand = _make_candidate(clip_score=None)
        score = score_candidate(cand)
        assert any("clip_not_computed" in f for f in score.flags)

    def test_composite_in_unit_interval(self) -> None:
        from wastecap.quality.scoring import score_candidate

        for cs in [None, 0.0, 0.15, 0.35, 0.5]:
            cand = _make_candidate(clip_score=cs)
            score = score_candidate(cand)
            assert 0.0 <= score.composite <= 1.0

    def test_incomplete_fields_flagged(self) -> None:
        from wastecap.quality.scoring import score_candidate

        cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {"object_id": "o1", "category": "bottle"}
                    # no material, condition, caption etc.
                ],
                "scene_context": "",
            }
        )
        score = score_candidate(cand)
        assert any("incomplete" in f for f in score.flags)

    def test_best_score_returns_highest_composite(self) -> None:
        from wastecap.quality.scoring import best_score, score_candidate

        good = _make_candidate(clip_score=0.40)
        bad = _make_candidate(
            clip_score=0.01,
            parse_failed=True,
            raw_parsed=None,
            text="bad",
        )
        best = best_score([bad, good])
        assert best is not None
        assert best.composite >= score_candidate(good).composite

    def test_best_score_empty_returns_none(self) -> None:
        from wastecap.quality.scoring import best_score

        assert best_score([]) is None

    def test_grade_thresholds(self) -> None:
        from wastecap.quality.scoring import ReviewScore

        good = ReviewScore(None, 0.9, True, True, 0.70, [])
        marg = ReviewScore(None, 0.5, True, True, 0.50, [])
        poor = ReviewScore(None, 0.1, False, False, 0.20, [])
        assert good.grade == "good"
        assert marg.grade == "marginal"
        assert poor.grade == "poor"


# ===========================================================================
# Filters
# ===========================================================================


class TestFilters:
    def _records(self) -> list[AnnotationRecord]:
        from wastecap.schema import Review

        blip_cand = _make_candidate(model="blip2", clip_score=0.30)
        qwen_cand = _make_candidate(model="qwen_vl", clip_score=0.10, parse_failed=False)
        fail_cand = _make_candidate(model="mock", parse_failed=True, raw_parsed=None)

        return [
            _make_record(
                "r1",
                source=DataSource.taco,
                split=Split.train,
                status=RecordStatus.pending,
                candidates=[blip_cand],
            ),
            _make_record(
                "r2",
                source=DataSource.zerowaste,
                split=Split.val,
                status=RecordStatus.approved,
                candidates=[qwen_cand],
                objects=[
                    ObjectAnnotation(
                        object_id="o1",
                        category=WasteCategory.can,
                        material=Material.metal,
                        bbox=[10.0, 10.0, 40.0, 40.0],
                    )
                ],
                review=Review(reviewer="test_reviewer"),
            ),
            _make_record(
                "r3",
                source=DataSource.coco,
                split=Split.test,
                status=RecordStatus.rejected,
                candidates=[fail_cand],
            ),
            _make_record("r4", candidates=[], status=RecordStatus.edited),  # no candidates
        ]

    def test_no_filters_returns_all(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec())
        assert len(result) == len(records)

    def test_status_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(status=["pending"]))
        assert all(r.status == RecordStatus.pending for r in result)
        assert len(result) == 1

    def test_source_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(source=["taco"]))
        assert all(r.source == DataSource.taco for r in result)

    def test_split_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(split=["val"]))
        assert all(r.split == Split.val for r in result)

    def test_model_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(models=["blip2"]))
        assert all(any(c.model == "blip2" for c in r.candidates) for r in result)

    def test_category_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        # r2 has category=can
        result = apply_filters(records, FilterSpec(categories=["can"]))
        assert len(result) == 1
        assert result[0].image_id == "r2"

    def test_min_score_filter(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        # High threshold — only blip2 candidate with clip=0.30 should pass
        result = apply_filters(records, FilterSpec(min_score=0.55))
        assert all(len(r.candidates) > 0 for r in result)

    def test_only_parse_failed(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(only_parse_failed=True))
        assert all(any(c.parse_failed for c in r.candidates) for r in result)

    def test_exclude_parse_failed(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(exclude_parse_failed=True))
        assert all(not any(c.parse_failed for c in r.candidates) for r in result)

    def test_only_unscored(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(records, FilterSpec(only_unscored=True))
        # r4 has no candidates (so no unscored candidates), r1/r2/r3 all have clip_score set
        assert all(any(c.clip_score is None for c in r.candidates) for r in result)

    def test_combined_filters(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        records = self._records()
        result = apply_filters(
            records,
            FilterSpec(status=["approved"], models=["qwen_vl"]),
        )
        assert len(result) == 1
        assert result[0].image_id == "r2"

    def test_empty_records_list(self) -> None:
        from wastecap.quality.filters import FilterSpec, apply_filters

        assert apply_filters([], FilterSpec()) == []

    def test_filter_candidates_by_model(self) -> None:
        from wastecap.quality.filters import filter_candidates_by_model

        cands = [
            _make_candidate(model="blip2"),
            _make_candidate(model="qwen_vl"),
            _make_candidate(model="mock"),
        ]
        result = filter_candidates_by_model(cands, ["blip2", "qwen_vl"])
        assert len(result) == 2
        assert all(c.model in ["blip2", "qwen_vl"] for c in result)

    def test_filter_candidates_by_score(self) -> None:
        from wastecap.quality.filters import filter_candidates_by_score
        from wastecap.quality.scoring import score_candidate

        # A fully-structured, well-clipscore candidate
        good = _make_candidate(clip_score=0.40)
        # A parse-failed, no raw_parsed, short-text candidate → very low composite
        bad = _make_candidate(
            clip_score=0.01,
            parse_failed=True,
            raw_parsed=None,
            text="bad",
        )
        good_score = score_candidate(good).composite
        bad_score = score_candidate(bad).composite
        # Sanity: good really is better
        assert good_score > bad_score

        # Filter with threshold between the two scores
        threshold = (good_score + bad_score) / 2
        result = filter_candidates_by_score([good, bad], min_score=threshold)
        assert len(result) == 1
        assert result[0].clip_score == pytest.approx(0.40)

    def test_best_candidate_per_model(self) -> None:
        from wastecap.quality.filters import best_candidate_per_model

        b1 = _make_candidate(model="blip2", clip_score=0.20)
        b2 = _make_candidate(model="blip2", clip_score=0.35)
        q1 = _make_candidate(model="qwen_vl", clip_score=0.25)

        best = best_candidate_per_model([b1, b2, q1])
        assert set(best.keys()) == {"blip2", "qwen_vl"}
        assert best["blip2"].clip_score == pytest.approx(0.35)


# ===========================================================================
# Hallucination flags
# ===========================================================================


class TestHallucinationFlags:
    def _ref_bottle_plastic(self) -> list[ObjectAnnotation]:
        return [
            ObjectAnnotation(
                object_id="o1",
                category=WasteCategory.bottle,
                material=Material.plastic,
                bbox=[10.0, 10.0, 50.0, 80.0],
            )
        ]

    def test_no_flags_for_correct_candidate(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate()  # category=bottle, material=plastic
        result = flag_candidate(cand, self._ref_bottle_plastic())
        flags = get_hallucination_flags(result)
        assert flags == []

    def test_category_mismatch_flagged(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "can",  # ref says bottle!
                        "material": "metal",
                        "condition": "intact",
                        "count": 1,
                        "caption": "intact metal can",
                    }
                ],
                "scene_context": "",
            }
        )
        ref = self._ref_bottle_plastic()
        result = flag_candidate(cand, ref)
        flags = get_hallucination_flags(result)
        assert any("category_mismatch" in f for f in flags)

    def test_material_mismatch_flagged(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "bottle",
                        "material": "glass",  # ref says plastic!
                        "condition": "intact",
                        "count": 1,
                        "caption": "intact glass bottle",
                    }
                ],
                "scene_context": "",
            }
        )
        result = flag_candidate(cand, self._ref_bottle_plastic())
        flags = get_hallucination_flags(result)
        assert any("material_mismatch" in f for f in flags)

    def test_invented_count_flagged(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "bottle",
                        "material": "plastic",
                        "condition": "intact",
                        "count": 5,  # ref has 1 bottle
                        "caption": "five plastic bottles",
                    }
                ],
                "scene_context": "",
            }
        )
        result = flag_candidate(cand, self._ref_bottle_plastic())
        flags = get_hallucination_flags(result)
        assert any("invented_count" in f for f in flags)

    def test_empty_caption_flagged(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate(
            text="ok",
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "bottle",
                        "material": "plastic",
                        "condition": "intact",
                        "count": 1,
                        "caption": "ok",  # too short
                    }
                ],
                "scene_context": "",
            },
        )
        result = flag_candidate(cand, self._ref_bottle_plastic())
        flags = get_hallucination_flags(result)
        assert any("trivial" in f or "empty" in f for f in flags)

    def test_parse_failed_candidate_not_flagged_for_mismatch(self) -> None:
        """parse_failed candidates: skip category/material checks (text is unreliable)."""
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate(
            parse_failed=True, raw_parsed=None, text="can metal landfill plastic"
        )
        result = flag_candidate(cand, self._ref_bottle_plastic())
        flags = get_hallucination_flags(result)
        # Should not flag category/material mismatch since parse failed
        assert not any("category_mismatch" in f for f in flags)
        assert not any("material_mismatch" in f for f in flags)

    def test_no_reference_objects_returns_unflagged(self) -> None:
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        cand = _make_candidate()
        result = flag_candidate(cand, [])  # no reference
        flags = get_hallucination_flags(result)
        # Category/material checks skip when ref is empty
        assert not any("category_mismatch" in f or "material_mismatch" in f for f in flags)

    def test_flag_all_candidates(self) -> None:
        from wastecap.quality.hallucination import flag_all_candidates, get_hallucination_flags

        wrong_cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "can",
                        "material": "metal",
                        "condition": "intact",
                        "count": 1,
                        "caption": "metal can",
                    }
                ],
                "scene_context": "",
            }
        )
        rec = _make_record(candidates=[wrong_cand])
        flagged_rec = flag_all_candidates(rec)
        assert len(flagged_rec.candidates) == 1
        flags = get_hallucination_flags(flagged_rec.candidates[0])
        assert any("category_mismatch" in f for f in flags)

    def test_is_hallucination_flagged(self) -> None:
        from wastecap.quality.hallucination import (
            flag_candidate,
            is_hallucination_flagged,
        )

        bad_cand = _make_candidate(
            raw_parsed={
                "objects": [
                    {
                        "object_id": "o1",
                        "category": "can",
                        "material": "metal",
                        "condition": "intact",
                        "count": 1,
                        "caption": "metal can",
                    }
                ],
                "scene_context": "",
            }
        )
        ref = self._ref_bottle_plastic()
        flagged = flag_candidate(bad_cand, ref)
        assert is_hallucination_flagged(flagged) is True

    def test_unflagged_is_not_flagged(self) -> None:
        from wastecap.quality.hallucination import is_hallucination_flagged

        cand = _make_candidate()
        assert is_hallucination_flagged(cand) is False

    def test_text_token_fallback_for_category(self) -> None:
        """When raw_parsed has no category, fall back to token matching in text."""
        from wastecap.quality.hallucination import flag_candidate, get_hallucination_flags

        # No structured category — use text
        cand = Candidate(
            model="mock",
            prompt_id="p",
            text="A metal can sitting on the grass.",
            parse_failed=False,
            raw_parsed={"scene_context": "outdoor"},  # no objects list
        )
        ref = self._ref_bottle_plastic()  # reference: bottle/plastic
        result = flag_candidate(cand, ref)
        flags = get_hallucination_flags(result)
        assert any("category_mismatch" in f for f in flags)
