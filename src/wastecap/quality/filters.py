"""
Quality filters for WasteCap AnnotationRecords and Candidates.

All filters are pure functions — they never write to the DB.
They operate on in-memory lists (already loaded from the DB) so
the caller decides when to page/limit the DB query.

FilterSpec
----------
A dataclass collecting all optional filter criteria. Pass it to
apply_filters() which composes the individual predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wastecap.quality.scoring import best_score, score_candidate
from wastecap.schema import AnnotationRecord, Candidate

# ---------------------------------------------------------------------------
# FilterSpec
# ---------------------------------------------------------------------------


@dataclass
class FilterSpec:
    """Declarative filter specification.

    All fields are optional. Unset fields (None / empty list) are ignored.
    """

    # Record-level filters
    status: list[str] = field(default_factory=list)
    source: list[str] = field(default_factory=list)
    split: list[str] = field(default_factory=list)

    # Candidate-level filters (applied to any candidate in the record)
    models: list[str] = field(default_factory=list)

    # Object/category filters (any object in the record)
    categories: list[str] = field(default_factory=list)

    # Score range (based on best composite score across candidates)
    min_score: float | None = None
    max_score: float | None = None

    # Flags
    only_flagged: bool = False  # only records with ≥1 hallucination flag
    only_parse_failed: bool = False  # only records where any candidate parse_failed
    only_unscored: bool = False  # only records where any candidate clip_score is None
    exclude_parse_failed: bool = False  # exclude records with any parse_failed candidate


# ---------------------------------------------------------------------------
# Individual predicates
# ---------------------------------------------------------------------------


def _matches_status(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.status:
        return True
    return rec.status.value in spec.status


def _matches_source(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.source:
        return True
    return rec.source.value in spec.source


def _matches_split(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.split:
        return True
    return rec.split is not None and rec.split.value in spec.split


def _matches_model(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.models:
        return True
    candidate_models = {c.model for c in rec.candidates}
    return bool(candidate_models & set(spec.models))


def _matches_category(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.categories:
        return True
    obj_categories = {o.category.value for o in rec.objects}
    return bool(obj_categories & set(spec.categories))


def _matches_score_range(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if spec.min_score is None and spec.max_score is None:
        return True
    bs = best_score(rec.candidates)
    if bs is None:
        # No candidates → only include if min_score is None (no lower bound)
        return spec.min_score is None
    if spec.min_score is not None and bs.composite < spec.min_score:
        return False
    return not (spec.max_score is not None and bs.composite > spec.max_score)


def _matches_flagged(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.only_flagged:
        return True
    # "flagged" = review.hallucination=True OR any candidate has hallucination flag
    if rec.review and rec.review.hallucination:
        return True
    return any("hallucination" in flag for c in rec.candidates for flag in _get_candidate_flags(c))


def _get_candidate_flags(cand: Candidate) -> list[str]:
    """Retrieve hallucination flags from candidate raw_parsed."""
    if cand.raw_parsed and isinstance(cand.raw_parsed, dict):
        return cand.raw_parsed.get("_hallucination_flags", [])
    return []


def _matches_parse_failed(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.only_parse_failed:
        return True
    return any(c.parse_failed for c in rec.candidates)


def _matches_unscored(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.only_unscored:
        return True
    return any(c.clip_score is None for c in rec.candidates)


def _matches_exclude_parse_failed(rec: AnnotationRecord, spec: FilterSpec) -> bool:
    if not spec.exclude_parse_failed:
        return True
    return not any(c.parse_failed for c in rec.candidates)


# ---------------------------------------------------------------------------
# Composed filter
# ---------------------------------------------------------------------------


def apply_filters(
    records: list[AnnotationRecord],
    spec: FilterSpec,
) -> list[AnnotationRecord]:
    """Return the subset of *records* matching all criteria in *spec*.

    Predicates are short-circuit evaluated in cheapest-first order:
    status/source/split are O(1) string checks; score requires scoring
    all candidates and runs last.
    """
    result: list[AnnotationRecord] = []
    for rec in records:
        if not _matches_status(rec, spec):
            continue
        if not _matches_source(rec, spec):
            continue
        if not _matches_split(rec, spec):
            continue
        if not _matches_model(rec, spec):
            continue
        if not _matches_category(rec, spec):
            continue
        if not _matches_parse_failed(rec, spec):
            continue
        if not _matches_exclude_parse_failed(rec, spec):
            continue
        if not _matches_unscored(rec, spec):
            continue
        if not _matches_flagged(rec, spec):
            continue
        if not _matches_score_range(rec, spec):
            continue
        result.append(rec)
    return result


# ---------------------------------------------------------------------------
# Candidate-level filtering (for Compare page)
# ---------------------------------------------------------------------------


def filter_candidates_by_model(
    candidates: list[Candidate],
    models: list[str],
) -> list[Candidate]:
    """Return candidates whose model is in *models*."""
    if not models:
        return candidates
    return [c for c in candidates if c.model in models]


def filter_candidates_by_score(
    candidates: list[Candidate],
    min_score: float | None = None,
    max_score: float | None = None,
) -> list[Candidate]:
    """Return candidates whose composite ReviewScore falls within [min, max]."""
    if min_score is None and max_score is None:
        return candidates
    result: list[Candidate] = []
    for cand in candidates:
        s = score_candidate(cand)
        if min_score is not None and s.composite < min_score:
            continue
        if max_score is not None and s.composite > max_score:
            continue
        result.append(cand)
    return result


def best_candidate_per_model(
    candidates: list[Candidate],
) -> dict[str, Candidate]:
    """Return {model: best_candidate} by highest composite score.

    Used by the Compare page to pick one representative candidate per model.
    """
    from collections import defaultdict

    by_model: dict[str, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_model[c.model].append(c)

    result: dict[str, Candidate] = {}
    for model, cands in by_model.items():
        scored = [(score_candidate(c).composite, c) for c in cands]
        result[model] = max(scored, key=lambda x: x[0])[1]
    return result
