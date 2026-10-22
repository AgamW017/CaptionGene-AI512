"""
Hallucination detection heuristics for WasteCap captions.

These are HINTS for the human reviewer — never automatic verdicts.
A flag means "this candidate may contain hallucinated content; please check."
The reviewer sets review.hallucination=True (or False) after inspection.

Heuristics implemented
----------------------
1. category_mismatch:
   The caption (or raw_parsed category field) names a WasteCategory that
   is absent from the record's reference object list.
   e.g. caption says "bottle" but all reference objects are "can".

2. material_mismatch:
   Same logic for material (e.g. caption says "glass" but ref says "plastic").

3. invented_count:
   The structured count field is > 1 but the record has no reference for it
   (there is only 1 annotated object of that category in the reference).

4. empty_caption_hallucination:
   The parse succeeded but the caption string is empty or trivially short
   (< 3 words), suggesting the model produced a non-description.

API
---
flag_candidate(candidate, reference_objects) -> Candidate
    Returns a copy of the candidate with hallucination flags stored in
    raw_parsed["_hallucination_flags"]. Does NOT write to DB.

flag_all_candidates(rec) -> AnnotationRecord
    Applies flag_candidate to every candidate in a record. Returns a copy.
    Does NOT write to DB — caller decides whether to persist.
"""

from __future__ import annotations

import logging
import re

from wastecap.schema import Candidate, Material, ObjectAnnotation, WasteCategory

logger = logging.getLogger(__name__)

# Minimum words for a non-trivial caption
_MIN_CAPTION_WORDS = 3

# Waste category and material tokens extracted from caption text
# We match lowercase category/material names as whole words in the caption.
_ALL_CATEGORIES = frozenset(c.value for c in WasteCategory if c != WasteCategory.unknown)
_ALL_MATERIALS = frozenset(m.value for m in Material if m != Material.unknown)


# ---------------------------------------------------------------------------
# Token extraction from free text
# ---------------------------------------------------------------------------


def _tokens_in_text(text: str, vocab: frozenset[str]) -> set[str]:
    """Return the subset of *vocab* words that appear as whole words in *text*."""
    lower = text.lower()
    found: set[str] = set()
    for word in vocab:
        # Match whole-word (handles multi-word vocab like "food_waste")
        pattern = r"\b" + re.escape(word.replace("_", " ")) + r"\b"
        if re.search(pattern, lower.replace("_", " ")):
            found.add(word)
    return found


def _caption_text(candidate: Candidate) -> str:
    """Extract the best caption string from a Candidate."""
    if candidate.raw_parsed:
        objects = candidate.raw_parsed.get("objects") or []
        if objects and isinstance(objects, list) and objects[0].get("caption"):
            return str(objects[0]["caption"])
        if candidate.raw_parsed.get("caption"):
            return str(candidate.raw_parsed["caption"])
    return candidate.text


def _parsed_category(candidate: Candidate) -> str | None:
    """Get the structured category from raw_parsed (if available)."""
    if not candidate.raw_parsed:
        return None
    objects = candidate.raw_parsed.get("objects") or []
    if objects and isinstance(objects, list):
        return objects[0].get("category")
    return candidate.raw_parsed.get("category")


def _parsed_material(candidate: Candidate) -> str | None:
    if not candidate.raw_parsed:
        return None
    objects = candidate.raw_parsed.get("objects") or []
    if objects and isinstance(objects, list):
        return objects[0].get("material")
    return candidate.raw_parsed.get("material")


def _parsed_count(candidate: Candidate) -> int:
    if not candidate.raw_parsed:
        return 1
    objects = candidate.raw_parsed.get("objects") or []
    if objects and isinstance(objects, list):
        return int(objects[0].get("count", 1))
    return int(candidate.raw_parsed.get("count", 1))


# ---------------------------------------------------------------------------
# Individual heuristics
# ---------------------------------------------------------------------------


def _check_category_mismatch(
    candidate: Candidate,
    ref_objects: list[ObjectAnnotation],
) -> list[str]:
    """Flag if the candidate's category is absent from reference object categories."""
    if candidate.parse_failed:
        return []
    if not ref_objects:
        return []

    ref_categories = {o.category.value for o in ref_objects if o.category != WasteCategory.unknown}
    if not ref_categories:
        return []

    # Check structured field first (most reliable)
    parsed_cat = _parsed_category(candidate)
    if parsed_cat and parsed_cat not in ("unknown",):
        if parsed_cat not in ref_categories:
            return [
                f"category_mismatch: caption says '{parsed_cat}' "
                f"but reference has {sorted(ref_categories)}"
            ]
        return []

    # Fall back to text token matching
    caption = _caption_text(candidate)
    mentioned = _tokens_in_text(caption, _ALL_CATEGORIES)
    hallucinated = mentioned - ref_categories
    if hallucinated:
        return [
            f"category_mismatch: caption mentions {sorted(hallucinated)} "
            f"absent from reference {sorted(ref_categories)}"
        ]
    return []


def _check_material_mismatch(
    candidate: Candidate,
    ref_objects: list[ObjectAnnotation],
) -> list[str]:
    """Flag if the candidate's material is absent from reference object materials."""
    if candidate.parse_failed:
        return []
    if not ref_objects:
        return []

    ref_materials = {o.material.value for o in ref_objects if o.material != Material.unknown}
    if not ref_materials:
        return []  # all reference materials unknown — can't check

    parsed_mat = _parsed_material(candidate)
    if parsed_mat and parsed_mat not in ("unknown",):
        if parsed_mat not in ref_materials:
            return [
                f"material_mismatch: caption says '{parsed_mat}' "
                f"but reference has {sorted(ref_materials)}"
            ]
        return []

    # Text token check
    caption = _caption_text(candidate)
    mentioned = _tokens_in_text(caption, _ALL_MATERIALS)
    hallucinated = mentioned - ref_materials
    if hallucinated:
        return [
            f"material_mismatch: caption mentions {sorted(hallucinated)} "
            f"absent from reference {sorted(ref_materials)}"
        ]
    return []


def _check_invented_count(
    candidate: Candidate,
    ref_objects: list[ObjectAnnotation],
) -> list[str]:
    """Flag if structured count > number of reference objects of that category."""
    if candidate.parse_failed or not ref_objects:
        return []

    parsed_cat = _parsed_category(candidate)
    parsed_cnt = _parsed_count(candidate)

    if parsed_cnt <= 1:
        return []

    if parsed_cat and parsed_cat not in ("unknown",):
        ref_count = sum(1 for o in ref_objects if o.category.value == parsed_cat)
        if ref_count > 0 and parsed_cnt > ref_count:
            return [
                f"invented_count: caption says {parsed_cnt}× '{parsed_cat}' "
                f"but reference has {ref_count}"
            ]
    return []


def _check_empty_caption(candidate: Candidate) -> list[str]:
    """Flag parse-succeeded candidates with trivially short captions."""
    if candidate.parse_failed:
        return []
    caption = _caption_text(candidate)
    words = len(re.findall(r"\S+", caption))
    if words < _MIN_CAPTION_WORDS:
        return [f"empty_or_trivial_caption ({words} words after parse)"]
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flag_candidate(
    candidate: Candidate,
    reference_objects: list[ObjectAnnotation],
) -> Candidate:
    """Apply all hallucination heuristics to one candidate.

    Returns a new Candidate with flags stored in
    raw_parsed["_hallucination_flags"]. If raw_parsed is None, a minimal
    dict is created to hold the flags.

    This function does NOT write to the DB.
    """
    flags: list[str] = []
    flags.extend(_check_category_mismatch(candidate, reference_objects))
    flags.extend(_check_material_mismatch(candidate, reference_objects))
    flags.extend(_check_invented_count(candidate, reference_objects))
    flags.extend(_check_empty_caption(candidate))

    if not flags:
        return candidate  # return original unchanged (common path)

    new_raw = dict(candidate.raw_parsed) if candidate.raw_parsed else {}
    new_raw["_hallucination_flags"] = flags

    return candidate.model_copy(update={"raw_parsed": new_raw})


def flag_all_candidates(rec: object) -> object:
    """Apply hallucination flags to every candidate in an AnnotationRecord.

    Returns a copy of the record with updated candidates.
    Does NOT write to the DB.
    """
    updated_candidates = [flag_candidate(c, rec.objects) for c in rec.candidates]
    return rec.model_copy(update={"candidates": updated_candidates})


def get_hallucination_flags(candidate: Candidate) -> list[str]:
    """Return the stored hallucination flags from a candidate's raw_parsed."""
    if candidate.raw_parsed and isinstance(candidate.raw_parsed, dict):
        return candidate.raw_parsed.get("_hallucination_flags", [])
    return []


def is_hallucination_flagged(candidate: Candidate) -> bool:
    """Return True if the candidate has any hallucination flags."""
    return bool(get_hallucination_flags(candidate))
