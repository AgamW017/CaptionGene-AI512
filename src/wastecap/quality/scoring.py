"""
ReviewScore: composite quality signal for a caption Candidate.

The score is NOT a ground-truth metric. It is a hint to the human reviewer
pointing out potential issues. The components are:

clip_score      float | None  Raw CLIPScore (0–0.5, higher = better).
                              None if CLIP not available.
completeness    float [0,1]   Fraction of structured fields populated.
                              Fields: category, material, condition, count,
                              caption, location. From raw_parsed objects.
parse_ok        bool          True if parse_failed=False.
length_ok       bool          True if caption word count is in [3, 60].
composite       float [0,1]   Weighted combination of the above.
flags           list[str]     Human-readable flag strings (e.g. "parse_failed").

Composite formula
-----------------
composite = 0.50 * clip_component
          + 0.25 * completeness
          + 0.15 * parse_component
          + 0.10 * length_component

where:
  clip_component = min(clip_score / 0.30, 1.0) if clip_score else 0.15
                   (0.15 = neutral prior when CLIP unavailable)
  parse_component = 1.0 if parse_ok else 0.0
  length_component = 1.0 if length_ok else 0.0

Weights and thresholds are deliberately conservative and can be tuned
without changing the schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wastecap.schema import Candidate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLIP_GOOD_THRESHOLD = 0.30  # normalise clip_score to [0,1]
_CLIP_NEUTRAL_PRIOR = 0.15  # used when clip_score is None
_MIN_WORDS = 3
_MAX_WORDS = 60

_WEIGHTS = {
    "clip": 0.50,
    "completeness": 0.25,
    "parse": 0.15,
    "length": 0.10,
}

# Structured fields we check for completeness
_STRUCTURED_FIELDS = ("category", "material", "condition", "count", "caption", "location")


# ---------------------------------------------------------------------------
# ReviewScore
# ---------------------------------------------------------------------------


@dataclass
class ReviewScore:
    """Quality signals for one Candidate."""

    clip_score: float | None
    completeness: float  # [0, 1]
    parse_ok: bool
    length_ok: bool
    composite: float  # [0, 1]
    flags: list[str] = field(default_factory=list)

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def clip_available(self) -> bool:
        return self.clip_score is not None

    @property
    def grade(self) -> str:
        """Human-readable grade: good / marginal / poor."""
        if self.composite >= 0.65:
            return "good"
        if self.composite >= 0.40:
            return "marginal"
        return "poor"


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _completeness_from_raw_parsed(raw_parsed: dict | None) -> float:
    """Return fraction of structured fields that are non-None/non-empty."""
    if not raw_parsed:
        return 0.0

    # Try objects list first (scene-level response)
    objects = raw_parsed.get("objects")
    obj = (
        objects[0]
        if objects and isinstance(objects, list) and len(objects) > 0
        else raw_parsed  # object-level flat dict
    )

    present = sum(1 for f in _STRUCTURED_FIELDS if obj.get(f) not in (None, "", "unknown"))
    return present / len(_STRUCTURED_FIELDS)


def score_candidate(candidate: Candidate) -> ReviewScore:
    """Compute a ReviewScore for a single Candidate.

    This is a pure function — it reads from the candidate and returns a score.
    It does NOT write to the DB.
    """
    flags: list[str] = []

    # ── Parse ──────────────────────────────────────────────────────────────
    parse_ok = not candidate.parse_failed
    if not parse_ok:
        flags.append("parse_failed")

    # ── Length ─────────────────────────────────────────────────────────────
    word_count = _count_words(candidate.text)
    length_ok = _MIN_WORDS <= word_count <= _MAX_WORDS
    if word_count < _MIN_WORDS:
        flags.append(f"too_short ({word_count} words)")
    elif word_count > _MAX_WORDS:
        flags.append(f"too_long ({word_count} words)")

    # ── Completeness ───────────────────────────────────────────────────────
    completeness = _completeness_from_raw_parsed(candidate.raw_parsed)
    if completeness < 0.5:
        flags.append(f"incomplete_fields ({completeness:.0%})")

    # ── CLIPScore component ────────────────────────────────────────────────
    clip = candidate.clip_score
    if clip is not None:
        clip_component = min(clip / _CLIP_GOOD_THRESHOLD, 1.0)
        if clip < 0.20:
            flags.append(f"low_clip ({clip:.3f})")
    else:
        clip_component = _CLIP_NEUTRAL_PRIOR
        flags.append("clip_not_computed")

    # ── Composite ──────────────────────────────────────────────────────────
    composite = (
        _WEIGHTS["clip"] * clip_component
        + _WEIGHTS["completeness"] * completeness
        + _WEIGHTS["parse"] * (1.0 if parse_ok else 0.0)
        + _WEIGHTS["length"] * (1.0 if length_ok else 0.0)
    )

    return ReviewScore(
        clip_score=clip,
        completeness=completeness,
        parse_ok=parse_ok,
        length_ok=length_ok,
        composite=round(composite, 4),
        flags=flags,
    )


def best_score(candidates: list[Candidate]) -> ReviewScore | None:
    """Return the ReviewScore of the best candidate (highest composite).

    Returns None if the list is empty.
    """
    if not candidates:
        return None
    scores = [score_candidate(c) for c in candidates]
    return max(scores, key=lambda s: s.composite)
