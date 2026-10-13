"""Hallucination detection heuristic — stub (M4)."""

from __future__ import annotations

from wastecap.schema import Candidate


def flag_hallucinations(candidate: Candidate) -> Candidate:
    """Flag a candidate if its named objects/materials are not in the structured parse.

    Not implemented until M4.
    """
    raise NotImplementedError("Hallucination detection is implemented in M4.")
