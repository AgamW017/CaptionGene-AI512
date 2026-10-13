"""Sorting instruction generator — stub (M5)."""

from __future__ import annotations

from wastecap.schema import ObjectAnnotation


def generate_instruction(obj: ObjectAnnotation) -> str:
    """Generate a natural-language sorting instruction for one object.

    Every instruction is grounded to the object's bbox (object_id + bbox required).
    Not implemented until M5.
    """
    raise NotImplementedError("Instruction generator is implemented in M5.")
