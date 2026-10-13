"""Structured annotation form: object/material/condition/count/context — stub (M3)."""

from __future__ import annotations

from wastecap.schema import ObjectAnnotation


def structured_form(obj: ObjectAnnotation | None = None) -> ObjectAnnotation | None:
    """Render structured annotation form.

    Returns updated ObjectAnnotation or None. Not implemented until M3.
    """
    raise NotImplementedError("structured_form is implemented in M3.")
