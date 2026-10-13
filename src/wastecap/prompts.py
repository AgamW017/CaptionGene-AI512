"""Prompt template loader and structured-output JSON parser — stub (M2)."""

from __future__ import annotations


def load_prompt(prompt_id: str, model: str | None = None) -> str:
    """Load and render a prompt template. Not implemented until M2."""
    raise NotImplementedError("Prompt loader is implemented in M2.")


def parse_structured_output(text: str) -> dict:
    """Three-stage JSON parser for model outputs. Not implemented until M2."""
    raise NotImplementedError("Structured output parser is implemented in M2.")
