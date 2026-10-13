"""CIDEr and BLEU caption metrics — stub (M7)."""

from __future__ import annotations


def compute_cider(hypotheses: list[str], references: list[list[str]]) -> float:
    """Compute CIDEr score. Not implemented until M7."""
    raise NotImplementedError("Caption metrics are implemented in M7.")


def compute_bleu(hypotheses: list[str], references: list[list[str]]) -> dict:
    """Compute BLEU scores (1–4). Not implemented until M7."""
    raise NotImplementedError("Caption metrics are implemented in M7.")
