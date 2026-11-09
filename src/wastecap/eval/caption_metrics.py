"""Caption metrics with the official COCO evaluator when available."""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _ngrams(tokens: list[str], order: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + order]) for i in range(len(tokens) - order + 1)]


def compute_test_bleu(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute dependency-free BLEU for metric unit tests and diagnostics."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    precisions: list[float] = []
    for order in range(1, 5):
        matched = total = 0
        for hypothesis, refs in zip(hypotheses, references, strict=True):
            hyp = _tokenize(hypothesis)
            ref_counts = Counter()
            for ref in refs:
                tokens = _tokenize(ref)
                ref_counts.update(_ngrams(tokens, order))
            hyp_counts = Counter(_ngrams(hyp, order))
            total += sum(hyp_counts.values())
            matched += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append(matched / total if total else 0.0)
    return {f"bleu_{i}": value for i, value in enumerate(precisions, 1)}


def _coco_scores(
    hypotheses: list[str], references: list[list[str]]
) -> tuple[dict[str, float], float]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        raise ImportError(
            "pycocoevalcap is required for official BLEU/CIDEr metrics. "
            "Install it with: pip install git+https://github.com/tylin/coco-caption.git"
        ) from exc

    gts = {str(i): [{"caption": ref} for ref in refs] for i, refs in enumerate(references)}
    res = {str(i): [{"caption": hyp}] for i, hyp in enumerate(hypotheses)}
    tokenizer = PTBTokenizer()
    try:
        tokenized_gts = tokenizer.tokenize(gts)
        tokenized_res = tokenizer.tokenize(res)
    except FileNotFoundError:
        # The scorers remain the official pycocoevalcap implementations. This
        # fallback keeps evaluation runnable on machines without Java.
        tokenized_gts = {
            key: [" ".join(_tokenize(item["caption"])) for item in values]
            for key, values in gts.items()
        }
        tokenized_res = {
            key: [" ".join(_tokenize(item["caption"])) for item in values]
            for key, values in res.items()
        }
    bleu_values, _ = Bleu(4).compute_score(tokenized_gts, tokenized_res)
    cider, _ = Cider().compute_score(tokenized_gts, tokenized_res)
    return {f"bleu_{i}": float(value) for i, value in enumerate(bleu_values, 1)}, float(cider)


def compute_cider(hypotheses: list[str], references: list[list[str]]) -> float:
    """Compute official COCO CIDEr, requiring ``pycocoevalcap``."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    return _coco_scores(hypotheses, references)[1]


def compute_bleu(hypotheses: list[str], references: list[list[str]]) -> dict[str, float]:
    """Compute official COCO BLEU-1..4, requiring ``pycocoevalcap``."""
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    return _coco_scores(hypotheses, references)[0]
