"""M7 metric and error-analysis tests using tiny deterministic examples."""

from __future__ import annotations

import pytest

from wastecap.eval.caption_metrics import compute_test_bleu
from wastecap.eval.error_analysis import failure_categories, group_failures, markdown_summary
from wastecap.eval.grounding_eval import compute_grounding_accuracy


def test_known_exact_caption_has_perfect_test_bleu() -> None:
    scores = compute_test_bleu(["a bright red plastic bottle"], [["a bright red plastic bottle"]])
    assert scores == {
        "bleu_1": pytest.approx(1.0),
        "bleu_2": pytest.approx(1.0),
        "bleu_3": pytest.approx(1.0),
        "bleu_4": pytest.approx(1.0),
    }


def test_grounding_requires_iou_threshold() -> None:
    assert compute_grounding_accuracy([[0, 0, 10, 10]], [[0, 0, 10, 10]]) == 1.0
    assert compute_grounding_accuracy([[20, 20, 5, 5]], [[0, 0, 10, 10]]) == 0.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_grounding_accuracy([], [], 1.1)


def test_error_analysis_groups_only_failures() -> None:
    result = {
        "per_record": [
            {
                "hallucination": True,
                "parse_failed": False,
                "grounding": {"accuracy": 1.0},
                "classes": ["bottle"],
                "conditions": ["crushed"],
                "occluded": True,
            },
            {
                "hallucination": False,
                "parse_failed": False,
                "grounding": {"accuracy": 1.0},
                "classes": ["can"],
                "conditions": ["intact"],
                "occluded": False,
            },
        ]
    }
    groups = group_failures(result)
    assert groups["class"] == {"bottle": 1}
    assert groups["condition"] == {"crushed": 1}
    assert "partially_occluded" in markdown_summary(groups)


def test_failure_categories_are_explicit() -> None:
    result = {
        "per_record": [
            {
                "parse_failed": True,
                "hallucination": True,
                "clipscore": 0.1,
                "grounding": {"accuracy": 0.0},
            }
        ]
    }
    assert failure_categories(result) == {
        "grounding_below_iou_threshold": 1,
        "hallucination": 1,
        "low_clipscore": 1,
        "parse_failed": 1,
    }
