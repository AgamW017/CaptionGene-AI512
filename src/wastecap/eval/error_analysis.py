"""Group baseline failures by reference taxonomy and occlusion."""

from __future__ import annotations

from collections import Counter
from typing import Any


def group_failures(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Count failed records by class, condition, and occlusion."""
    groups = {"class": Counter(), "condition": Counter(), "occlusion": Counter()}
    for record in result.get("per_record", []):
        grounding_failed = record["grounding"]["accuracy"] < 1.0
        failed = record["hallucination"] or record["parse_failed"] or grounding_failed
        if not failed:
            continue
        for value in record.get("classes", []):
            groups["class"][value] += 1
        for value in record.get("conditions", []):
            groups["condition"][value] += 1
        groups["occlusion"]["partially_occluded" if record.get("occluded") else "not_occluded"] += 1
    return {key: dict(sorted(value.items())) for key, value in groups.items()}


def markdown_summary(groups: dict[str, dict[str, int]]) -> str:
    lines = ["| Dimension | Value | Failed records |", "|---|---|---:|"]
    for dimension, values in groups.items():
        for value, count in values.items():
            lines.append(f"| {dimension} | {value} | {count} |")
    return "\n".join(lines) + "\n"


def failure_categories(result: dict[str, Any]) -> dict[str, int]:
    """Return counts for the distinct failure mechanisms in a result."""
    counts = Counter()
    for record in result.get("per_record", []):
        if record.get("parse_failed"):
            counts["parse_failed"] += 1
        if record.get("hallucination"):
            counts["hallucination"] += 1
        if record.get("grounding", {}).get("accuracy", 0.0) < 1.0:
            counts["grounding_below_iou_threshold"] += 1
        if record.get("clipscore") is not None and record["clipscore"] < 0.3:
            counts["low_clipscore"] += 1
    return dict(sorted(counts.items()))
