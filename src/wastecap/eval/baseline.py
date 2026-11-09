"""Real baseline evaluation for approved records."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from wastecap.eval.caption_metrics import compute_bleu, compute_cider
from wastecap.eval.grounding_eval import grounding_summary
from wastecap.grounding.boxes import qwen_to_pixel, xyxy_to_xywh
from wastecap.models.registry import get_model
from wastecap.prompts import render_prompt
from wastecap.quality.clipscore import compute_clipscore
from wastecap.store import list_records

logger = logging.getLogger(__name__)


def _caption(candidate: Any) -> str:
    parsed = candidate.raw_parsed or {}
    objects = parsed.get("objects") or []
    captions = [str(obj["caption"]) for obj in objects if obj.get("caption")]
    return " ".join(captions) if captions else candidate.text


def _reference_boxes(record: Any) -> list[list[float]]:
    return [obj.bbox for obj in record.objects if obj.bbox is not None]


def _predicted_boxes(candidate: Any, record: Any) -> list[list[float]]:
    boxes = candidate.grounding_boxes or []
    return [xyxy_to_xywh(qwen_to_pixel(box, record.width, record.height)) for box in boxes]


def run_baseline(
    model: str,
    split: str,
    *,
    limit: int | None = None,
    db_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate and evaluate one real model against approved records."""
    records = list_records(split=split, status="approved", limit=limit, db_path=db_path)
    if not records:
        raise ValueError(f"No approved records found for split {split!r}")

    backend = get_model(model)
    prompt = render_prompt(model, "scene_level")
    hypotheses: list[str] = []
    references: list[list[str]] = []
    per_record: list[dict[str, Any]] = []

    for record in records:
        image_path = Path(record.file)
        if not image_path.exists():
            raise FileNotFoundError(f"Image for {record.image_id} does not exist: {image_path}")
        from PIL import Image

        with Image.open(image_path) as image:
            candidate = backend.generate(
                image.convert("RGB"),
                prompt.text + "\nFor every visible object, also emit <ref>name</ref>"
                "<box>[[x1,y1,x2,y2]]</box> tags using coordinates 0-999.",
                n=1,
                prompt_id=prompt.prompt_id,
                system=prompt.system,
            )[0]
            hypothesis = _caption(candidate).strip()
            clip_score = compute_clipscore(image.convert("RGB"), hypothesis)
        if clip_score is None:
            raise RuntimeError(
                "CLIPScore could not be computed. Install the ML extras and model weights "
                "before running the real baseline: pip install -e '.[ml]'."
            )

        refs = [record.caption] if record.caption else []
        if not refs:
            raise ValueError(f"Approved record {record.image_id} has no reference caption")
        predicted = _predicted_boxes(candidate, record)
        grounding = grounding_summary(predicted, _reference_boxes(record))
        hypotheses.append(hypothesis)
        references.append(refs)
        per_record.append(
            {
                "image_id": record.image_id,
                "hypothesis": hypothesis,
                "reference": refs[0],
                "clipscore": clip_score,
                "parse_failed": candidate.parse_failed,
                "hallucination": bool(
                    candidate.parse_failed or (record.review and record.review.hallucination)
                ),
                "grounding": grounding,
                "classes": sorted({obj.category.value for obj in record.objects}),
                "conditions": sorted(
                    {obj.condition.value for obj in record.objects if obj.condition}
                ),
                "occluded": any(
                    obj.condition and obj.condition.value == "partially_occluded"
                    for obj in record.objects
                )
                or "occlud" in (record.scene_context or "").lower(),
            }
        )

    bleu = compute_bleu(hypotheses, references)
    cider = compute_cider(hypotheses, references)
    summary = {
        "model": model,
        "split": split,
        "records": len(records),
        "bleu": bleu,
        "cider": cider,
        "caption_metric_tokenizer": "ptb" if shutil.which("java") else "whitespace",
        "clipscore": sum(r["clipscore"] for r in per_record) / len(per_record),
        "hallucination_rate": sum(r["hallucination"] for r in per_record) / len(per_record),
        "grounding_accuracy": sum(r["grounding"]["matched_boxes"] for r in per_record)
        / max(sum(r["grounding"]["reference_boxes"] for r in per_record), 1),
        "grounding_accuracy_by_category": _grounding_by_category(records),
        "hallucination_denominator": len(per_record),
        "per_record": per_record,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _grounding_by_category(records: list[Any]) -> dict[str, dict[str, Any]]:
    """Report category denominators without inventing matches.

    Qwen's parsed boxes currently have no object labels, so a global match
    cannot be attributed to a category. The explicit ``accuracy: None`` value
    keeps that limitation visible in machine-readable output.
    """
    references: dict[str, int] = {}
    for record in records:
        for obj in record.objects:
            if obj.bbox is not None:
                references[obj.category.value] = references.get(obj.category.value, 0) + 1
    return {
        category: {
            "reference_boxes": count,
            "matched_boxes": None,
            "accuracy": None,
            "reason": "predicted boxes have no object labels",
        }
        for category, count in sorted(references.items())
    }


def main() -> int:
    """Run the baseline evaluator as ``python -m wastecap.eval.baseline``."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen_vl")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()
    result = run_baseline(
        args.model,
        args.split,
        limit=args.limit,
        db_path=args.db_path,
        output_path=args.output,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_record"},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
