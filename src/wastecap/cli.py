"""
WasteCap CLI entry point.

Subcommands
-----------
generate    Generate captions for images in the DB and write candidates back.
export      Export approved annotations (M6).
stats       Print dataset statistics (M6).
eval        Run baseline evaluation (M7).

Usage
-----
    python -m wastecap.cli generate --model mock --split train --n 3 --limit 20
    python -m wastecap.cli generate --model blip2 --split val --n 1
    python -m wastecap.cli generate --model qwen_vl --split train --n 3 --limit 50
    python -m wastecap.cli generate --model mock --level object --split train --n 1 --limit 5
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from wastecap.models.base import CaptionModel
    from wastecap.schema import AnnotationRecord, Candidate

logging.basicConfig(
    level=os.environ.get("WASTECAP_LOG_LEVEL", "INFO"),
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# generate subcommand implementation
# ---------------------------------------------------------------------------


def _generate_scene(
    model: CaptionModel,
    image: PILImage,
    model_name: str,
    n: int,
    prompt_override: str | None = None,
) -> list[Candidate]:
    """Run full-image captioning; parse and annotate each candidate."""
    from wastecap.prompts import parse_structured_output, render_prompt

    rendered = render_prompt(model_name, "scene_level")
    gen_kwargs: dict = {}
    if rendered.system:
        gen_kwargs["system"] = rendered.system

    raw: list[Candidate] = model.generate(
        image,
        prompt_override or rendered.text,
        n=n,
        prompt_id=rendered.prompt_id,
        **gen_kwargs,
    )
    result: list[Candidate] = []
    for cand in raw:
        pr = parse_structured_output(cand.text)
        result.append(
            cand.model_copy(
                update={
                    "parse_failed": pr.parse_failed,
                    "raw_parsed": pr.raw_parsed,
                    "grounding_boxes": pr.grounding_boxes,
                }
            )
        )
    return result


def _generate_objects(
    model: CaptionModel,
    image: PILImage,
    rec: AnnotationRecord,
    model_name: str,
    n: int,
) -> list[Candidate]:
    """Run per-object captioning on bbox crops; tag each with object_id."""
    from wastecap.prompts import crop_bbox, parse_structured_output, render_prompt

    results: list[Candidate] = []
    for obj in rec.objects:
        if obj.bbox is None:
            continue
        crop = crop_bbox(image, obj.bbox)
        rendered = render_prompt(model_name, "object_level", object_id=obj.object_id)
        gen_kwargs: dict = {}
        if rendered.system:
            gen_kwargs["system"] = rendered.system

        raw: list[Candidate] = model.generate(
            crop, rendered.text, n=n, prompt_id=rendered.prompt_id, **gen_kwargs
        )
        for cand in raw:
            pr = parse_structured_output(cand.text)
            data = dict(pr.raw_parsed) if pr.raw_parsed else dict(pr.data)
            data["_object_id"] = obj.object_id
            results.append(
                cand.model_copy(
                    update={
                        "parse_failed": pr.parse_failed,
                        "raw_parsed": data,
                        "grounding_boxes": pr.grounding_boxes,
                    }
                )
            )
    return results


def _cmd_generate(args: argparse.Namespace) -> int:
    """Generate captions and write Candidates to the DB. Returns exit code."""
    from PIL import Image

    from wastecap.models.registry import get_model
    from wastecap.store import init_db, list_records, update_record

    model = get_model(args.model)
    db_path = init_db()

    records = list_records(
        split=args.split if args.split != "all" else None,
        db_path=db_path,
    )

    if not records:
        logger.error("No records found for split=%r. Run scripts/ingest.py first.", args.split)
        return 1

    if args.limit:
        records = records[: args.limit]

    logger.info(
        "Generating %s-level captions: model=%s split=%s n=%d records=%d",
        args.level,
        args.model,
        args.split,
        args.n,
        len(records),
    )

    ok = skipped = failed = 0

    for rec in records:
        img_path = Path(rec.file)
        if not img_path.exists():
            logger.debug("Image not found on disk: %s — skipping.", img_path)
            skipped += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Cannot open %s: %s — skipping.", img_path, exc)
            skipped += 1
            continue

        try:
            if args.level == "scene":
                new_candidates = _generate_scene(model, image, args.model, args.n)
            else:
                new_candidates = _generate_objects(model, image, rec, args.model, args.n)
        except Exception as exc:
            logger.error("Model error on %s: %s", rec.image_id, exc)
            failed += 1
            continue

        if not new_candidates:
            skipped += 1
            continue

        rec.candidates.extend(new_candidates)
        update_record(rec, db_path)
        ok += 1

        if args.verbose:
            for c in new_candidates:
                flag = " [PARSE_FAIL]" if c.parse_failed else ""
                logger.info("  %s  %s%s: %s", rec.image_id, c.prompt_id, flag, c.text[:80])

    logger.info("Done. ok=%d skipped=%d failed=%d", ok, skipped, failed)
    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _add_generate_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    p = subparsers.add_parser(
        "generate",
        help="Generate captions and write candidates to the DB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("WASTECAP_BACKEND", "mock"),
        choices=["mock", "blip2", "qwen_vl", "llava"],
        help="Caption model backend.",
    )
    p.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "all"],
        help="DB split to process.",
    )
    p.add_argument(
        "--level",
        default="scene",
        choices=["scene", "object"],
        help=(
            "'scene' captions the whole image; 'object' crops each bbox and captions individually."
        ),
    )
    p.add_argument("--n", type=int, default=3, help="Candidates per image.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process (None = all).",
    )
    p.add_argument("--verbose", action="store_true", help="Print each caption.")
    p.set_defaults(func=_cmd_generate)


def _cmd_export(args: argparse.Namespace) -> int:
    from wastecap.export import ExportValidationError, export_splits

    try:
        counts = export_splits(Path(args.output_dir), include_all=args.include_all)
    except ExportValidationError as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Exported annotations: %s", counts)
    return 0


def _add_export_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    p = subparsers.add_parser(
        "export",
        help="Export approved annotations as JSON, JSONL, and CSV per split.",
    )
    p.add_argument("--output-dir", default="results/exports")
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Export all records after validation instead of approved records only.",
    )
    p.set_defaults(func=_cmd_export)


def _cmd_stats(args: argparse.Namespace) -> int:
    import json

    from wastecap.stats import compute_statistics

    print(json.dumps(compute_statistics(), indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wastecap",
        description="WasteCap dataset curation pipeline",
    )
    parser.add_argument("--version", action="version", version="wastecap 0.1.0")
    subparsers = parser.add_subparsers(dest="command")

    _add_generate_parser(subparsers)
    _add_export_parser(subparsers)
    stats_parser = subparsers.add_parser("stats", help="Print dataset statistics.")
    stats_parser.set_defaults(func=_cmd_stats)
    subparsers.add_parser("eval", help="Run baseline evaluation (M7)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if not hasattr(args, "func"):
        logger.error("Subcommand '%s' is not yet implemented.", args.command)
        sys.exit(1)

    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
