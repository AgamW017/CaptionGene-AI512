"""Run Qwen grounding over downloaded source images without changing review status."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from PIL import Image

from wastecap.models.registry import get_model
from wastecap.prompts import parse_structured_output
from wastecap.store import list_records, update_record

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/qwen_taco_grounding.json"))
    parser.add_argument("--db-path", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    records = list_records(source="taco", limit=args.limit, db_path=args.db_path)
    if not records:
        raise SystemExit("No taco records found")
    model = get_model("qwen_vl")
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records, 1):
        image_path = Path(record.file)
        if not image_path.exists():
            logger.warning("Skipping missing image %s", image_path)
            continue
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            object_results: list[dict[str, object]] = []
            predicted_boxes = 0
            parse_failures = 0
            for obj in record.objects:
                description = " ".join(
                    part
                    for part in (
                        obj.condition.value if obj.condition else None,
                        obj.material.value if obj.material.value != "unknown" else None,
                        obj.category.value,
                    )
                    if part
                )
                prompt = (
                    f"Locate exactly one instance of {description} in this image. "
                    "Return only this format, with integer coordinates normalized from 0 to 999: "
                    f"<ref>{description}</ref><box>[[x1,y1,x2,y2]]</box>. "
                    "Do not describe the image, add JSON, markdown, or any other text."
                )
                candidate = model.generate(
                    rgb_image,
                    prompt,
                    n=1,
                    prompt_id="qwen_grounding_object_v2",
                )[0]
                parsed = parse_structured_output(candidate.text)
                candidate = candidate.model_copy(
                    update={
                        "parse_failed": parsed.parse_failed,
                        "raw_parsed": parsed.raw_parsed,
                        "grounding_boxes": parsed.grounding_boxes,
                    }
                )
                record.candidates.append(candidate)
                boxes = candidate.grounding_boxes
                predicted_boxes += len(boxes or [])
                if not boxes:
                    parse_failures += 1
                object_results.append(
                    {
                        "object_id": obj.object_id,
                        "description": description,
                        "predicted_boxes": boxes or [],
                    }
                )
        update_record(record, args.db_path)
        rows.append(
            {
                "image_id": record.image_id,
                "reference_boxes": sum(obj.bbox is not None for obj in record.objects),
                "predicted_boxes": predicted_boxes,
                "parse_failed": parse_failures > 0,
                "objects": object_results,
            }
        )
        logger.info(
            "[%d/%d] %s predicted_boxes=%d parse_failed=%s",
            index,
            len(records),
            record.image_id,
            predicted_boxes,
            parse_failures > 0,
        )

    output = {
        "model": "qwen_vl",
        "source": "taco",
        "requested": args.limit,
        "processed": len(rows),
        "with_predicted_boxes": sum(bool(row["predicted_boxes"]) for row in rows),
        "parse_failures": sum(bool(row["parse_failed"]) for row in rows),
        "elapsed_seconds": time.monotonic() - started,
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
