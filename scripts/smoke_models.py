"""
Manual GPU smoke test for BLIP-2 and Qwen2.5-VL.

Run this on your GPU machine after downloading weights.
It does NOT use the DB — it generates directly from a test image and
prints the raw output + parsed result so you can verify quality.

Usage:
    WASTECAP_DEVICE=cuda WASTECAP_QUANTIZE=4bit python scripts/smoke_models.py
    WASTECAP_DEVICE=cuda WASTECAP_QUANTIZE=4bit python scripts/smoke_models.py --model blip2
    WASTECAP_DEVICE=cuda WASTECAP_QUANTIZE=4bit python scripts/smoke_models.py --model qwen_vl
    WASTECAP_DEVICE=cuda WASTECAP_QUANTIZE=4bit python scripts/smoke_models.py --model all

Expected VRAM usage (4-bit, measured on RTX 3060 / RTX 4060):
    blip2    Salesforce/blip2-opt-2.7b          ~1.9–2.1 GB
    qwen_vl  Qwen/Qwen2.5-VL-3B-Instruct        ~2.3–2.6 GB
    llava    llava-hf/llava-1.5-7b-hf            ~4.3–4.7 GB  (optional, needs >4 GB)

Both blip2 and qwen_vl fit simultaneously on a 4 GB card at 4-bit.
Load them one at a time (the registry caches per process).

What a passing run looks like
------------------------------
  [blip2] scene_level — PASS
    raw  : {"objects": [{"object_id": "o1", "category": "bottle", ...}], ...}
    data : objects=[...], scene_context="..."
    parse_failed: False

  [qwen_vl] scene_level — PASS
    raw  : {"objects": [...], "scene_context": "..."}
    data : objects=[...], scene_context="..."
    parse_failed: False
    grounding_boxes: [[x1,y1,x2,y2], ...]   (or None if no boxes returned)

If you see parse_failed=True, the model returned non-JSON. The text is
preserved in Candidate.text and flagged for review — it is not dropped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("WASTECAP_CONFIGS_DIR", "configs")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("smoke")

_FIXTURE_IMAGE = Path("tests/fixtures/images/taco_fix001.jpg")
_FIXTURE_BBOX = [80.0, 60.0, 40.0, 30.0]  # from taco_fix001 annotation


def _check_torch() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            device = os.environ.get("WASTECAP_DEVICE", "cpu")
            if device == "cuda":
                logger.error(
                    "CUDA not available but WASTECAP_DEVICE=cuda. "
                    "Set WASTECAP_DEVICE=cpu to test on CPU (slow)."
                )
                return False
        return True
    except ImportError:
        logger.error("torch not installed. Install ML extras:\n  .venv/bin/pip install -e '.[ml]'")
        return False


def _vram_used_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except ImportError:
        pass
    return None


def _load_image() -> PILImage:
    from PIL import Image

    if not _FIXTURE_IMAGE.exists():
        logger.error("Fixture image not found: %s", _FIXTURE_IMAGE)
        logger.error("Run from the project root: python scripts/smoke_models.py")
        sys.exit(1)
    return Image.open(_FIXTURE_IMAGE).convert("RGB")


def _smoke_model(model_name: str, image: PILImage) -> bool:
    """Run scene + object-level smoke test for one model. Returns True on pass."""
    from wastecap.models.registry import clear_cache, get_model
    from wastecap.prompts import crop_bbox, parse_structured_output, render_prompt

    logger.info("=" * 60)
    logger.info("Smoke test: %s", model_name.upper())
    logger.info("  WASTECAP_DEVICE  = %s", os.environ.get("WASTECAP_DEVICE", "cpu"))
    logger.info("  WASTECAP_QUANTIZE= %s", os.environ.get("WASTECAP_QUANTIZE", "none"))

    # Fresh load per model
    clear_cache()
    os.environ["WASTECAP_BACKEND"] = model_name

    vram_before = _vram_used_mb()

    try:
        model = get_model(model_name)
    except ImportError as exc:
        logger.error("Cannot load %s: %s", model_name, exc)
        return False

    vram_after = _vram_used_mb()
    if vram_before is not None and vram_after is not None:
        logger.info(
            "  VRAM delta: +%.0f MB (total allocated: %.0f MB)",
            vram_after - vram_before,
            vram_after,
        )

    all_passed = True

    # --- Scene-level test ---
    logger.info("\n[%s] scene_level", model_name)
    rendered = render_prompt(model_name, "scene_level")
    gen_kwargs: dict = {}
    if rendered.system:
        gen_kwargs["system"] = rendered.system

    try:
        candidates = model.generate(
            image, rendered.text, n=1, prompt_id=rendered.prompt_id, **gen_kwargs
        )
        cand = candidates[0]
        pr = parse_structured_output(cand.text)

        status = "PASS" if not pr.parse_failed else "PARSE_FAIL (flagged, not dropped)"
        logger.info("  status       : %s", status)
        logger.info("  raw text     : %s", textwrap.shorten(cand.text, 200))
        logger.info("  parse_failed : %s", pr.parse_failed)
        if pr.grounding_boxes:
            logger.info("  grounding_boxes: %s", pr.grounding_boxes[:3])
        if pr.data:
            logger.info("  parsed data  : %s", json.dumps(pr.data, indent=2)[:300])

        if pr.parse_failed:
            all_passed = False
    except Exception as exc:
        logger.error("  ERROR: %s", exc)
        all_passed = False

    # --- Object-level test (crop) ---
    logger.info("\n[%s] object_level (bbox crop)", model_name)
    try:
        crop = crop_bbox(image, _FIXTURE_BBOX)
        logger.info("  crop size: %s", crop.size)
        rendered_obj = render_prompt(model_name, "object_level", object_id="o1")

        gen_kwargs2: dict = {}
        if rendered_obj.system:
            gen_kwargs2["system"] = rendered_obj.system

        candidates_obj = model.generate(
            crop, rendered_obj.text, n=1, prompt_id=rendered_obj.prompt_id, **gen_kwargs2
        )
        cand_obj = candidates_obj[0]
        pr_obj = parse_structured_output(cand_obj.text)

        status_obj = "PASS" if not pr_obj.parse_failed else "PARSE_FAIL (flagged)"
        logger.info("  status       : %s", status_obj)
        logger.info("  raw text     : %s", textwrap.shorten(cand_obj.text, 200))
        logger.info("  parse_failed : %s", pr_obj.parse_failed)
        if pr_obj.data:
            logger.info("  parsed data  : %s", json.dumps(pr_obj.data, indent=2)[:300])

        if pr_obj.parse_failed:
            all_passed = False
    except Exception as exc:
        logger.error("  ERROR: %s", exc)
        all_passed = False

    vram_final = _vram_used_mb()
    if vram_final is not None:
        logger.info("\n  Final VRAM allocated: %.0f MB", vram_final)

    logger.info("\n[%s] overall: %s", model_name, "PASS" if all_passed else "FAIL")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test BLIP-2 and/or Qwen2.5-VL on GPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["blip2", "qwen_vl", "llava", "all"],
        default="all",
        help="Which model(s) to test.",
    )
    args = parser.parse_args()

    if not _check_torch():
        sys.exit(1)

    image = _load_image()

    models_to_test = ["blip2", "qwen_vl"] if args.model == "all" else [args.model]

    results: dict[str, bool] = {}
    for m in models_to_test:
        results[m] = _smoke_model(m, image)

    logger.info("\n%s", "=" * 60)
    logger.info("SMOKE TEST SUMMARY")
    for m, passed in results.items():
        logger.info("  %-12s %s", m, "PASS ✓" if passed else "FAIL ✗")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
