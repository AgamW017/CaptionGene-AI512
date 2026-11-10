"""
Prompt template loader and structured-output parser for WasteCap.

Prompt loading
--------------
Templates live in configs/prompts/<model>.yaml.
Each file has two top-level keys: ``scene_level`` and ``object_level``.
Each key is a dict with: id, description, template, and optionally system.

The public API is:

    render_prompt(model, level, object_id=None) -> RenderedPrompt
    get_prompt_id(model, level) -> str

Structured-output parsing
--------------------------
Models are asked to produce JSON. In practice they often wrap it in prose,
markdown fences, or produce malformed JSON. We use a three-stage parser:

  Stage 1: json.loads(text) — works if the model returned bare JSON.
  Stage 2: extract the first {...} block with a regex, then json.loads.
  Stage 3: extract ```json ... ``` fence content, then json.loads.
  Fallback: store {\"caption\": text, \"parse_error\": true} and set
            parse_failed=True on the Candidate.

Parsed fields are coerced into our vocabulary (WasteCategory, Material,
Condition). Out-of-vocab values become "unknown"; the original value is
preserved in raw_parsed for inspection.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from wastecap.schema import Condition, Material, WasteCategory

if TYPE_CHECKING:
    from PIL import Image as PILImageModule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_VALID_LEVELS = frozenset(["scene_level", "object_level"])
_VALID_CATEGORIES = frozenset(c.value for c in WasteCategory)
_VALID_MATERIALS = frozenset(m.value for m in Material)
_VALID_CONDITIONS = frozenset(c.value for c in Condition)


def _prompts_dir() -> Path:
    configs_dir = os.environ.get("WASTECAP_CONFIGS_DIR", "configs")
    return Path(configs_dir) / "prompts"


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderedPrompt:
    prompt_id: str
    text: str
    system: str | None  # Only set for models that use a system message (e.g. Qwen)


@lru_cache(maxsize=16)
def _load_template_file(model: str) -> dict:
    """Load and cache <model>.yaml from the prompts directory.

    Falls back to default.yaml if the model-specific file doesn't exist.
    """
    d = _prompts_dir()
    path = d / f"{model}.yaml"
    if not path.exists():
        fallback = d / "default.yaml"
        if not fallback.exists():
            raise FileNotFoundError(
                f"No prompt template file found for model {model!r} "
                f"(tried {path} and {fallback}). "
                "Set WASTECAP_CONFIGS_DIR to the configs/ directory."
            )
        logger.debug("No prompt file for %r — using default.yaml", model)
        path = fallback

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    logger.debug("Loaded prompt templates from %s", path)
    return data


def reload_prompts() -> None:
    """Clear the template cache (useful in tests)."""
    _load_template_file.cache_clear()


def render_prompt(
    model: str,
    level: str,
    *,
    object_id: str | None = None,
) -> RenderedPrompt:
    """Load and render a prompt template.

    Args:
        model:     Backend name, e.g. 'blip2', 'qwen_vl', 'mock'.
        level:     'scene_level' or 'object_level'.
        object_id: Required for object_level prompts; substituted into
                   {object_id} placeholders in the template.

    Returns:
        RenderedPrompt with the filled-in prompt text, its id, and
        an optional system message.
    """
    if level not in _VALID_LEVELS:
        raise ValueError(f"level must be one of {_VALID_LEVELS}, got {level!r}")
    if level == "object_level" and object_id is None:
        raise ValueError("object_id is required for object_level prompts")

    tmpl_data = _load_template_file(model)

    if level not in tmpl_data:
        raise KeyError(
            f"Level {level!r} not found in prompt file for model {model!r}. "
            f"Available levels: {list(tmpl_data.keys())}"
        )

    entry = tmpl_data[level]
    prompt_id: str = entry["id"]
    raw_template: str = entry["template"]
    system: str | None = entry.get("system")

    # Substitute only the known {object_id} placeholder using regex so that
    # literal JSON braces in the template (e.g. {"objects": ...}) are left alone.
    text = raw_template
    if object_id is not None:
        text = re.sub(r"\{object_id\}", object_id, text)

    return RenderedPrompt(
        prompt_id=prompt_id,
        text=text.strip(),
        system=system.strip() if system else None,
    )


def get_prompt_id(model: str, level: str) -> str:
    """Return the prompt_id for a (model, level) pair without rendering."""
    return _load_template_file(model)[level]["id"]


# ---------------------------------------------------------------------------
# Structured-output parsing (3-stage)
# ---------------------------------------------------------------------------

# Regex to find the outermost {...} block
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
# Regex for ```json ... ``` fences
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Regex for Qwen grounding tags: <ref>text</ref><box>[[x1,y1,x2,y2]]</box>
_QWEN_BOX_RE = re.compile(r"<ref>[^<]*</ref><box>(\[\[.*?\]\])</box>", re.DOTALL)


def _try_parse(text: str) -> dict | None:
    """Attempt json.loads; return None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        try:
            result = ast.literal_eval(text)
            if isinstance(result, dict):
                return result
        except (SyntaxError, ValueError, TypeError):
            pass
    return None


def _coerce_field(value: Any, valid_set: frozenset, fallback: str) -> tuple[str, str | None]:
    """Coerce a field value to a valid enum value.

    Returns (coerced_value, original_value_if_changed).
    """
    if isinstance(value, str) and value.lower() in valid_set:
        return value.lower(), None
    original = str(value) if value is not None else None
    return fallback, original


def _normalise_object(obj: dict) -> dict:
    """Coerce category/material/condition to canonical values in place."""
    out = dict(obj)

    cat, orig_cat = _coerce_field(obj.get("category"), _VALID_CATEGORIES, "unknown")
    out["category"] = cat
    if orig_cat:
        out.setdefault("_raw", {})["category"] = orig_cat

    mat, orig_mat = _coerce_field(obj.get("material"), _VALID_MATERIALS, "unknown")
    out["material"] = mat
    if orig_mat:
        out.setdefault("_raw", {})["material"] = orig_mat

    if "condition" in obj and obj["condition"] is not None:
        cond, orig_cond = _coerce_field(obj["condition"], _VALID_CONDITIONS, "intact")
        out["condition"] = cond
        if orig_cond:
            out.setdefault("_raw", {})["condition"] = orig_cond

    # Ensure count is an integer ≥ 1
    try:
        out["count"] = max(1, int(obj.get("count", 1)))
    except (TypeError, ValueError):
        out["count"] = 1

    return out


def _extract_grounding_boxes(text: str) -> list[list[float]] | None:
    """Extract Qwen2.5-VL grounding boxes from raw model output text.

    Qwen outputs: <ref>object</ref><box>[[x1,y1,x2,y2]]</box>
    Coordinates are in Qwen's 0-999 normalised space.
    """
    matches = _QWEN_BOX_RE.findall(text)
    if not matches:
        return None
    boxes: list[list[float]] = []
    for match in matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, list):
                for box in parsed:
                    if isinstance(box, list) and len(box) == 4:
                        boxes.append([float(v) for v in box])
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return boxes if boxes else None


def _grounding_data(text: str, boxes: list[list[float]] | None) -> dict | None:
    """Convert Qwen grounding tags into the same structured shape as JSON."""
    if not boxes:
        return None
    refs = re.findall(r"<ref>\s*([^<]+?)\s*</ref>", text, flags=re.IGNORECASE)
    objects = [
        {
            "object_id": f"o{index}",
            "category": "unknown",
            "material": "unknown",
            "condition": "intact",
            "count": 1,
            "location": "grounded image region",
            "caption": ref.strip(),
        }
        for index, ref in enumerate(refs[: len(boxes)], 1)
    ]
    return {
        "objects": objects,
        "grounding_boxes": boxes,
        "caption": " ".join(obj["caption"] for obj in objects),
    }


def _extract_json_candidates(text: str) -> list[str]:
    """Extract balanced JSON objects, including objects inside prose."""
    candidates: list[str] = []
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    return candidates


@dataclass
class ParseResult:
    """Result of parsing a model's raw text output."""

    data: dict  # The parsed/coerced data dict (always present)
    parse_failed: bool  # True if we fell back to free-text
    raw_parsed: dict | None  # The raw parsed JSON before coercion (None if parse_failed)
    grounding_boxes: list[list[float]] | None  # Qwen grounding boxes if present


def parse_structured_output(text: str) -> ParseResult:
    """Three-stage JSON parser with free-text fallback.

    Stage 1: json.loads(text) directly.
    Stage 2: Extract the first {...} block via regex, then json.loads.
    Stage 3: Extract ```json...``` fence content, then json.loads.
    Fallback: Wrap raw text as {"caption": text, "parse_error": true}.
              Sets parse_failed=True. The text is NEVER dropped.

    Out-of-vocabulary fields (category, material, condition) are coerced
    to "unknown"/"intact" and the original value preserved in raw_parsed.

    Also extracts Qwen2.5-VL grounding boxes from <ref>/<box> tags if present.

    Returns:
        ParseResult with data, parse_failed, raw_parsed, grounding_boxes.
    """
    stripped = text.strip()

    # Always try to extract grounding boxes from the original text
    grounding_boxes = _extract_grounding_boxes(stripped)

    # Stage 1: direct parse
    parsed = _try_parse(stripped)

    # Stage 2: parse fenced JSON before searching prose. This avoids a
    # greedy prose match swallowing multiple unrelated brace blocks.
    if parsed is None:
        m2 = _CODE_FENCE_RE.search(stripped)
        if m2:
            parsed = _try_parse(m2.group(1))

    # Stage 3: extract balanced JSON from prose.
    if parsed is None:
        for candidate in _extract_json_candidates(stripped):
            parsed = _try_parse(candidate)
            if parsed is not None:
                break

    # Grounding-tag output is a valid structured result even though it is not
    # JSON. Preserve the raw text and expose normalized object data.
    if parsed is None and grounding_boxes:
        parsed = _grounding_data(stripped, grounding_boxes)

    # Fallback: free text
    if parsed is None:
        logger.debug("parse_structured_output: all stages failed, using free-text fallback")
        return ParseResult(
            data={"caption": stripped, "parse_error": True},
            parse_failed=True,
            raw_parsed=None,
            grounding_boxes=grounding_boxes,
        )

    # Coerce objects list if present
    raw = dict(parsed)
    if "objects" in parsed and isinstance(parsed["objects"], list):
        parsed["objects"] = [_normalise_object(o) for o in parsed["objects"]]

    # Coerce top-level single-object response (object_level prompt returns a flat dict)
    if "category" in parsed:
        parsed = _normalise_object(parsed)

    return ParseResult(
        data=parsed,
        parse_failed=False,
        raw_parsed=raw,
        grounding_boxes=grounding_boxes,
    )


# ---------------------------------------------------------------------------
# Image cropping utilities (used by CLI and app for object-level captions)
# ---------------------------------------------------------------------------


def crop_bbox(
    image: PILImageModule.Image,
    bbox: list[float],
) -> PILImageModule.Image:
    """Crop a PIL image to a [x, y, w, h] bounding box with a small padding.

    Args:
        image: PIL Image (RGB).
        bbox:  [x, y, w, h] in pixels.

    Returns:
        Cropped PIL Image.
    """

    x, y, w, h = bbox
    pad = max(5, min(w, h) * 0.05)
    x1 = max(0, int(x - pad))
    y1 = max(0, int(y - pad))
    x2 = min(image.width, int(x + w + pad))
    y2 = min(image.height, int(y + h + pad))
    cropped = image.crop((x1, y1, x2, y2))
    if cropped.mode != "RGB":
        cropped = cropped.convert("RGB")
    return cropped
