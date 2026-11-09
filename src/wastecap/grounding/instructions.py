"""
Natural-language sorting instruction generator for WasteCap.

Each instruction is grounded to a specific object via its object_id and
bounding box. The instruction names:
  1. The object (condition + material + category)
  2. Its spatial location in the image (derived from bbox)
  3. The sorting action (what bin/action to take)

Design constraints (from CLAUDE.md):
  - Every instruction must point to an ObjectAnnotation with a valid bbox.
  - Instructions are stored in ObjectAnnotation.instruction.
  - They are written to the DB only after explicit human approval on the
    Instructions page.
  - Qwen2.5-VL predicted boxes are evaluated against reference boxes for
    IoU accuracy in M7 — they are NOT used to generate instructions.

Public API
----------
  generate_instruction(obj, image_width, image_height) -> str
  validate_instruction(obj) -> tuple[bool, str]
  generate_all_instructions(rec, image_width, image_height) -> AnnotationRecord
  query_qwen_grounding(image, object_desc, model) -> list[list[float]] | None
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from wastecap.models.base import CaptionModel

from wastecap.schema import (
    AnnotationRecord,
    ObjectAnnotation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action → instruction phrase templates
# ---------------------------------------------------------------------------

# {object} is replaced with the described object string
# {location} is replaced with the spatial description
_ACTION_TEMPLATES: dict[str, str] = {
    "recycle_plastic": (
        "Pick up the {object} at the {location} and place it in the plastic recycling bin."
    ),
    "recycle_metal": (
        "Pick up the {object} at the {location} and place it in the metal recycling bin."
    ),
    "recycle_paper": (
        "Pick up the {object} at the {location} and place it in the paper recycling bin."
    ),
    "recycle_cardboard": (
        "Flatten the {object} at the {location} and place it in the cardboard recycling bin."
    ),
    "recycle_glass": (
        "Carefully pick up the {object} at the {location} and place it in the glass recycling bin."
    ),
    "compost": (
        "Pick up the {object} at the {location} and place it in the compost or organic waste bin."
    ),
    "landfill": (
        "Pick up the {object} at the {location} and place it in the general waste (landfill) bin."
    ),
    "hazardous": (
        "Do not handle the {object} at the {location} with bare hands — "
        "it requires special hazardous-waste disposal."
    ),
}

# Condition → adjective for the description
_CONDITION_ADJECTIVE: dict[str, str] = {
    "intact": "intact",
    "crushed": "crushed",
    "torn": "torn",
    "dirty": "dirty",
    "deformed": "deformed",
    "partially_occluded": "partially occluded",
}

# Category → noun (singular)
_CATEGORY_NOUN: dict[str, str] = {
    "bottle": "bottle",
    "can": "can",
    "cup": "cup",
    "carton": "carton",
    "bag": "bag",
    "wrapper": "wrapper",
    "container": "container",
    "box": "box",
    "straw": "straw",
    "lid": "lid",
    "tube": "tube",
    "food_waste": "food waste",
    "electronic": "electronic item",
    "clothing": "piece of clothing",
    "other": "waste item",
    "unknown": "waste item",
}

# Material → adjective (used only when not "unknown" or "mixed")
_MATERIAL_ADJECTIVE: dict[str, str] = {
    "plastic": "plastic",
    "metal": "metal",
    "paper": "paper",
    "cardboard": "cardboard",
    "glass": "glass",
    "organic": "organic",
    "textile": "textile",
    "mixed": "",  # omit from description
    "unknown": "",  # omit from description
}


# ---------------------------------------------------------------------------
# Object description helper
# ---------------------------------------------------------------------------


def _describe_object(obj: ObjectAnnotation) -> str:
    """Build a compact description: '<condition> <material> <category>'.

    Examples:
        crushed plastic bottle
        deformed metal can
        intact glass bottle
        waste item      (when category and material are unknown)
    """
    parts: list[str] = []

    if obj.condition is not None:
        adj = _CONDITION_ADJECTIVE.get(obj.condition.value, "")
        if adj:
            parts.append(adj)

    mat_adj = _MATERIAL_ADJECTIVE.get(obj.material.value, "")
    if mat_adj:
        parts.append(mat_adj)

    noun = _CATEGORY_NOUN.get(obj.category.value, "waste item")
    parts.append(noun)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_instruction(
    obj: ObjectAnnotation,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """Generate a grounded natural-language sorting instruction for one object.

    Args:
        obj:          The ObjectAnnotation to generate an instruction for.
                      Must have a valid bbox and a sorting_action.
        image_width:  Image width in pixels (used for spatial description).
                      Pass 0 to skip spatial description.
        image_height: Image height in pixels.

    Returns:
        Instruction string, e.g.:
        "Pick up the crushed plastic bottle at the lower left and place
        it in the plastic recycling bin."

    Raises:
        ValueError: if obj has no bbox (instructions require grounding).
        ValueError: if obj has no sorting_action.
    """
    if obj.bbox is None:
        raise ValueError(
            f"Cannot generate instruction for object {obj.object_id!r}: "
            "bbox is required for grounded instructions. "
            "Set the bounding box before calling generate_instruction."
        )

    if obj.sorting_action is None:
        raise ValueError(
            f"Cannot generate instruction for object {obj.object_id!r}: sorting_action is required."
        )

    # Build object description
    obj_desc = _describe_object(obj)

    # Build location string
    if image_width > 0 and image_height > 0:
        from wastecap.grounding.boxes import spatial_description

        location = spatial_description(obj.bbox, image_width, image_height)
    elif obj.location:
        location = obj.location
    else:
        location = "the image"

    # Get the action template
    action_key = obj.sorting_action.value
    template = _ACTION_TEMPLATES.get(action_key)
    if template is None:
        # Fallback for future action values not yet in the template dict
        logger.warning("No template for sorting_action %r — using generic", action_key)
        template = "Pick up the {object} at the {location} and dispose of it appropriately."

    return template.format(object=obj_desc, location=location)


def validate_instruction(
    obj: ObjectAnnotation,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[bool, str]:
    """Validate that an ObjectAnnotation is ready for an approved instruction.

    Checks:
      - bbox is present, positive, and within image bounds when supplied
      - an optional mask can be converted to a valid bbox
      - sorting_action is set
      - instruction is not empty (if already set)

    Returns:
        (is_valid, reason). reason is "" when valid.
    """
    if obj.bbox is None:
        return False, f"Object {obj.object_id!r} has no bbox."

    from wastecap.grounding.boxes import validate_bbox

    bbox_valid, bbox_reason = validate_bbox(obj.bbox, image_width, image_height)
    if not bbox_valid:
        return False, f"Object {obj.object_id!r} has invalid bbox: {bbox_reason}"

    if obj.mask is not None:
        from wastecap.grounding.masks import mask_to_bbox

        mask_bbox = mask_to_bbox(obj.mask)
        if mask_bbox is None:
            return False, f"Object {obj.object_id!r} has an invalid mask."
        mask_valid, mask_reason = validate_bbox(mask_bbox, image_width, image_height)
        if not mask_valid:
            return False, f"Object {obj.object_id!r} has an invalid mask: {mask_reason}"

    if obj.sorting_action is None:
        return False, f"Object {obj.object_id!r} has no sorting_action."

    if obj.instruction is not None and not obj.instruction.strip():
        return False, f"Object {obj.object_id!r} has an empty instruction string."

    return True, ""


def generate_all_instructions(
    rec: AnnotationRecord,
    image_width: int = 0,
    image_height: int = 0,
    overwrite: bool = False,
) -> AnnotationRecord:
    """Generate instructions for every object in a record that needs one.

    Args:
        rec:          AnnotationRecord whose objects will be updated.
        image_width:  Used for spatial descriptions (pass rec.width).
        image_height: Used for spatial descriptions (pass rec.height).
        overwrite:    If True, regenerate even for objects that already
                      have an instruction. Default False (safe).

    Returns:
        A copy of the record with instructions filled in for eligible objects.
        Does NOT write to the DB — caller decides when to persist.
    """
    updated_objects: list[ObjectAnnotation] = []
    for obj in rec.objects:
        if not overwrite and obj.instruction is not None:
            updated_objects.append(obj)
            continue

        valid, reason = validate_instruction(obj, rec.width, rec.height)
        if not valid:
            logger.debug(
                "Skipping instruction for %s/%s: %s",
                rec.image_id,
                obj.object_id,
                reason,
            )
            updated_objects.append(obj)
            continue

        try:
            instr = generate_instruction(obj, image_width, image_height)
            updated_objects.append(obj.model_copy(update={"instruction": instr}))
        except ValueError as exc:
            logger.warning(
                "Failed to generate instruction for %s/%s: %s",
                rec.image_id,
                obj.object_id,
                exc,
            )
            updated_objects.append(obj)

    return rec.model_copy(update={"objects": updated_objects})


def validate_all_instructions(rec: AnnotationRecord) -> list[tuple[str, str]]:
    """Validate every approved instruction in a record.

    Returns:
        List of (object_id, error_message) for invalid instructions.
        Empty list means all instructions are valid (or none are set).
    """
    errors: list[tuple[str, str]] = []
    for obj in rec.objects:
        if obj.instruction is None:
            continue
        valid, reason = validate_instruction(obj, rec.width, rec.height)
        if not valid:
            errors.append((obj.object_id, reason))
    return errors


# ---------------------------------------------------------------------------
# Qwen2.5-VL grounding query
# ---------------------------------------------------------------------------


def query_qwen_grounding(
    image: PILImage,
    object_description: str,
    model: CaptionModel,
    image_width: int = 0,
    image_height: int = 0,
) -> list[list[float]] | None:
    """Ask Qwen2.5-VL to locate an object and return predicted pixel boxes.

    This sends a grounding prompt to a Qwen2.5-VL model instance and
    extracts the predicted bounding boxes from the model output.
    Boxes are stored for IoU evaluation in M7 — they are NOT ground truth.

    Args:
        image:              PIL Image (RGB).
        object_description: Short description, e.g. "crushed plastic bottle".
        model:              A QwenVLModel (or MockModel for tests).
        image_width:        Image width in pixels (for coordinate conversion).
        image_height:       Image height in pixels.

    Returns:
        List of [x, y, w, h] predicted boxes in pixel coordinates, or None
        if no boxes were returned / model doesn't support grounding.
    """
    if not model.supports_grounding:
        logger.debug(
            "Model %r does not support grounding — skipping Qwen grounding query.",
            model.name,
        )
        return None

    grounding_prompt = (
        f"Locate exactly one instance of {object_description} in this image. "
        "Return only this format, with integer coordinates normalized from 0 to 999: "
        f"<ref>{object_description}</ref><box>[[x1,y1,x2,y2]]</box>. "
        "Do not describe the image, add JSON, markdown, or any other text."
    )

    try:
        candidates = model.generate(
            image,
            grounding_prompt,
            n=1,
            prompt_id="grounding_query_v1",
        )
    except Exception as exc:
        logger.warning("Qwen grounding query failed: %s", exc)
        return None

    if not candidates:
        return None

    cand = candidates[0]

    # Grounding boxes may be in Candidate.grounding_boxes (parsed during generate)
    if cand.grounding_boxes:
        if image_width > 0 and image_height > 0:
            # Qwen returns 0-999 normalised; convert to pixel xywh
            from wastecap.grounding.boxes import qwen_to_pixel, xyxy_to_xywh

            return [
                xyxy_to_xywh(qwen_to_pixel(b, image_width, image_height))
                for b in cand.grounding_boxes
            ]
        return cand.grounding_boxes

    # Try parsing grounding boxes from raw text
    from wastecap.prompts import parse_structured_output

    result = parse_structured_output(cand.text)
    if result.grounding_boxes and image_width > 0 and image_height > 0:
        from wastecap.grounding.boxes import qwen_to_pixel, xyxy_to_xywh

        return [
            xyxy_to_xywh(qwen_to_pixel(b, image_width, image_height))
            for b in result.grounding_boxes
        ]

    return result.grounding_boxes
