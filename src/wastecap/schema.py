"""
Pydantic models for WasteCap annotation records.

This is the canonical schema, frozen after M0.
Any changes require explicit approval; update store.py migration and
tests/test_schema.py together.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations (values must match configs/taxonomy.yaml)
# ---------------------------------------------------------------------------


class WasteCategory(str, Enum):
    bottle = "bottle"
    can = "can"
    cup = "cup"
    carton = "carton"
    bag = "bag"
    wrapper = "wrapper"
    container = "container"
    box = "box"
    straw = "straw"
    lid = "lid"
    tube = "tube"
    food_waste = "food_waste"
    electronic = "electronic"
    clothing = "clothing"
    other = "other"
    unknown = "unknown"


class Material(str, Enum):
    plastic = "plastic"
    metal = "metal"
    paper = "paper"
    cardboard = "cardboard"
    glass = "glass"
    organic = "organic"
    textile = "textile"
    mixed = "mixed"
    unknown = "unknown"


class Condition(str, Enum):
    intact = "intact"
    crushed = "crushed"
    torn = "torn"
    dirty = "dirty"
    deformed = "deformed"
    partially_occluded = "partially_occluded"


class SortingAction(str, Enum):
    recycle_plastic = "recycle_plastic"
    recycle_metal = "recycle_metal"
    recycle_paper = "recycle_paper"
    recycle_cardboard = "recycle_cardboard"
    recycle_glass = "recycle_glass"
    compost = "compost"
    landfill = "landfill"
    hazardous = "hazardous"


class RecordStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    edited = "edited"


class DataSource(str, Enum):
    taco = "taco"
    zerowaste = "zerowaste"
    coco = "coco"
    web = "web"


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


# ---------------------------------------------------------------------------
# Mask: typed union so RLE and polygon are unambiguous
# ---------------------------------------------------------------------------


class RLEMask(BaseModel):
    type: Literal["rle"] = "rle"
    counts: str = Field(..., description="COCO-format RLE string")
    size: list[int] = Field(..., min_length=2, max_length=2, description="[height, width]")


class PolygonMask(BaseModel):
    type: Literal["polygon"] = "polygon"
    data: list[list[float]] = Field(
        ..., description="List of polygon rings, each a flat [x, y, x, y, ...] list"
    )


# ---------------------------------------------------------------------------
# Candidate: one raw output from a caption model
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    model: str = Field(..., description="Backend name, e.g. 'blip2' or 'qwen_vl'")
    prompt_id: str = Field(..., description="Identifier of the prompt template used")
    text: str = Field(..., description="Raw model output text")
    clip_score: float | None = Field(
        default=None, description="CLIPScore; None until computed by compute_clipscore.py"
    )
    parse_failed: bool = Field(
        default=False,
        description="True if structured-output JSON parsing failed; text preserved as-is",
    )
    raw_parsed: dict[str, Any] | None = Field(
        default=None,
        description="Full parsed JSON from model, including out-of-vocab fields",
    )
    grounding_boxes: list[list[float]] | None = Field(
        default=None,
        description="Bounding boxes returned by grounding-capable models (e.g. Qwen2.5-VL). "
        "Format: [[x, y, w, h], ...]. Stored for evaluation; not ground truth.",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# ObjectAnnotation: one object instance within an image
# ---------------------------------------------------------------------------


class ObjectAnnotation(BaseModel):
    object_id: str = Field(..., description="Unique within the image, e.g. 'o1', 'o2'")
    category: WasteCategory = Field(default=WasteCategory.unknown)
    material: Material = Field(default=Material.unknown)
    condition: Condition | None = Field(default=None)
    count: int = Field(
        default=1,
        ge=1,
        description="Always 1 per ObjectAnnotation entry. One annotation = one instance.",
    )
    location: str | None = Field(
        default=None, description="Free-text spatial reference, e.g. 'lower left, next to a can'"
    )
    bbox: list[float] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="[x, y, width, height] in pixel coordinates",
    )
    mask: RLEMask | PolygonMask | None = Field(default=None)
    caption: str | None = Field(
        default=None,
        description="Object-level caption following style: '<condition> <material> <object>'",
    )
    sorting_action: SortingAction | None = Field(default=None)
    instruction: str | None = Field(
        default=None,
        description="Natural-language sorting instruction grounded to this object",
    )

    @field_validator("bbox")
    @classmethod
    def bbox_non_negative(cls, v: list[float] | None) -> list[float] | None:
        if v is not None and any(x < 0 for x in v):
            raise ValueError("bbox values must be non-negative")
        return v

    @model_validator(mode="after")
    def instruction_requires_bbox(self) -> ObjectAnnotation:
        if self.instruction is not None and self.bbox is None:
            raise ValueError(
                "An object with an instruction must have a bbox. "
                "Set bbox before writing the instruction."
            )
        return self


# ---------------------------------------------------------------------------
# Review: human review decision
# ---------------------------------------------------------------------------


class Review(BaseModel):
    reviewer: str = Field(..., description="Reviewer identifier from WASTECAP_REVIEWER env var")
    notes: str = Field(default="")
    hallucination: bool = Field(
        default=False,
        description="True if the approved caption names objects/materials not visible in the image",
    )
    reviewed_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# AnnotationRecord: top-level record for one image
# ---------------------------------------------------------------------------


class AnnotationRecord(BaseModel):
    image_id: str = Field(
        ..., description="Globally unique, e.g. 'taco_000123' or 'zerowaste_00042'"
    )
    file: str = Field(
        ...,
        description=(
            "Relative path from project root, e.g. 'data/processed/images/taco_000123.jpg'"
        ),
    )
    source: DataSource
    license: str = Field(default="", description="License string for the source image")
    width: int = Field(..., ge=0)
    height: int = Field(..., ge=0)
    split: Split | None = Field(default=None, description="Assigned after splits.py runs")
    scene_context: str | None = Field(
        default=None,
        description="One sentence: clutter level, occlusion, surroundings",
    )
    caption: str | None = Field(
        default=None, description="Final human-approved image-level caption"
    )
    objects: list[ObjectAnnotation] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    status: RecordStatus = Field(default=RecordStatus.pending)
    review: Review | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def approved_requires_review(self) -> AnnotationRecord:
        """Approved records must have a review entry."""
        if self.status == RecordStatus.approved and self.review is None:
            raise ValueError(
                "status='approved' requires a Review entry. "
                "Only set approved via the UI's approve action."
            )
        return self

    @model_validator(mode="after")
    def unique_object_ids(self) -> AnnotationRecord:
        ids = [o.object_id for o in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate object_ids in record {self.image_id}: {ids}")
        return self

    def model_post_init(self, __context: Any) -> None:  # noqa: ANN401
        """Keep updated_at fresh on every mutation."""
        # Pydantic v2: called after __init__. updated_at is set at construction;
        # callers must update it explicitly when mutating a stored record.
        pass
