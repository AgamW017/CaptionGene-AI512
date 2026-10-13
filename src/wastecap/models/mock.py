"""Mock caption model backend.

Deterministic, GPU-free, fixture-driven.
Used in all tests and for local development without weights.

Behaviour is controlled by the WASTECAP_MOCK_MODE env var:
  normal (default) — returns well-formed JSON captions
  parse_fail       — returns intentionally malformed JSON (tests parser fallback)
  free_text        — returns plain text (no JSON wrapper)
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate

# A small bank of deterministic fixture captions.
_CAPTION_BANK = [
    {
        "objects": [
            {
                "object_id": "o1",
                "category": "bottle",
                "material": "plastic",
                "condition": "crushed",
                "count": 1,
                "location": "centre",
                "caption": "crushed plastic bottle on pavement",
            }
        ],
        "scene_context": "outdoor, cluttered, partially occluded",
    },
    {
        "objects": [
            {
                "object_id": "o1",
                "category": "can",
                "material": "metal",
                "condition": "deformed",
                "count": 1,
                "location": "lower left",
                "caption": "deformed metal can next to grass",
            }
        ],
        "scene_context": "outdoor, low clutter",
    },
    {
        "objects": [
            {
                "object_id": "o1",
                "category": "wrapper",
                "material": "plastic",
                "condition": "torn",
                "count": 1,
                "location": "upper right",
                "caption": "torn plastic wrapper on ground",
            }
        ],
        "scene_context": "outdoor, moderate clutter",
    },
    {
        "objects": [
            {
                "object_id": "o1",
                "category": "carton",
                "material": "cardboard",
                "condition": "dirty",
                "count": 1,
                "location": "centre left",
                "caption": "dirty cardboard carton partially crushed",
            }
        ],
        "scene_context": "indoor, recycling area",
    },
    {
        "objects": [
            {
                "object_id": "o1",
                "category": "cup",
                "material": "paper",
                "condition": "intact",
                "count": 1,
                "location": "centre",
                "caption": "intact paper cup with plastic lid",
            }
        ],
        "scene_context": "outdoor, park bench",
    },
]


def _seed_index(image_id_or_hash: str, candidate_index: int) -> int:
    """Deterministic index into _CAPTION_BANK from image identifier + candidate index."""
    h = hashlib.md5(f"{image_id_or_hash}:{candidate_index}".encode()).hexdigest()
    return int(h, 16) % len(_CAPTION_BANK)


class MockModel(CaptionModel):
    """Deterministic mock that never loads weights."""

    @property
    def name(self) -> str:
        return "mock"

    @property
    def supports_grounding(self) -> bool:
        return False

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "default",
        **kwargs: object,
    ) -> list[Candidate]:
        mode = os.environ.get("WASTECAP_MOCK_MODE", "normal").lower()
        # Use image size as a stable identifier when no image_id is available.
        img_key = f"{image.size[0]}x{image.size[1]}"

        candidates: list[Candidate] = []
        for i in range(n):
            if mode == "parse_fail":
                text = "{ bad json :"
                candidates.append(
                    Candidate(
                        model=self.name,
                        prompt_id=prompt_id,
                        text=text,
                        parse_failed=True,
                        raw_parsed=None,
                        created_at=datetime.now(tz=UTC),
                    )
                )
            elif mode == "free_text":
                text = "A crushed plastic bottle lies on the pavement."
                candidates.append(
                    Candidate(
                        model=self.name,
                        prompt_id=prompt_id,
                        text=text,
                        parse_failed=False,
                        raw_parsed={"caption": text},
                        created_at=datetime.now(tz=UTC),
                    )
                )
            else:
                idx = _seed_index(img_key, i)
                payload = _CAPTION_BANK[idx]
                text = json.dumps(payload)
                candidates.append(
                    Candidate(
                        model=self.name,
                        prompt_id=prompt_id,
                        text=text,
                        parse_failed=False,
                        raw_parsed=payload,
                        created_at=datetime.now(tz=UTC),
                    )
                )

        return candidates
