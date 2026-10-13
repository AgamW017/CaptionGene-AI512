"""Qwen2.5-VL caption model backend — stub (implemented in M2)."""

from __future__ import annotations

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate


class QwenVLModel(CaptionModel):
    """Qwen2.5-VL-3B-Instruct backend. Supports grounding output."""

    @property
    def name(self) -> str:
        return "qwen_vl"

    @property
    def supports_grounding(self) -> bool:
        return True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "default",
        **kwargs: object,
    ) -> list[Candidate]:
        raise NotImplementedError("Qwen2.5-VL backend is implemented in M2.")
