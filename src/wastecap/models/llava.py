"""LLaVA caption model backend — optional stub (implemented in M2 if resources allow)."""

from __future__ import annotations

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate


class LlavaModel(CaptionModel):
    """LLaVA-1.5-7B backend. Optional; requires sufficient VRAM."""

    @property
    def name(self) -> str:
        return "llava"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "default",
        **kwargs: object,
    ) -> list[Candidate]:
        raise NotImplementedError("LLaVA backend is implemented in M2 (optional).")
