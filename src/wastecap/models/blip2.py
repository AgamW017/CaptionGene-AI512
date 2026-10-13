"""BLIP-2 caption model backend — stub (implemented in M2)."""

from __future__ import annotations

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate


class Blip2Model(CaptionModel):
    """BLIP-2 OPT-2.7B backend. Requires GPU and HF weights download."""

    @property
    def name(self) -> str:
        return "blip2"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "default",
        **kwargs: object,
    ) -> list[Candidate]:
        raise NotImplementedError("BLIP-2 backend is implemented in M2.")
