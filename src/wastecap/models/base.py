"""Abstract base class for all caption model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from wastecap.schema import Candidate


class CaptionModel(ABC):
    """Interface all backends must implement.

    The UI and CLI only interact with models through this interface via registry.py.
    No backend is ever imported directly by the app.
    """

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "default",
        **kwargs: object,
    ) -> list[Candidate]:
        """Generate n candidate captions for the given image and prompt.

        Args:
            image:     PIL image (RGB).
            prompt:    Prompt string (already rendered from template).
            n:         Number of candidates to generate.
            prompt_id: Identifier of the prompt template, stored in each Candidate.
            **kwargs:  Model-specific kwargs (e.g. max_new_tokens, temperature).

        Returns:
            List of Candidate objects, length n.  Never empty.
        """
        ...

    @property
    def name(self) -> str:
        """Backend identifier string, e.g. 'blip2'."""
        raise NotImplementedError

    @property
    def supports_grounding(self) -> bool:
        """True if this backend returns bounding boxes in its output."""
        return False
