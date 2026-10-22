"""
BLIP-2 OPT-2.7B caption model backend.

Model: Salesforce/blip2-opt-2.7b
VRAM: ~3.2 GB at float16; ~2.0 GB at 4-bit (bitsandbytes)
Quantization: controlled by WASTECAP_QUANTIZE env var (none|4bit|8bit)
Device: controlled by WASTECAP_DEVICE env var (cuda|cpu)

Lazy loading: weights are NOT loaded at import time. They are loaded on
the first call to generate() and cached as a module-level singleton via
the registry's @lru_cache.

BLIP-2 is a frozen-encoder + Q-Former + language model architecture. The
OPT-2.7B variant is autoregressive and responds to text prompts appended
to the image encoding. It does NOT produce bounding boxes.

For n > 1 we sample with temperature=0.9, top_p=0.95 to get diversity.
For n == 1 we use greedy decoding (deterministic, reproducible).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "Salesforce/blip2-opt-2.7b"
_MAX_NEW_TOKENS = 256


class Blip2Model(CaptionModel):
    """BLIP-2 OPT-2.7B caption backend.

    Weights are loaded on first generate() call, not at import time.
    """

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "blip2"

    @property
    def supports_grounding(self) -> bool:
        return False

    def _load(self) -> None:
        """Lazily load the model and processor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as exc:
            raise ImportError(
                "BLIP-2 requires torch and transformers. "
                "Install the [ml] extras: pip install -e '.[ml]'"
            ) from exc

        device = os.environ.get("WASTECAP_DEVICE", "cpu")
        quantize = os.environ.get("WASTECAP_QUANTIZE", "none").lower()

        logger.info("Loading BLIP-2 (%s) on %s with quantize=%s …", _HF_MODEL_ID, device, quantize)

        self._processor = Blip2Processor.from_pretrained(_HF_MODEL_ID)

        load_kwargs: dict = {"device_map": "auto" if device == "cuda" else None}

        if quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to float16")
                quantize = "none"

        if quantize == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to float16")
                quantize = "none"

        if quantize == "none":
            if device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
            else:
                load_kwargs["torch_dtype"] = torch.float32

        if device != "cuda":
            load_kwargs.pop("device_map", None)

        self._model = Blip2ForConditionalGeneration.from_pretrained(_HF_MODEL_ID, **load_kwargs)

        if device != "cuda" or quantize == "none":
            self._model = self._model.to(device)

        self._device = device
        logger.info("BLIP-2 loaded on %s", device)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "blip2_scene_v1",
        **kwargs: object,
    ) -> list[Candidate]:
        """Generate n caption candidates.

        Args:
            image:     PIL Image (RGB).
            prompt:    Rendered prompt string.
            n:         Number of candidates. n=1 uses greedy; n>1 samples.
            prompt_id: Stored on each Candidate for traceability.

        Returns:
            List of n Candidate objects.
        """
        import torch

        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_new_tokens = int(kwargs.get("max_new_tokens", _MAX_NEW_TOKENS))

        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        candidates: list[Candidate] = []

        with torch.inference_mode():
            if n == 1:
                # Greedy — deterministic
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
                texts = [self._processor.decode(output_ids[0], skip_special_tokens=True)]
            else:
                # Sampling — diversity across n candidates
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    num_return_sequences=n,
                    temperature=0.9,
                    top_p=0.95,
                )
                texts = [
                    self._processor.decode(ids, skip_special_tokens=True) for ids in output_ids
                ]

        for text in texts:
            candidates.append(
                Candidate(
                    model=self.name,
                    prompt_id=prompt_id,
                    text=text.strip(),
                    created_at=datetime.now(tz=UTC),
                )
            )

        return candidates
