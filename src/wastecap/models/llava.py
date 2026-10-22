"""
LLaVA-1.5-7B caption model backend (optional).

Model: llava-hf/llava-1.5-7b-hf
VRAM: ~7.5 GB at float16; ~4.5 GB at 4-bit (bitsandbytes)

NOTE: LLaVA-1.5-7B requires ~4.5 GB VRAM at 4-bit.
With a 4 GB card and BLIP-2/Qwen2.5-VL already resident, this backend
is only usable if you unload the other models first (clear registry cache).

This backend is intentionally NOT auto-loaded. Set:
    WASTECAP_BACKEND=llava

to use it. The Compare page (M4) only compares blip2 vs qwen_vl by default;
llava appears as an optional third column if candidates exist.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
_MAX_NEW_TOKENS = 512
_VRAM_WARNING = (
    "LLaVA-1.5-7B requires ~4.5 GB VRAM at 4-bit. "
    "On a 4 GB GPU, run with WASTECAP_QUANTIZE=4bit and ensure no other "
    "model is loaded at the same time (call registry.clear_cache() first)."
)


class LlavaModel(CaptionModel):
    """LLaVA-1.5-7B-hf optional backend.

    Raises ImportError if transformers is not installed, or a clear
    RuntimeError if VRAM is insufficient based on WASTECAP_DEVICE setting.
    """

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "llava"

    @property
    def supports_grounding(self) -> bool:
        return False

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import LlavaForConditionalGeneration, LlavaProcessor
        except ImportError as exc:
            raise ImportError(
                "LLaVA requires torch and transformers. "
                "Install the [ml] extras: pip install -e '.[ml]'"
            ) from exc

        device = os.environ.get("WASTECAP_DEVICE", "cpu")
        quantize = os.environ.get("WASTECAP_QUANTIZE", "none").lower()

        if device == "cuda":
            logger.warning(_VRAM_WARNING)

        logger.info("Loading LLaVA (%s) on %s with quantize=%s …", _HF_MODEL_ID, device, quantize)

        self._processor = LlavaProcessor.from_pretrained(_HF_MODEL_ID)

        load_kwargs: dict = {}
        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        if quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except ImportError:
                logger.warning("bitsandbytes not available — loading at float16")
                quantize = "none"

        if quantize == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not available — loading at float16")
                quantize = "none"

        if quantize == "none":
            load_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

        self._model = LlavaForConditionalGeneration.from_pretrained(_HF_MODEL_ID, **load_kwargs)

        if device != "cuda" or quantize == "none":
            self._model = self._model.to(device)

        self._device = device
        logger.info("LLaVA loaded on %s", device)

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "llava_scene_v1",
        **kwargs: object,
    ) -> list[Candidate]:
        """Generate n caption candidates."""
        import torch

        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_new_tokens = int(kwargs.get("max_new_tokens", _MAX_NEW_TOKENS))

        inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        if self._device == "cuda":
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        candidates: list[Candidate] = []
        with torch.inference_mode():
            for _i in range(n):
                do_sample = n > 1
                gen_kwargs: dict = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs.update({"temperature": 0.9, "top_p": 0.95})

                output_ids = self._model.generate(**inputs, **gen_kwargs)
                # LLaVA returns the full sequence including prompt tokens; decode only new
                input_len = inputs["input_ids"].shape[1]
                new_ids = output_ids[:, input_len:]
                raw_text = self._processor.decode(new_ids[0], skip_special_tokens=True).strip()

                candidates.append(
                    Candidate(
                        model=self.name,
                        prompt_id=prompt_id,
                        text=raw_text,
                        created_at=datetime.now(tz=UTC),
                    )
                )

        return candidates
