"""
Qwen2.5-VL-3B-Instruct caption model backend.

Model: Qwen/Qwen2.5-VL-3B-Instruct
VRAM: ~4.0 GB at bfloat16; ~2.5 GB at 4-bit (bitsandbytes)
Quantization: controlled by WASTECAP_QUANTIZE (none|4bit|8bit)
Device: controlled by WASTECAP_DEVICE (cuda|cpu)

Qwen2.5-VL is a chat-tuned model that:
- Accepts system + user messages (chat template format)
- Can emit grounding tags: <ref>text</ref><box>[[x1,y1,x2,y2]]</box>
  where coordinates are normalised to 0-999 range over the input image
- Supports longer, richer outputs than BLIP-2

For n > 1 we cannot call generate() with num_return_sequences on chat
models reliably; instead we call generate() n times with sampling.
For n == 1 we use greedy (deterministic).

Grounding box parsing is handled by wastecap.prompts.parse_structured_output
which extracts <ref>/<box> tags from the raw text and stores them in
Candidate.grounding_boxes. These are NOT ground-truth boxes; they are
a model prediction stored for later IoU evaluation (M7).
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from PIL import Image

from wastecap.models.base import CaptionModel
from wastecap.schema import Candidate

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
_MAX_NEW_TOKENS = 512


class QwenVLModel(CaptionModel):
    """Qwen2.5-VL-3B-Instruct caption backend with optional grounding."""

    def __init__(self) -> None:
        self._processor = None
        self._model = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "qwen_vl"

    @property
    def supports_grounding(self) -> bool:
        return True

    def _load(self) -> None:
        """Lazily load model and processor."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise ImportError(
                "Qwen2.5-VL requires torch and transformers>=4.45. "
                "Install the [ml] extras: pip install -e '.[ml]'"
            ) from exc

        device = os.environ.get("WASTECAP_DEVICE", "cpu")
        quantize = os.environ.get("WASTECAP_QUANTIZE", "none").lower()

        logger.info(
            "Loading Qwen2.5-VL (%s) on %s with quantize=%s …",
            _HF_MODEL_ID,
            device,
            quantize,
        )

        self._processor = AutoProcessor.from_pretrained(_HF_MODEL_ID, trust_remote_code=True)

        load_kwargs: dict = {}

        if device == "cuda":
            load_kwargs["device_map"] = "auto"

        if quantize == "4bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to bfloat16")
                quantize = "none"

        if quantize == "8bit":
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            except ImportError:
                logger.warning("bitsandbytes not installed — falling back to bfloat16")
                quantize = "none"

        if quantize == "none":
            load_kwargs["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            trust_remote_code=True,
            **load_kwargs,
        )

        if device != "cuda" or quantize == "none":
            self._model = self._model.to(device)

        self._device = device
        logger.info("Qwen2.5-VL loaded on %s", device)

    def _build_messages(
        self,
        image: Image.Image,
        prompt: str,
        system: str | None,
    ) -> list[dict]:
        """Build the chat messages list for the processor."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )
        return messages

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        n: int = 1,
        prompt_id: str = "qwen_scene_v1",
        system: str | None = None,
        **kwargs: object,
    ) -> list[Candidate]:
        """Generate n caption candidates.

        Args:
            image:     PIL Image (RGB).
            prompt:    Rendered prompt string (user message content).
            n:         Number of candidates.
            prompt_id: Stored on each Candidate.
            system:    Optional system message (from RenderedPrompt.system).

        Returns:
            List of n Candidate objects. Raw grounding text is preserved in
            Candidate.text; Candidate.grounding_boxes is populated by the
            CLI/caller via parse_structured_output.
        """
        import torch

        self._load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        max_new_tokens = int(kwargs.get("max_new_tokens", _MAX_NEW_TOKENS))
        messages = self._build_messages(image, prompt, system)

        # Apply chat template
        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        candidates: list[Candidate] = []
        with torch.inference_mode():
            for _i in range(n):
                inputs = self._processor(
                    text=[text_input],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
                if self._device == "cuda":
                    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                do_sample = n > 1
                gen_kwargs: dict = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }
                if do_sample:
                    gen_kwargs.update({"temperature": 0.9, "top_p": 0.95})

                output_ids = self._model.generate(**inputs, **gen_kwargs)

                # Decode only the newly generated tokens
                input_len = inputs["input_ids"].shape[1]
                new_ids = output_ids[:, input_len:]
                raw_text = self._processor.batch_decode(
                    new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                candidates.append(
                    Candidate(
                        model=self.name,
                        prompt_id=prompt_id,
                        text=raw_text,
                        created_at=datetime.now(tz=UTC),
                    )
                )

        return candidates
