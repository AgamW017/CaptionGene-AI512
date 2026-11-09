"""Run a short, deterministic mock-backend demo without a database."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wastecap.models.registry import clear_cache, get_model
from wastecap.prompts import parse_structured_output, render_prompt


def main() -> int:
    os.environ["WASTECAP_BACKEND"] = "mock"
    os.environ["WASTECAP_DEVICE"] = "cpu"
    clear_cache()
    image_path = Path("tests/fixtures/images/taco_fix001.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Demo fixture not found: {image_path}")
    prompt = render_prompt("mock", "scene_level")
    with Image.open(image_path) as image:
        candidate = get_model("mock").generate(
            image.convert("RGB"),
            prompt.text,
            n=1,
            prompt_id=prompt.prompt_id,
        )[0]
    parsed = parse_structured_output(candidate.text)
    print(f"image: {image_path}")
    print(f"caption: {candidate.text}")
    print(f"parsed: {parsed.data}")
    print("Next: run `streamlit run app/main.py` for the interactive demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
