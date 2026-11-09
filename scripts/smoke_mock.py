"""Fresh-clone smoke check: fixture data, mock generation, and schema loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image

from wastecap.models.registry import clear_cache, get_model
from wastecap.prompts import render_prompt
from wastecap.schema import Candidate
from wastecap.store import init_db


def main() -> int:
    os.environ["WASTECAP_BACKEND"] = "mock"
    os.environ["WASTECAP_DEVICE"] = "cpu"
    with tempfile.TemporaryDirectory(prefix="wastecap-smoke-") as directory:
        db_path = Path(directory) / "wastecap.db"
        init_db(db_path)
        clear_cache()
        model = get_model("mock")
        rendered = render_prompt("mock", "scene_level")
        candidates = model.generate(
            Image.new("RGB", (64, 64), color="white"),
            rendered.text,
            n=1,
            prompt_id=rendered.prompt_id,
        )
        if len(candidates) != 1 or not isinstance(candidates[0], Candidate):
            raise RuntimeError("Mock backend did not return one Candidate")
    print("mock backend smoke check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
