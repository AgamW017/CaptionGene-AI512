"""Download datasets/models and optionally run every configured model sequentially.

Examples:
    python scripts/setup_all.py --dry-run
    python scripts/setup_all.py --limit 100 --download-models --yes
    python scripts/setup_all.py --full --download-models --run-models --yes

The default is bounded to 100 images per source. Full datasets and model
weights require --yes because they can consume many gigabytes.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("setup_all")
ROOT = Path(__file__).resolve().parents[1]
MODELS = ("qwen_vl", "blip2", "llava")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    logger.info("$ %s", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _download_datasets(limit: int | None, dry_run: bool) -> None:
    commands = [
        ["download_taco.py", "--limit", str(limit or 0)],
        ["download_zerowaste.py", "--limit", str(limit or 0)],
        ["download_coco.py", "--split", "val", "--limit", str(limit or 0)],
    ]
    for script, *args in commands:
        command = [sys.executable, str(ROOT / "scripts" / script), *args]
        if dry_run:
            command.append("--dry-run")
        try:
            _run(command)
        except subprocess.CalledProcessError:
            if script == "download_zerowaste.py":
                logger.error(
                    "ZeroWaste download failed. Install gdown and retry, or download the "
                    "official archive manually as data/raw/zerowaste/zerowaste-f.zip."
                )
                raise
            raise


def _download_models(models: list[str], dry_run: bool) -> None:
    model_ids = {
        "qwen_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
        "blip2": "Salesforce/blip2-opt-2.7b",
        "llava": "llava-hf/llava-1.5-7b-hf",
    }
    if dry_run:
        for model in models:
            logger.info("[dry-run] Would download %s (%s)", model, model_ids[model])
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model downloads require huggingface_hub. Install ML dependencies with "
            "python -m pip install -e '.[ml]'."
        ) from exc

    for model in models:
        logger.info("Downloading model %s (%s) sequentially", model, model_ids[model])
        snapshot_download(model_ids[model])


def _ingest(limit: int | None, dry_run: bool) -> None:
    if dry_run:
        logger.info("[dry-run] Would ingest all downloaded sources")
        return
    command = [sys.executable, str(ROOT / "scripts" / "ingest.py")]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    _run(command)


def _run_models(models: list[str], limit: int | None, dry_run: bool) -> None:
    if dry_run:
        for model in models:
            logger.info("[dry-run] Would generate with %s", model)
        return

    for model in models:
        env = os.environ.copy()
        env.update(
            {
                "WASTECAP_BACKEND": model,
                "WASTECAP_DEVICE": env.get("WASTECAP_DEVICE", "cuda"),
                "WASTECAP_QUANTIZE": env.get("WASTECAP_QUANTIZE", "4bit"),
            }
        )
        command = [
            sys.executable,
            "-m",
            "wastecap.cli",
            "generate",
            "--model",
            model,
            "--split",
            "train",
            "--level",
            "scene",
            "--n",
            "1",
            "--verbose",
        ]
        if limit is not None:
            command.extend(["--limit", str(limit)])
        logger.info("Running %s sequentially; stop other model processes first", model)
        _run(command, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WasteCap data and models, then optionally run all backends.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=100, help="Images per source.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download and ingest all available images instead of the bounded limit.",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all three Hugging Face model repositories.",
    )
    parser.add_argument(
        "--run-models",
        action="store_true",
        help="Generate one scene caption per image with each model sequentially.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=list(MODELS),
        help="Subset of models for --download-models/--run-models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without changing data.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm potentially multi-gigabyte downloads and full runs.",
    )
    args = parser.parse_args()

    if (args.full or args.download_models) and not args.yes and not args.dry_run:
        parser.error("--full and --download-models require --yes")
    if args.limit < 1 and not args.full:
        parser.error("--limit must be positive unless --full is used")

    limit = None if args.full else args.limit
    logger.info("Preparing datasets with limit=%s", limit or "all")
    _download_datasets(limit, args.dry_run)
    _ingest(limit, args.dry_run)
    if args.download_models:
        _download_models(args.models, args.dry_run)
    if args.run_models:
        _run_models(args.models, limit, args.dry_run)
    logger.info("Setup complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
