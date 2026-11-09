# Waste Caption Dataset Curation: Project Guide

## What this project is
A 5-week project to build a **verified waste image–caption–instruction dataset** and a **Streamlit app to curate it**. The full brief is in `SPEC.md`. Read it before planning and whenever scope is unclear. This file is the working agreement.

Both parts are in scope and required:
1. **Curation app (Streamlit):** generate, review, edit, filter, compare and export captions.
2. **Dataset and evaluation pipeline:** aggregated images (TACO, ZeroWaste, COCO subset), structured captions, grounded sorting instructions, baseline evaluation, dataset card and docs.

Week map (see `SPEC.md` for detail):
- W1 setup, taxonomy, schema, data prep, splits
- W2 caption generation, linked to boxes/masks
- W3 verification, sorting instructions grounded to boxes/masks
- W4 JSON dataset, baseline evaluation (caption quality, hallucination rate, grounding accuracy)
- W5 error analysis, final splits, dataset card, annotation guidelines, report, demo

## Stack
- Python 3.11, PyTorch, Hugging Face `transformers` + `accelerate`; `bitsandbytes` for optional 4-bit
- Streamlit UI; SQLite as the source of truth for annotation state; JSON / JSONL / CSV for export
- `pycocotools`, `pycocoevalcap` (BLEU, CIDEr), CLIPScore via HF CLIP or `open_clip`
- `pydantic` for schemas, `pytest`, `ruff`
- Models: BLIP-2 (required), Qwen2.5-VL (required), LLaVA (optional third backend)
- Ask before adding any dependency outside this list.

## Hardware (fill in, then keep updated)
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM
- RAM / disk: Not fixed in repository; check local machine before downloads
- Model sizes to use by default: Qwen2.5-VL-3B 4-bit; mock backend for tests

Backends are selected by environment variables, never hard-coded:
`WASTECAP_BACKEND` (`mock` | `blip2` | `qwen_vl` | `llava`), `WASTECAP_DEVICE` (`cuda` | `cpu`), `WASTECAP_QUANTIZE` (`none` | `4bit` | `8bit`).

## Repo layout
```
SPEC.md  CLAUDE.md  PROMPTS.md
configs/        taxonomy.yaml, models.yaml, prompts/*.yaml
data/           (gitignored) raw/, processed/, db/   -- splits.json IS committed
src/wastecap/
  schema.py       pydantic models for the annotation record
  taxonomy.py     loads configs/taxonomy.yaml, label mapping from TACO/ZeroWaste/COCO
  store.py        SQLite access (only module that touches the DB)
  data/           taco.py, zerowaste.py, coco_subset.py, manifest.py, splits.py
  models/         base.py (CaptionModel), mock.py, blip2.py, qwen_vl.py, llava.py, registry.py
  prompts.py      template loading + structured-output parsing
  quality/        clipscore.py, filters.py, hallucination.py
  grounding/      boxes.py, masks.py, instructions.py, iou.py
  export.py  stats.py  cli.py
  eval/           caption_metrics.py, grounding_eval.py, baseline.py
app/            main.py, pages/ (Curate, Compare, Review queue, Instructions, Export, Stats)
scripts/        download + one-off helpers (all take --limit and --dry-run)
tests/          fixtures/ (tiny synthetic images + annotations)
docs/           dataset_card.md, annotation_guidelines.md, error_analysis.md
```
You may refine this in the plan, but explain why.

## Annotation record (draft; frozen after M0 unless I approve a change)
```json
{
  "image_id": "taco_000123",
  "file": "data/processed/images/taco_000123.jpg",
  "source": "taco|zerowaste|coco|web",
  "license": "...", "width": 0, "height": 0,
  "split": "train|val|test",
  "scene_context": "outdoor, cluttered, partially occluded",
  "caption": "final approved image-level caption",
  "objects": [{
    "object_id": "o1",
    "category": "bottle", "material": "plastic", "condition": "crushed",
    "count": 1, "location": "lower left, next to a can",
    "bbox": [x, y, w, h], "mask": "RLE or polygon, optional",
    "caption": "crushed plastic bottle on grass",
    "sorting_action": "recycle_plastic",
    "instruction": "Pick up the crushed plastic bottle at the lower left and place it in the plastic recycling bin."
  }],
  "candidates": [{"model": "blip2", "prompt_id": "...", "text": "...", "clip_score": 0.0, "created_at": "..."}],
  "status": "pending|approved|rejected|edited",
  "review": {"reviewer": "...", "notes": "...", "hallucination": false, "reviewed_at": "..."}
}
```
Draft taxonomy (finalize in `configs/taxonomy.yaml` in M0 and map every source label to it):
- material: plastic, metal, paper, cardboard, glass, organic, textile, mixed, unknown
- condition: intact, crushed, torn, dirty, deformed, partially_occluded
- sorting_action: recycle_plastic, recycle_metal, recycle_paper, recycle_glass, compost, landfill, hazardous

## Architecture rules
- **Model interface:** `CaptionModel.generate(image, prompt, n=1, **kw) -> list[Candidate]`. Every backend, including `MockModel`, implements it. UI and CLI only talk to the interface via `registry.py`.
- **Lazy loading:** never load a model or weights at import time. In Streamlit use `@st.cache_resource`.
- **Everything works with `WASTECAP_BACKEND=mock` and no GPU.** Tests never download weights or datasets.
- **Structured output:** prompts ask for JSON (object, material, condition, count, scene_context, caption). Parse robustly, fall back to free text, and flag unparsed outputs rather than dropping them.
- **Grounding:** every object caption and instruction keeps its `object_id` and its bbox/mask. Qwen2.5-VL grounding output is a baseline to evaluate, not ground truth.
- **Storage:** all DB access goes through `store.py`. Exports are derived from the DB, never hand-edited.

## Integrity rules (important)
- `status = approved` is set **only by a human action in the UI**. Never auto-approve. Test data uses fixtures.
- **Never invent numbers.** Evaluation results (CIDEr, BLEU, CLIPScore, hallucination rate, grounding accuracy) must come from scripts that were actually run. If something wasn't run, say so.
- A caption counts as **hallucinated** if it names an object, material, count or attribute not visible in the image.
- Metrics: grounding accuracy = IoU >= 0.5 against reference boxes; report CLIPScore as a filter signal, not as truth.
- Record source and license per image. Never commit datasets, weights or the DB. Commit `splits.json`.
- Split by source image so there is no leakage between train/val/test. Seed everything.

## Caption style (put the same rules in prompts and in docs/annotation_guidelines.md)
- Object-level: `<condition> <material> <object>`, e.g. "crushed plastic bottle", plus location if useful.
- Scene-level: one sentence on clutter, occlusion and surroundings.
- State only what is visible. Use "unknown" instead of guessing a material.

## Commands (keep this section accurate as the project grows)
```
pip install -e ".[dev]"
pip install -e ".[eval,ml]"  # official BLEU/CIDEr, CLIPScore, and model runtime
# pycocoevalcap's PTB tokenizer also requires Java (default-jre-headless)
ruff check . && ruff format --check .
pytest -q
streamlit run app/main.py
python -m wastecap.cli generate --model mock --split train --n 3 --limit 20
python -m wastecap.cli export --output-dir results/exports
python -m wastecap.cli stats
python scripts/run_baseline.py --model qwen_vl --split test --output-dir results
python scripts/error_analysis.py results/baseline_qwen_vl_test.json
python -m wastecap.eval.baseline --model qwen_vl --split val
```

## Workflow
- **Plan first** for anything touching more than ~3 files. Work **one milestone at a time** (milestones are in `PROMPTS.md`).
- Before calling a milestone done: ruff clean, pytest green, and the app boots (use `streamlit.testing.v1.AppTest` for a smoke test).
- Commit per milestone with a clear message. Update the Commands section and `docs/` as you go.
- End every milestone with a short report: what works, how to run it, what is left, and any decision I need to make.
- **Ask before:** schema or taxonomy changes after M0, downloads over 1 GB, deleting files, new dependencies.
- When you hit an ambiguity in `SPEC.md`, list the options and ask instead of picking silently.

## Code conventions
- Type hints everywhere, small functions, docstrings only where behavior is non-obvious.
- `logging`, not `print`. No bare `except`. Config from `configs/` and env vars, not magic constants.
- Streamlit pages stay thin: logic lives in `src/wastecap/` so it can be tested without the UI.
