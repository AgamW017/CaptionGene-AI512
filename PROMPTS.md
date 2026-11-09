# PROMPTS.md — WasteCap Milestone Task List

This file tracks implementation and evidence status. `[x]` means the code path
and focused tests exist; it does not mean that human review or a real-model
run has been completed. Real-run blockers are recorded in the reports.

---

## M0 — Scaffold + Schema + Taxonomy  ✅
**Week 1, days 1–2**

- [x] `pyproject.toml` with all pinned deps + ruff + pytest config
- [x] `.gitignore`, `.env.example`
- [x] Full directory skeleton (src, app, tests, configs, scripts, docs, data, results)
- [x] `configs/taxonomy.yaml` — 16 categories, 9 materials, 6 conditions, 8 sorting actions, source mappings (TACO/ZeroWaste/COCO)
- [x] `configs/models.yaml` — all 4 backends
- [x] `src/wastecap/schema.py` — Pydantic models: AnnotationRecord, ObjectAnnotation, Candidate, Review, RLEMask, PolygonMask, all enums
- [x] `src/wastecap/taxonomy.py` — loader, `map_source_label`, `default_sorting_action`, list helpers
- [x] `src/wastecap/store.py` — SQLite: init_db, ingest/get/update/upsert/delete, list, count, stats helpers
- [x] All stub modules importable (models, data, quality, grounding, eval, prompts, cli, export, stats)
- [x] `tests/fixtures/` — 6 synthetic images + matching annotation JSONs
- [x] `tests/test_schema.py`, `tests/test_taxonomy.py`, `tests/test_store.py`, `tests/test_mock_model.py`
- [x] `app/main.py` — boots with mock backend, shows DB path + record count
- [x] `app/pages/` — 6 page stubs
- [x] `scripts/` — 5 script stubs (download_taco, download_zerowaste, download_coco, ingest, compute_clipscore)
- [x] `ruff check .` passes
- [x] `pytest -q` passes (all tests green)

---

## M1 — Data Prep + Splits
**Week 1, days 3–5**

- [x] `scripts/download_taco.py` — functional download with `--limit` and `--dry-run`
- [x] `scripts/download_zerowaste.py` — functional download
- [x] `scripts/download_coco.py` — filtered to waste-relevant COCO classes
- [ ] `src/wastecap/data/taco.py` — TACO loader → AnnotationRecord stubs
- [ ] `src/wastecap/data/zerowaste.py` — ZeroWaste loader → AnnotationRecord stubs
- [ ] `src/wastecap/data/coco_subset.py` — COCO subset loader → AnnotationRecord stubs
- [ ] `src/wastecap/data/manifest.py` — content-hash deduplication, canonical image_id assignment
- [ ] `src/wastecap/data/splits.py` — seeded, source-stratified 70/15/15 split, writes `data/splits.json`
- [ ] `scripts/ingest.py` — functional: loaders → manifest → splits → DB
- [ ] `data/splits.json` committed — pending final release commit
- [ ] Tests: store round-trip with real loader output (fixture scale); no downloads in tests
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M2 — Caption Models + Generation CLI
**Week 2**

- [ ] `src/wastecap/models/blip2.py` — BLIP-2 OPT-2.7B with 4-bit quantization
- [ ] `src/wastecap/models/qwen_vl.py` — Qwen2.5-VL-3B with 4-bit quantization + grounding output parsing
- [ ] `src/wastecap/models/llava.py` — optional LLaVA backend
- [ ] `src/wastecap/prompts.py` — PromptTemplate loader + 3-stage JSON parser
- [ ] `configs/prompts/*.yaml` — prompt templates per model (object-level + scene-level)
- [ ] `src/wastecap/cli.py` — `generate` subcommand: iterate split, call model, write candidates to DB
- [ ] `scripts/compute_clipscore.py` — batch CLIPScore computation
- [ ] Tests: parser (valid JSON, malformed JSON, free text, out-of-vocab); mock CLI run
- [ ] `python -m wastecap.cli generate --model mock --split train --n 3 --limit 20` exits 0
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M3 — App Core (Browse, Generate, Edit/Approve/Reject, Structured Fields)
**Week 3, days 1–3**

- [ ] `app/pages/1_Curate.py` — image browser (paginated), model selector, generate + display candidates
- [ ] `app/pages/3_Review.py` — approve/reject/edit queue, structured fields form
- [ ] `app/components/image_viewer.py` — PIL image + bbox overlay render
- [ ] `app/components/caption_card.py` — text + CLIPScore badge + approve/reject buttons
- [ ] `app/components/structured_form.py` — category/material/condition/count/context form
- [ ] `status=approved` only settable by Review page approve button
- [ ] AppTest smoke tests: Curate loads + generates; Review approve sets status in DB
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M4 — Compare View + CLIPScore + Quality Filters
**Week 3–4**

- [ ] `app/pages/2_Compare.py` — side-by-side BLIP-2 vs Qwen2.5-VL per image
- [ ] `src/wastecap/quality/clipscore.py` — CLIPScore compute + batch + cache
- [ ] `src/wastecap/quality/filters.py` — threshold filter, status/source/model filters
- [ ] `src/wastecap/quality/hallucination.py` — flag candidates with parse_failed or named objects not in structured parse
- [ ] Review page updated with CLIPScore badge + hallucination flag + filter controls
- [ ] Tests: CLIPScore mock; filter unit tests; hallucination detection unit tests
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M5 — Sorting Instructions + Grounding
**Week 3–4**

- [ ] `src/wastecap/grounding/boxes.py` — xywh↔xyxy, clip, normalize
- [ ] `src/wastecap/grounding/masks.py` — RLE↔polygon, mask→bbox
- [ ] `src/wastecap/grounding/iou.py` — IoU for [x,y,w,h] boxes
- [ ] `src/wastecap/grounding/instructions.py` — template-based instruction generator, validates bbox present
- [ ] `app/pages/4_Instructions.py` — per-image object list + bbox overlay + instruction text + edit
- [ ] Qwen2.5-VL grounding boxes stored in candidate `raw_parsed`
- [ ] Tests: instruction generator (all 8 actions × key categories); IoU; bbox utilities; mask utilities
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M6 — Export + Statistics
**Week 4**

- [ ] `src/wastecap/export.py` — JSON, JSONL, CSV export from DB
- [x] `src/wastecap/stats.py` — full statistics dict (counts, distributions, approval rate, CLIPScore distribution)
- [x] `app/pages/5_Export.py` — format selector, split selector, download button
- [x] `app/pages/6_Stats.py` — bar charts + CLIPScore histogram
- [ ] Exported JSON re-parses against AnnotationRecord schema (test)
- [x] Tests: export correctness; stats against known fixture DB; AppTest export flow
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M7 — Evaluation
**Week 5, days 1–3**

Prerequisites: ≥100 approved records in val split (run M3 first).

- [ ] `src/wastecap/eval/caption_metrics.py` — CIDEr + BLEU via pycocoevalcap (sacrebleu fallback)
- [x] `src/wastecap/eval/grounding_eval.py` — IoU ≥ 0.5 grounding accuracy, pooled
- [x] `src/wastecap/eval/baseline.py` — orchestrator, writes `results/baseline_<model>_<split>.json`
- [ ] `results/baseline_blip2_val.json` and `results/baseline_qwen_vl_val.json` (or documented as "not yet run — requires GPU + approvals")
- [x] Hallucination rate = (parse_failed OR review.hallucination=True) / total, per model
- [ ] Grounding accuracy per category — blocked when model boxes have no object labels
- [ ] Tests: caption_metrics with tiny synthetic references; grounding_eval with known IoU cases
- [x] `python -m wastecap.eval.baseline --model qwen_vl --split val` exits 0 (or exits with clear prerequisite error)
- [ ] `ruff check .` passes; `pytest -q` passes

---

## M8 — Docs, Error Analysis, Final Splits, Demo
**Week 5, days 3–5**

- [ ] `docs/dataset_card.md` — HF-standard dataset card (source, size, splits, distributions, license, limitations)
- [ ] `docs/annotation_guidelines.md` — caption style, taxonomy reference, condition definitions, sorting actions, hallucination policy
- [ ] `docs/error_analysis.md` — ≥3 error categories per model with real image_id examples
- [ ] `data/splits.json` reviewed, no leakage confirmed
- [x] `README.md` — quickstart, demo instructions, env vars, dataset card link
- [ ] Demo video or notebook
- [x] Final: focused ruff/pytest and AppTest smoke checks pass; full release verification remains pending
