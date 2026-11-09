# WasteCap Dataset Card

## Summary

WasteCap is a curated image-caption-instruction dataset for waste
understanding. Each record links an image-level caption to object annotations,
optional masks, pixel bounding boxes, object captions, sorting actions, and
human review state. Approved records are the only default export scope.

This repository contains the curation code and a frozen split manifest. It
does not commit raw images, annotations, model weights, or the SQLite database.
The current local snapshot used during finalization contains 106 records:
73 train, 17 validation, and 16 test. These counts describe the local
curation database at finalization time, not a claim that the repository ships
106 images.

## Sources and licenses

| Source | Use | Recorded license/policy | Notes |
|---|---|---|---|
| TACO | Litter images and COCO-style annotations | CC BY 4.0 as recorded by the loader | Preserve source attribution and check the original image license metadata. |
| ZeroWaste-F | Deformable waste segmentation | Research use only (non-commercial) as recorded by the loader | Not included in the repository; obtain it from the official source and follow its terms. |
| COCO subset | Waste-relevant context images | CC BY 4.0 as recorded by the loader | The subset is derived from COCO categories; follow COCO attribution terms. |

Source URLs and download procedures are implemented in the dataset scripts and
should be checked again before redistribution. A record's `source` and
`license` fields must be retained in every export.

## Collection and preprocessing

Images are downloaded by bounded scripts, loaded into the canonical schema,
mapped to the WasteCap taxonomy, quality-filtered, deduplicated by image hash
when files are available, and assigned a source-stratified split. The frozen
manifest is [../data/processed/splits.json](../data/processed/splits.json).
No split is assigned by caption quality or model output.

## Annotation process

1. Import source objects, dimensions, source, and license.
2. Generate candidate captions through the selected backend.
3. Parse structured output while preserving unparsed text.
4. Human reviewers edit and approve image-level captions.
5. Reviewers verify object category, material, condition, location, bbox/mask,
   hallucination flag, and grounded sorting instruction.
6. Export only after schema, caption, and object-reference validation.

`approved` is a human action. Model confidence, CLIPScore, or a successful
parse never auto-approves a record.

## Schema and splits

The canonical Pydantic schema is in `src/wastecap/schema.py`. Splits are
source-image based, mutually exclusive, seeded, and stored as image IDs.
Run `python scripts/validate_final_splits.py` against a populated local DB to
check that the DB assignments match the committed manifest.

## Known limitations

- The current checked-in code does not ship the underlying image dataset.
- The final local snapshot is dominated by TACO; ZeroWaste download access was
  unavailable during the real-data run.
- Only a small fraction of local records were human-approved at finalization.
- Qwen grounding is a baseline prediction, not ground truth.
- CLIPScore is a filter signal, not a truth metric.
- Human hallucination flags are available only for reviewed records.
- Source licensing and image-host terms must be rechecked before
  redistribution.
- Model output quality depends on prompt, model version, quantization, and GPU.

## Intended use

Use this dataset for research and prototyping in waste captioning, grounded
sorting assistance, review tooling, and multimodal evaluation. It is not
intended as an autonomous waste-disposal authority, a legal recycling
classifier, or a safety-critical decision system. Human verification remains
necessary, especially for hazardous, ambiguous, occluded, or contaminated
items.
