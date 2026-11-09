# M5 Real-Data and Real-Model Run Report

**Run date:** 2026-11-03  
**Repository:** `AgamW017/CaptionGene-AI512`  
**Requested scope:** TACO + ZeroWaste, up to 100 images, Qwen2.5-VL and BLIP-2  
**Actual scope completed:** 100 TACO images; one fresh Qwen2.5-VL inference result; no fresh BLIP-2 inference result; ZeroWaste download attempted but unavailable

## 1. Executive summary

The real-data pipeline was exercised on TACO data and the M5 grounded-instruction
path was inspected on real annotations.

Key outcomes:

- 100 TACO images were downloaded successfully from the official TACO image host
  after adding a browser-style `User-Agent` to the downloader request.
- TACO ingestion produced 102 TACO records in the database, including two
  pre-existing TACO records. The 100 downloaded images were materialized into
  `data/processed/images/` so the generation CLI could open them.
- ZeroWaste was attempted through its documented Google Drive download, but
  `gdown` could not retrieve the public file URL. No ZeroWaste data was used or
  substituted.
- CUDA was available on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GB VRAM.
- Qwen2.5-VL-3B-Instruct loaded with 4-bit quantization and generated one fresh
  real-image result. The result was preserved in the database and was a
  structured-output parse failure with mostly `unknown` fields.
- The requested 100-image Qwen run was stopped after the first persisted result
  because throughput was not practical on this machine: model download/loading
  took approximately 12 minutes and the first end-to-end inference was still
  running at approximately 19 minutes.
- BLIP-2 OPT-2.7B was attempted for one real image with 4-bit settings. The
  bounded 10-minute attempt spent its time downloading/loading model artifacts
  and produced no fresh candidate. Existing BLIP-2 candidates in the database
  were not counted as experiment output.
- Grounded instruction generation worked on the real TACO annotations. The
  report includes examples derived from real object boxes and sorting actions.

This run demonstrates that the pipeline is wired to real data and real model
backends, but it does **not** constitute a statistically meaningful model
evaluation. No CIDEr, BLEU, CLIPScore, hallucination-rate, or grounding-accuracy
claim is made from this run.

## 1.1 Follow-up three-image inference

After the initial bounded run was stopped, a corrective three-image run was
performed using the cached Qwen weights. This run used resized real TACO images,
one candidate per image, and `max_new_tokens=96`:

```text
taco_000096: 50.7 seconds, parse_failed=True
taco_000095:  6.1 seconds, parse_failed=True
taco_000094:  6.0 seconds, parse_failed=True
```

The model did execute inference on all three images, but none produced a usable
structured result. The first output contained a partially intelligible Chinese
description of a blue plastic bottle cap, surrounded by malformed JSON and
garbled tokens. The other two outputs contained malformed fenced JSON with
repetition and unknown/garbled content. These outputs are evidence of a real
model/runtime result, but not usable captions.

Raw outputs are preserved locally in
`results/m5_real_qwen_three_outputs.json`, with the execution log in
`results/m5_real_qwen_three_outputs.log`; both are ignored by Git.

## 1.2 Runtime fix and usable comparison run

The malformed output was traced to two issues in the first experiment:

1. The environment had installed Transformers 5.22.1 even though the project
   targets the Qwen implementation against the 4.x Transformers API.
2. The Qwen backend passed image inputs as a one-element list and decoded with
   the processor wrapper. The backend now uses the pinned Transformers 4.53.2
   runtime, passes the PIL image directly, resizes oversized inputs to a
   bounded 768-pixel edge, and decodes newly generated tokens through the
   processor tokenizer.

After the fix, a fresh three-image run used plain-English output rather than
fragile JSON, making it possible to compare the model description with the
source annotation:

| Image | Qwen2.5-VL output | Source annotation | Time |
|---|---|---|---:|
| `taco_000096` | “A discarded soda can lies on the mossy ground.” | metal can | 27.7 s |
| `taco_000095` | “A pink ball lies on the grass.” | metal lid | 1.7 s |
| `taco_000094` | “There is a blue plastic bottle cap on the grass.” | plastic lid | 2.0 s |

The first and third outputs identify the visible waste type reasonably enough
to support review, although the first does not explicitly say metal and the
second is a clear false object classification. This is a useful real result:
the model is now producing readable output, while the comparison exposes its
remaining error rate rather than hiding it behind parser failures.

The self-contained side-by-side table is available at
`results/m5_image_comparison.html`, and the annotated contact sheet is at
`results/m5_image_comparison_contact_sheet.png`. Red boxes in the visual
comparison are source-annotation boxes used for reference; they are not Qwen
predicted boxes.

## 2. Experiment plan

The run was planned in five stages:

1. Inspect hardware, installed dependencies, dataset availability, model
   configuration, and CLI entry points.
2. Download at most 100 images from each requested real source.
3. Ingest source annotations into SQLite and materialize image files at the
   paths expected by the generation CLI.
4. Run Qwen2.5-VL and BLIP-2 sequentially with one scene-level candidate per
   image and 4-bit CUDA settings.
5. Extract representative real outputs, grounded instruction examples, timing,
   failures, and next-step recommendations into this report.

The experiment intentionally used one candidate per image and scene-level
generation to bound runtime and GPU memory. Model artifacts, raw data, the
SQLite database, and run logs remain ignored by Git.

## 3. Environment

### Hardware

| Resource | Observed value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| GPU memory | 4.0 GiB |
| Driver | 616.92 |
| CUDA reported by Torch | 13.2 |
| System RAM | 7.6 GiB |
| Free disk before run | approximately 926 GiB |

### Software

| Component | Observed value |
|---|---|
| Python | 3.14.4 |
| Torch | 2.15.0+cu132 |
| Transformers | 5.22.1 |
| Accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| CUDA available | `True` |

The declared `.[ml]` extra could not be installed unchanged because
`torchvision==0.22.0` has no Python 3.14 build. A compatible environment was
installed locally without changing the dependency manifest:

```text
torch==2.15.0
torchvision==0.29.2
transformers>=4.53.2
accelerate>=1.8.1
bitsandbytes>=0.47.0
```

`gdown==6.5.1` was installed only in the local virtual environment to attempt
the documented ZeroWaste download. It was not added to `pyproject.toml`.

## 4. Dataset acquisition and ingestion

### TACO

Command:

```bash
.venv/bin/python scripts/download_taco.py --limit 100 --delay 0.1
.venv/bin/python scripts/ingest.py --source taco --limit 100 --seed 42
```

Result:

```text
TACO downloads: ok=100 failed=0 skipped=0
TACO records in DB: 102
TACO splits: train=72, val=15, test=15
Objects: 305
Objects with bbox: 305
Objects with mask: 302
Objects with sorting_action: 305
```

The database contained two existing TACO records in addition to the 100 newly
downloaded records. The ingestion loaders intentionally write expected
processed paths, so the 100 downloaded files were copied from
`data/raw/taco/data/` to `data/processed/images/` using the canonical temporary
IDs expected by the loader. This was required before invoking the generation
CLI; otherwise the CLI correctly skipped records whose processed image path did
not exist.

The taxonomy loader emitted warnings for several source labels, including
`Other carton`, `Food Can`, and `Glass cup`. Those labels were mapped to
`unknown` where no canonical mapping existed. This is a data-quality finding,
not a model result, and should be addressed before large-scale annotation.

### ZeroWaste

Command:

```bash
.venv/bin/pip install gdown
.venv/bin/python scripts/download_zerowaste.py --limit 100
```

Result:

```text
gdown: Cannot retrieve the public link of the file.
No ZeroWaste archive downloaded.
No ZeroWaste records ingested.
```

The official Google Drive file was inaccessible through the documented public
link. No alternate mirror or unauthorized bypass was used. A future run needs a
working official download URL, a manually supplied licensed archive, or a
maintainer-approved downloader update.

## 5. Model runs

### Qwen2.5-VL-3B-Instruct

Configuration:

```bash
WASTECAP_BACKEND=qwen_vl
WASTECAP_DEVICE=cuda
WASTECAP_QUANTIZE=4bit
WASTECAP_DB_PATH=data/db/wastecap.db
```

Command:

```bash
.venv/bin/python -m wastecap.cli generate \
  --model qwen_vl --split all --n 1 --limit 100 --verbose
```

Observed behavior:

- The model loaded successfully on the 4 GB GPU with 4-bit quantization.
- Model weight retrieval took approximately 12 minutes.
- GPU memory reached approximately 3.88 GiB of 4.0 GiB.
- One fresh TACO result was persisted before the run was stopped.
- The first fresh inference took approximately 19 minutes including load
  overhead; 100 images was therefore not a practical bounded run on this
  hardware.

Representative fresh output from `taco_000099`:

```text
parse_failed = True

~~~json
[
  {
    "object_id": "o1",
    "category": "unknown",
    "material": "unknown",
    "condition": "unknown",
    "count": 1,
    "location": "unknown",
    "caption": "unknown unknown unknown"
  }
]
~~~
```

The output was preserved rather than discarded, as required by the project
rules. The parser correctly flagged it as unparseable because the model returned
a JSON array/fenced response that did not match the expected structured-output
shape. The content also demonstrates a quality problem: the response did not
recover useful object attributes despite the image having source annotations.

The run log is stored locally at `results/m5_qwen_vl_100.log` and is ignored by
Git.

### BLIP-2 OPT-2.7B

Configuration:

```bash
WASTECAP_BACKEND=blip2
WASTECAP_DEVICE=cuda
WASTECAP_QUANTIZE=4bit
```

Command:

```bash
timeout 600 .venv/bin/python -m wastecap.cli generate \
  --model blip2 --split all --n 1 --limit 1 --verbose
```

Result:

- The model download/loading attempt did not reach a fresh inference result
  within 600 seconds.
- No fresh BLIP-2 candidate was persisted.
- Existing BLIP-2 candidates in the database were excluded from this run’s
  results.

The run log is stored locally at `results/m5_blip2_1.log` and is ignored by
Git. The main constraint was the combination of model artifact download,
limited GPU memory, and Python 3.14-compatible package versions.

## 6. Grounded instruction results on real annotations

Instruction generation was run against real TACO object annotations using:

```python
generate_instruction(object_annotation, record.width, record.height)
```

The generator used each annotation’s category, material, condition when
available, sorting action, and bbox-derived spatial location. It did not infer
unsupported object properties from the image model output.

Examples:

| Image | Source object fields | Generated instruction |
|---|---|---|
| `taco_000096` | `can`, `metal`, `recycle_metal`, bbox `[649, 55, 93, 88]` | “Pick up the metal can at the upper and place it in the metal recycling bin.” |
| `taco_000095` | `lid`, `metal`, `recycle_metal`, bbox `[654, 1783, 126, 114]` | “Pick up the metal lid at the left and place it in the metal recycling bin.” |
| `taco_000099` | `unknown`, `unknown`, `landfill`, bbox `[0, 0, 1528, 1687]` | “Pick up the waste item at the upper left and place it in the general waste (landfill) bin.” |

All 305 inspected TACO objects had a bbox and sorting action. 302 also had a
polygon mask. These are source-annotation facts and should not be confused
with model grounding accuracy. The real instructions were generated as
examples; they were not automatically approved or written as human-reviewed
ground truth.

The extracted examples are also available locally at
`results/m5_instruction_samples.json` and are ignored by Git.

## 7. Findings

### Revised per-object grounding probe

After the original scene-level grounding prompt returned zero boxes for all
100 images, the runner was changed to issue one explicit grounding query per
reference object. A fresh five-record CUDA run using Qwen2.5-VL-3B in 4-bit
mode completed on the RTX 3050 Laptop GPU:

| Metric | Measured value |
|---|---:|
| Records processed | 5 |
| Object queries | 20 |
| Records with predicted boxes | 1 |
| Total predicted boxes | 3 |
| Records with a parse failure | 5 |
| Elapsed time | 296.59 seconds |

The artifact is
`results/qwen_taco_grounding_object_v2_5.json`. The result is evidence that
the dedicated object prompt can elicit grounding tags, but it is not a
successful benchmark: most responses still failed the structured caption
contract, and the sample is too small to estimate model quality.

### What worked

1. The downloader and ingestion path can acquire and normalize real TACO data.
2. CUDA and 4-bit Qwen loading work on the available GPU.
3. Model output is persisted even when structured parsing fails.
4. Grounded instruction generation uses real bboxes and sorting actions.
5. The app/database schema preserves source annotations, masks, and candidate
   outputs without auto-approving instructions.

### Problems found

1. **ZeroWaste availability:** the official Google Drive link was not retrievable
   through `gdown`.
2. **TACO access headers:** the original downloader received HTTP 403 responses;
   adding a descriptive browser-style `User-Agent` fixed access.
3. **Processed-image materialization:** ingestion recorded processed paths but
   did not copy downloaded raw images into those paths. The run required an
   explicit bounded materialization step.
4. **Qwen runtime compatibility:** the initial Transformers 5.22.1 environment
   produced malformed/garbled output. Pinning Transformers 4.53.2 and fixing
   image passing/decoding produced readable English captions.
5. **Structured JSON reliability:** even after the runtime fix, the constrained
   JSON prompt remained unreliable on the three-image test. Plain-English
   generation is currently the more useful review path.
6. **Model quality:** one of three plain-English outputs was clearly wrong
   (“pink ball” for a metal lid); two identified waste-like content.
7. **Throughput:** 100-image inference is not practical on a 4 GB laptop GPU
   with the selected 3B vision-language model.
8. **BLIP-2 feasibility:** the one-image attempt did not reach inference within
   the ten-minute bound.
9. **Taxonomy coverage:** several TACO labels map to unknown, reducing the
   usefulness of automatic sorting actions and instructions.
10. **Evaluation availability:** `src/wastecap/eval/baseline.py` is still a
   documented M7 stub, so no benchmark metrics were claimed.

## 8. Recommended next plan

### Immediate engineering fixes

1. Make ingestion materialize or symlink downloaded images as an explicit,
   tested pipeline step rather than relying on a manual copy.
2. Improve structured-output parsing to accept a top-level JSON array for
   object-level Qwen responses while retaining the original raw text and
   parse-failure flag when fields are invalid.
3. Expand TACO taxonomy mappings for `Other carton`, `Food Can`, `Glass cup`,
   and other frequent labels; review resulting sorting actions manually.
4. Add a configurable `max_new_tokens` and image-resolution cap for practical
   local inference.

### Practical model experiment

For this laptop, use a small 3–5 image smoke run per model, one candidate per
image, with a hard wall-clock limit and explicit cache reuse. Do not describe
that run as a dataset evaluation. For a meaningful 100-image experiment, use
Colab/Kaggle or a machine with substantially more VRAM.

### Dataset completion

Obtain the ZeroWaste archive through an official accessible source and record
its license/access date before ingestion. Then repeat the same manifest,
source-level split, generation, and review process.

### Evaluation milestone

Implement the M7 baseline runner before making quality claims. At minimum,
report:

- parse success rate;
- taxonomy-field accuracy against source annotations where comparable;
- human-reviewed hallucination rate;
- bbox grounding accuracy at IoU >= 0.5;
- CLIPScore only as a filter signal;
- per-source and per-model runtime/failure rates.

## 9. Reproducibility and artifact policy

All commands were run from the repository root using `.venv/bin/python`.
Dataset files, model caches, SQLite state, and result logs are ignored by Git.
The only source change made specifically during acquisition was the
browser-style request header in
[`scripts/download_taco.py`](../scripts/download_taco.py).

No dataset, model weights, database, or generated result log is included in
the repository. This report records the observed run rather than inventing
metrics for work that was not completed.
