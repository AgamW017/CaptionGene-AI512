# WasteCap

WasteCap is a Streamlit curation app and evaluation pipeline for verified
waste-image captions and grounded sorting instructions. It stores annotation
state in SQLite, keeps model backends behind one interface, and exports
validated JSON, JSONL, and CSV datasets.

## Quick start from a fresh clone

```bash
git clone <repository-url> CaptionGene-AI512
cd CaptionGene-AI512
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/smoke_mock.py
python scripts/demo_mock.py
pytest -q
ruff check . && ruff format --check .
```

The smoke check uses no GPU, downloads, dataset, model weights, or database
files. It proves that the package imports, the mock backend generates a
`Candidate`, prompts load, and SQLite initialization works in a temporary
directory.

`scripts/demo_mock.py` runs the same deterministic backend against a committed
fixture image and prints the generated and parsed caption without modifying
the database.

## Run the app

```bash
export WASTECAP_BACKEND=mock
export WASTECAP_DEVICE=cpu
streamlit run app/main.py
```

Use the Curate page to upload/select images, override a scene prompt, and
generate candidates. The Review page approves captions,
the Instructions page to edit and approve grounded sorting instructions, and
the Export and Stats pages for validated outputs and progress.

### Run with Docker

The default image is CPU-compatible and uses the mock backend, so it runs
without a GPU or model downloads:

```bash
docker compose up --build
```

Open [http://localhost:8501](http://localhost:8501). SQLite state, uploaded
data, generated exports, and model caches are stored in named Docker volumes
and survive container restarts.

To select a backend without rebuilding:

```bash
WASTECAP_BACKEND=qwen_vl \
WASTECAP_DEVICE=cuda \
WASTECAP_QUANTIZE=4bit \
docker compose up
```

For model dependencies, rebuild with `INSTALL_ML=true`:

```bash
INSTALL_ML=true docker compose build
INSTALL_ML=true docker compose up
```

The ML image installs the Python model stack but does not download weights
during build. The first generation downloads the selected Hugging Face model.
For NVIDIA GPU inference, install the NVIDIA Container Toolkit on the host and
run the container with GPU access:

```bash
docker run --gpus all --rm -p 8501:8501 \
  -e WASTECAP_BACKEND=qwen_vl \
  -e WASTECAP_DEVICE=cuda \
  -e WASTECAP_QUANTIZE=4bit \
  wastecap:latest
```

Real model runs require the optional ML dependencies and weights:

```bash
python -m pip install -e '.[ml]'
export WASTECAP_BACKEND=qwen_vl
export WASTECAP_DEVICE=cuda
export WASTECAP_QUANTIZE=4bit
streamlit run app/main.py
```

The CLI and Curate generation path make one deterministic repair attempt when
a backend returns malformed structured output. Successful repairs are stored
as normalized JSON; irreparable raw text remains flagged for review and is
never treated as valid structured data. The original model output is logged
when both attempts fail.

The mock backend is the supported default for development and tests. Dataset
downloads are separate, bounded scripts and are never performed by tests.

## Dataset preparation and frozen splits

The documented source loaders support TACO, ZeroWaste-F, and a COCO
waste-class subset. Download only the source and limit needed for your run,
then ingest:

```bash
python scripts/download_taco.py --limit 100
python scripts/ingest.py --source taco --limit 100
python scripts/validate_final_splits.py
```

The committed split manifest is
[data/processed/splits.json](/home/wsl/CaptionGene-AI512/data/processed/splits.json).
It contains 73 train, 17 validation, and 16 test IDs from the local 106-record
curation snapshot. Raw images, annotations, SQLite state, weights, and result
JSON files remain gitignored. If the dataset snapshot changes, regenerate and
review the split manifest rather than silently reusing it.

## Export and evaluation

```bash
python -m wastecap.cli export --output-dir results/exports
python -m wastecap.cli stats
python -m pip install -e '.[eval,ml]'
python scripts/run_baseline.py --model qwen_vl --split test --limit 1 --output-dir results
python scripts/error_analysis.py results/baseline_qwen_vl_test.json
```

The baseline writes measured BLEU, CIDEr, CLIPScore, human-review/parse-failure
hallucination rate, pooled grounding accuracy, category reference denominators,
and per-record evidence. It does not
invent values when a dependency, image, model, or approved reference is
missing. See [docs/m7_baseline_report.md](docs/m7_baseline_report.md) for GPU
requirements and runtime notes.

The current local snapshot has 106 records but only one approved record and no
approved validation records. A complete validation benchmark therefore cannot
be claimed until a human reviewer approves the validation set.

## Short demo video

1. Start the app with `WASTECAP_BACKEND=mock` and `WASTECAP_DEVICE=cpu`.
2. Show the dashboard and the current record/status counts.
3. Open Curate, select an image, generate one or more mock candidates, and
   show that the candidate remains unapproved.
4. Open Review, edit a caption, add reviewer information, and explicitly
   approve it.
5. Open Instructions, select an object, show the highlighted bbox, edit the
   sorting instruction, and approve it.
6. Open Stats to show review progress and class counts.
7. Open Export, validate and export the split files, then show the generated
   filenames.
8. End by showing the terminal commands `pytest -q` and
   `python scripts/smoke_mock.py`.

Keep the demo on fixture or locally downloaded data. Do not show credentials,
private files, model caches, or unverified metrics.

## Documentation

- [Dataset card](docs/dataset_card.md)
- [Annotation guidelines](docs/annotation_guidelines.md)
- [Error analysis](docs/error_analysis.md)
- [M7 baseline report](docs/m7_baseline_report.md)
- [M5 real-data report](docs/m5_real_data_report.md)
