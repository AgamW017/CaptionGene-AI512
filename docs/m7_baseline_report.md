# M7 Baseline Evaluation

## What is evaluated

`scripts/run_baseline.py` evaluates approved records only:

- BLEU-1 through BLEU-4 and CIDEr using the `pycocoevalcap` scorers.
- CLIPScore using OpenAI ViT-B/32 through `open_clip`.
- Hallucination rate from `(candidate.parse_failed OR
  AnnotationRecord.review.hallucination) / total`.
- Qwen grounding accuracy as the fraction of reference boxes matched at IoU >= 0.5.
- Per-record evidence used by `scripts/error_analysis.py`.

The evaluator never substitutes a made-up value for a failed model or metric run.
Missing images, missing approved records, model failures, and unavailable CLIP
scores raise an error.

The COCO scorers normally use the bundled PTB tokenizer. If Java is unavailable,
the same official BLEU/CIDEr scorer implementations use deterministic whitespace
tokenization instead; the JSON records this as `"caption_metric_tokenizer":
`"whitespace"` so the result is not misrepresented as PTB-tokenized.

## GPU setup

This project was run on an NVIDIA GeForce RTX 3050 Laptop GPU with 4 GiB VRAM.
Install the declared evaluation and ML extras in the project virtualenv:

```bash
source .venv/bin/activate
pip install -e '.[eval,ml]'
```

For the optional PTB tokenizer, install Java if system administration access is
available:

```bash
sudo apt-get install default-jre-headless
```

Run Qwen in 4-bit mode on CUDA:

```bash
export WASTECAP_DEVICE=cuda
export WASTECAP_QUANTIZE=4bit
python scripts/run_baseline.py \
  --model qwen_vl \
  --split test \
  --limit 1 \
  --output-dir results
```

Start with `--limit 1`, then increase it after checking VRAM and output quality.
The model runs one image at a time; generation time is hardware-, image-, and
output-length-dependent. The observed one-record run on the RTX 3050 completed
within the 600-second execution bound, loaded the Qwen checkpoint in about 18
seconds, and loaded CLIP successfully. Use `time` around the command to measure
the actual throughput of the local setup rather than extrapolating an invented
duration:

```bash
time python scripts/run_baseline.py --model qwen_vl --split test --output-dir results
```

## Observed real run

Command:

```bash
WASTECAP_DEVICE=cuda WASTECAP_QUANTIZE=4bit \
python scripts/run_baseline.py --model qwen_vl --split train --limit 1 --output-dir results
```

Artifacts:

- `results/baseline_qwen_vl_train.json`
- `results/baseline_qwen_vl_train_error_analysis.md`

The run used one approved record (`demo_taco_001`) and produced these measured
aggregate values:

| Metric | Value |
|---|---:|
| Records | 1 |
| BLEU-1 | 0.00000000000000005 |
| BLEU-2 | 0.00000000000000005 |
| BLEU-3 | 0.00000000000000006 |
| BLEU-4 | 0.00000000000000006 |
| CIDEr | 0.0 |
| CLIPScore | 0.5581530556082726 |
| Human hallucination rate | 0.0 |
| Qwen grounding accuracy | 0.0 |

The generated hypothesis was malformed JSON and contained no grounding boxes:

~~~text
```json
[
    - and other waste objects not visible in the image]
```
~~~

The approved reference was “A crushed plastic bottle lying on a concrete
surface.” This is a one-image diagnostic, not a statistically meaningful
dataset result. The low caption and grounding scores are an actual model
failure preserved in the JSON evidence.

## Error analysis

Generate or regenerate the grouped failure table from any result:

```bash
python scripts/error_analysis.py \
  results/baseline_qwen_vl_test.json \
  --output results/baseline_qwen_vl_test_error_analysis.md
```

Failures are grouped by reference class, condition, and whether the record is
partially occluded. A record is considered failed when it has a human
hallucination flag, a parse failure, or grounding accuracy below 1.0.
