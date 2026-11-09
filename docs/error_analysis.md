# Baseline Error Analysis

This document is populated from the real M7 artifact
`results/baseline_qwen_vl_train.json`, generated with one approved train
record on the RTX 3050 using Qwen2.5-VL 3B in 4-bit CUDA mode.

## Measured result

| Metric | Value |
|---|---:|
| Records | 1 |
| BLEU-1 | ~0.0 |
| BLEU-2 | ~0.0 |
| BLEU-3 | ~0.0 |
| BLEU-4 | ~0.0 |
| CIDEr | 0.0 |
| CLIPScore | 0.5581530556082726 |
| Human hallucination rate | 0.0 |
| Grounding accuracy | 0.0 |

These are real one-record measurements, not estimates of the complete
dataset. The run used whitespace tokenization because Java was unavailable;
the JSON records this runtime detail.

## Failure categories

The one-record run had four overlapping failure signals: parse failure,
low CLIPScore, grounding below the IoU threshold, and the absence of a
human hallucination flag. These signals are reported separately by the
error-analysis script; they are not treated as independent examples.

## Failure groups

| Dimension | Value | Failed records |
|---|---|---:|
| Class | bottle | 1 |
| Condition | crushed | 1 |
| Occlusion | not_occluded | 1 |

## Observed failure

For `demo_taco_001`, the approved reference was:

> A crushed plastic bottle lying on a concrete surface.

Qwen returned malformed JSON containing a fragment about other waste objects
and returned no grounding boxes. The record was therefore a parse failure with
zero grounding matches. The human review flag was false; this must not be
interpreted as proof that the model output was correct because the human flag
was attached to the approved reference record, not a fresh model judgment.

## Interpretation and next runs

The failure suggests that the baseline prompt needs a stricter, shorter output
contract and that grounding tags should be evaluated separately from ordinary
caption generation. A complete multi-model error analysis remains blocked until
the validation split has enough approved records and a second model has a
successful real run. Run the full test split only after reviewing several
one-record outputs:

```bash
python scripts/run_baseline.py --model qwen_vl --split test --limit 1 --output-dir results
python scripts/run_baseline.py --model qwen_vl --split test --output-dir results
```

Regenerate this grouping from any result with:

```bash
python scripts/error_analysis.py results/baseline_qwen_vl_test.json \
  --output results/baseline_qwen_vl_test_error_analysis.md
```
