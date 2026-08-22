# Evaluation notes

The JSON summaries in this directory are exact outputs from the current
adapter. Scoring normalizes case and punctuation.

| Set/profile | Pages | CER | WER | Character edits | Word edits |
|---|---:|---:|---:|---:|---:|
| aligned100 / faithful | 100 | 32.41% | 41.14% | 26,390 | 6,124 |
| aligned100 / grounded-faithful | 100 | 30.04% | 38.66% | 24,462 | 5,755 |
| strict50 / faithful | 50 | 13.14% | 17.81% | 5,061 | 1,241 |
| strict50 / grounded-faithful | 50 | 8.13% | 12.51% | 3,133 | 872 |

These are development measurements. The 50-page subset is not an independent
held-out test, and some 100-page reference boundaries remain under audit. The
results measure OCR text edits; they do not prove complete visual coverage of
every page.

For upstream CHURRO and Epoch 19 comparisons, absolute improvements, relative
error reductions, and precision caveats, see
[`baseline_comparison.json`](baseline_comparison.json) and
[`../RESULTS_AND_PROVENANCE.md`](../RESULTS_AND_PROVENANCE.md).
