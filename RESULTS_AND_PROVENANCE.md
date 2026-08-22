# Results, improvements, and data provenance

## Audited LOC page inventory

The bundled inventory is cited and traceable at every level:

- Collection: [Charles S. Hamlin Papers](https://www.loc.gov/collections/charles-s-hamlin-papers/)
- Audit summary: [`data/inventory_summary.json`](data/inventory_summary.json)
- Accepted item records: [`data/accepted_items.jsonl`](data/accepted_items.jsonl)
- Rejected item records and exact reasons: [`data/rejected_items.jsonl`](data/rejected_items.jsonl)
- Page-level records: [`data/verified_remote_pages.jsonl`](data/verified_remote_pages.jsonl)

The item audit ran from **2026-08-20 21:49:16 UTC through 23:08:18 UTC**.
It queried the live LOC collection and item JSON APIs and failed closed:

| Audit result | Count |
|---|---:|
| Collection items audited | 204 |
| Accepted image-only items with no advertised transcript/full-text signal | 157 |
| Rejected items | 47 |
| Candidate page images | 14,637 |

Of the 47 rejected items, 45 were rejected because the item API request failed;
two advertised non-image/full-text or transcript-related resources. Categories
overlap for those two records. Every accepted item row includes its canonical
`item_url`; every page row includes its LOC resource-page `loc_url`, source
`item_url`, and LOC IIIF `image_url`.

This audit establishes only what the live API advertised during that window.
It is not a permanent statement that LOC will never add a transcript. Re-audit
before making a current claim.

## What was changed

The current adapter starts from the earlier Epoch 19 LOC LoRA rather than
retraining the 3B base model. The continuation used:

- 600 official LOC full-page image/transcript examples from the Abraham
  Lincoln Papers, with 593 trained and 7 over-length pages skipped;
- 17 validation pages;
- one continuation epoch at learning rate `5e-7`;
- rank-8 LoRA with 3,686,400 trainable parameters, approximately **0.0981%**
  of 3,758,309,376 total parameters;
- first-occurrence-weighted token loss (`alpha=2.0`) to emphasize the visual
  evidence for a word's first appearance;
- corrupted-prefix training on 35% of page examples to reduce dependence on a
  perfect autoregressive prefix; and
- image/text visual contrast on 25% of examples with weight `0.05` to penalize
  text that fits language context but is unsupported by the page image.

At inference, `grounded-faithful` adds a reference-free counterfactual image
check only when it detects a suspicious repetitive tail. It does not alter the
base tokenizer or CHURRO/Qwen vocabulary. Exact settings and loss history are
in [`training/run_config.json`](training/run_config.json),
[`training/history.json`](training/history.json), and
[`training/summary.json`](training/summary.json).

## Comparison with upstream CHURRO and Epoch 19

The following are case- and punctuation-insensitive development measurements
on the same aligned 100-page evaluation-set family:

| Version | CER ↓ | WER ↓ |
|---|---:|---:|
| Upstream CHURRO baseline | 40.45% | 49.90% |
| Earlier Epoch 19 adapter | 33.19% | 41.73% |
| Current adapter, faithful decoder | 32.41% | 41.14% |
| **Current adapter, grounded-faithful** | **30.04%** | **38.66%** |

Compared with upstream CHURRO, the current grounded pipeline reduced:

- CER by **10.41 percentage points**, a **25.73% relative error reduction**;
- WER by **11.24 percentage points**, a **22.53% relative error reduction**.

Compared with the earlier Epoch 19 adapter, it reduced:

- CER by **3.15 percentage points** (**9.49% relative**);
- WER by **3.07 percentage points** (**7.36% relative**).

Holding the current adapter fixed, the grounded check improved over faithful
decoding alone by 2.37 CER points (7.31% relative) and 2.48 WER points (6.03%
relative). Exact current-run counts are in [`evaluation/`](evaluation/), and
the derived baseline comparison is in
[`evaluation/baseline_comparison.json`](evaluation/baseline_comparison.json).

On the stricter 50-page subset, grounded-faithful reduced CER from 13.14% to
8.13% (5.01 points; 38.10% relative) and WER from 17.81% to 12.51% (5.30
points; 29.73% relative).

## Interpretation limits

These numbers are development results, not a final blind benchmark. The
50-page subset is not independent of model development, and some 100-page
reference boundaries remain under audit. The baseline and Epoch 19 values are
legacy measurements reported to two decimal places, whereas the current JSON
summaries retain full precision. The comparison measures text edit error; it
does not prove complete visual coverage of every page.
