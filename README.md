---
base_model: stanford-oval/churro-3B
library_name: peft
pipeline_tag: image-text-to-text
license: qwen-research
tags:
  - base_model:adapter:stanford-oval/churro-3B
  - qlora
  - handwriting-recognition
  - historical-documents
  - library-of-congress
  - human-in-the-loop
---

# Grounded CHURRO for LOC handwriting

This repository packages the current first-occurrence-grounded LoRA adapter for
[`stanford-oval/churro-3B`](https://huggingface.co/stanford-oval/churro-3B), a
resumable Library of Congress inference pipeline, and **Transcript Desk**, a
local side-by-side scan/transcript review application.

The adapter is a research preview for human-assisted transcription. Generated
text is unreviewed OCR, not an official Library of Congress transcript or a
training label. Do not publish model output as archival ground truth without
human review.

> **Improved using Qwen.** This package is distributed for noncommercial
> research and evaluation under the Qwen Research License. Review
> [`LICENSE`](LICENSE), [`NOTICE`](NOTICE), and
> [`MODIFICATIONS.md`](MODIFICATIONS.md) before redistribution or deployment.

## What changed after Epoch 19

The adapter starts from the earlier Epoch 19 LOC LoRA and adds one full-page
continuation epoch designed to resist language-prior substitutions and
repetitive continuation:

- first-occurrence-weighted token loss (`alpha=2.0`);
- corrupted-prefix training on 35% of page examples;
- visual contrast on 25% of examples with weight `0.05`;
- 600 official page-transcript examples from the LOC Abraham Lincoln Papers;
- 593 pages trained and 7 over-length pages skipped;
- 3,686,400 trainable LoRA parameters (rank 8).

Inference defaults to the deterministic `grounded-faithful` profile. It keeps
normal CHURRO generation but applies a counterfactual image check when a
repetitive-tail candidate is detected, removing only text that is unsupported
by the visible page.

## Development-set results

Case- and punctuation-insensitive scoring on the protected aligned 100-page
development set produced:

| Decoder | CER ↓ | WER ↓ | Character edits | Word edits |
|---|---:|---:|---:|---:|
| Current adapter, faithful | 32.41% | 41.14% | 26,390 | 6,124 |
| **Current adapter, grounded-faithful** | **30.04%** | **38.66%** | **24,462** | **5,755** |

Against the earlier upstream CHURRO measurement (40.45% CER / 49.90% WER),
grounded-faithful reduced CER by **10.41 percentage points (25.73% relative)**
and WER by **11.24 points (22.53% relative)**. Against the Epoch 19 adapter
(33.19% / 41.73%), it reduced CER by **3.15 points (9.49% relative)** and WER
by **3.07 points (7.36% relative)**.

For context, earlier measurements on the same development-set family were
40.45% CER / 49.90% WER for upstream CHURRO and 33.19% CER / 41.73% WER for the
Epoch 19 adapter. Page-boundary alignment and reference quality remain under
audit, so these are development measurements rather than a final
generalization claim. Exact machine-readable summaries are in
[`evaluation/`](evaluation/).
The full provenance, training explanation, and derived comparison table are in
[`RESULTS_AND_PROVENANCE.md`](RESULTS_AND_PROVENANCE.md).

On a stricter 50-page subset, grounded-faithful scored 8.13% CER / 12.51% WER,
compared with 13.14% CER / 17.81% WER for faithful decoding. The strict subset
must not be treated as an independent held-out benchmark.

## Repository contents

```text
adapter/                    LoRA weights, tokenizer, processor, and metrics
scripts/transcribe.py       Standalone image/directory transcription
scripts/loc_live/           LOC audit, download, grounded inference, progress
tools/transcript_desk/      Read-only side-by-side review application
data/                       Fail-closed LOC inventory and audit summaries
loc_transcription_drafts/   Public-domain scans paired with review-required drafts
evaluation/                 Exact JSON summaries and limitations
training/                   Continuation run configuration and loss history
RESULTS_AND_PROVENANCE.md   Inventory citations and baseline improvements
LOC_TRANSCRIPTION_GUIDELINES.md  LOC draft rules and human-review boundary
DATA_CARD.md                Inventory and output provenance
CITATIONS.bib               CHURRO and Qwen citations
LICENSE                     Qwen Research License
NOTICE                      Attribution and LOC source notes
MODIFICATIONS.md            Changes relative to upstream CHURRO/Qwen
SHA256SUMS                  Integrity hashes for every released file
```

The CHURRO/Qwen base weights, downloaded scans, generated predictions, logs,
and progress state are intentionally excluded.

## Installation

The tested configuration uses Python 3.12, CUDA, 4-bit `bitsandbytes` loading,
and approximately 8 GB of VRAM.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The first run downloads the upstream `stanford-oval/churro-3B` base model from
Hugging Face. The base model is not included in this repository.

## Transcribe local images

One full page:

```bash
python scripts/transcribe.py page.jpg --output outputs/page
```

A directory of scans:

```bash
python scripts/transcribe.py scans --recursive --output outputs/scans
```

Full-page generation is autoregressive. On the development machine it varied
from roughly 30 to 65 seconds per page in larger runs; dense pages can take
longer, and model loading adds startup time.

## Run the verified untranscribed LOC inventory

The bundled inventory contains 14,637 page URLs from 157 Charles S. Hamlin
Papers items. At inventory time, live LOC item JSON advertised image-only
resources and none of the following: official full text, transcript fields,
word coordinates, or a text-service resource. Forty-seven items were rejected,
including 45 whose live API audit failed. This is a dated, fail-closed snapshot;
re-audit before making current claims about transcription status.

The audit ran from 2026-08-20 21:49:16 UTC through 23:08:18 UTC. Each accepted
item is cited by its canonical LOC `item_url`, and each page is cited by its
LOC resource-page `loc_url`, source item URL, and IIIF image URL. See
[`data/README.md`](data/README.md) for the provenance chain and audit policy.

Run or resume a bounded batch:

```bash
python scripts/loc_live/run_best_churro_on_confirmed_untranscribed_loc.py \
  --maximum-pages 100
```

From Windows PowerShell with WSL:

```powershell
.\START_CONFIRMED_LOC_WSL.ps1 -MaximumPages 100
```

Add `-Background` to leave it running with stdout/stderr logs in the output
directory. The launcher converts the repository path directly and avoids a
dependency on `wslpath`.

Run every inventoried page by omitting `--maximum-pages` or setting it to zero.
Downloads and prediction JSONL are resumable. Output is written beneath
`outputs/confirmed_untranscribed_loc/`.

To infer on scans that have already downloaded while a larger download is
still active:

```bash
python scripts/loc_live/build_downloaded_inference_manifest.py \
  --remote-manifest data/verified_remote_pages.jsonl \
  --images outputs/confirmed_untranscribed_loc/images \
  --output outputs/confirmed_untranscribed_loc/live_downloaded.jsonl \
  --inventory-source data/inventory_summary.json \
  --limit 300

python scripts/loc_live/evaluate_churro_fullpage_qlora.py \
  --manifest outputs/confirmed_untranscribed_loc/live_downloaded.jsonl \
  --output outputs/confirmed_untranscribed_loc/predictions_grounded \
  --adapter adapter \
  --max-pixels 1605632 \
  --max-new-tokens 1536 \
  --decode-profile grounded-faithful \
  --max-incomplete-retries 0
```

The evaluator appends one completed record at a time and resumes by page ID.

## Export LOC By the People–formatted drafts

Recognition and LOC formatting are deliberately separate. Keep the
`grounded-faithful` prediction, then run the conservative exporter:

```bash
python scripts/loc_live/export_loc_btp_drafts.py \
  --predictions outputs/confirmed_untranscribed_loc/predictions_grounded/predictions.jsonl \
  --output outputs/confirmed_untranscribed_loc/loc_btp_drafts
```

This preserves the model's wording, spelling, capitalization, punctuation, and
line breaks; removes only transport markup; normalizes explicit illegibility
placeholders to `[?]`; and writes one text draft per scan. It also records audit
flags for omissions and possible broken words.

During a live run, use `scripts/loc_live/watch_loc_btp_drafts.py` with the same
input/output paths to refresh the LOC draft JSONL whenever a new prediction is
appended. The watcher reports exact page progress to the bundled progress
monitor.

The result is a **LOC-formatted draft**, not an official or completed LOC
transcription. Completeness, reading order, deleted text, insertions,
marginalia, bleed-through, and line-broken words still require line-by-line
visual review. See [`LOC_TRANSCRIPTION_GUIDELINES.md`](LOC_TRANSCRIPTION_GUIDELINES.md).

## Review with Transcript Desk

From PowerShell:

```powershell
.\tools\transcript_desk\OPEN_VIEWER.ps1 -Port 8815
```

Or point it at any compatible prediction JSONL:

```powershell
.\tools\transcript_desk\OPEN_VIEWER.ps1 -Port 8815 `
  -Predictions .\outputs\confirmed_untranscribed_loc\predictions_grounded\predictions.jsonl
```

Transcript Desk displays the scan and OCR side by side, reloads completed
records every 15 seconds, supports search and generation-health filters, and
can copy or download individual drafts. It is read-only and never converts
predictions into reference labels. See
[`tools/transcript_desk/README.md`](tools/transcript_desk/README.md).

## Limitations

- CHURRO is autoregressive and can hallucinate, repeat, substitute plausible
  phrases, or end generation before all visible writing is covered.
- Grounding reduces a specific unsupported repetitive-tail failure; it does
  not prove that every word is image-supported.
- Historical handwriting, names, abbreviations, page damage, bleed-through,
  marginalia, and unusual layouts remain difficult.
- The 100-page measurements are model-development results, not an independent
  blind benchmark.
- The bundled LOC inventory records what the live API exposed during one audit.
  LOC may add or revise transcriptions later.

## Citation

If this package supports research, cite CHURRO and Qwen2.5-VL using
[`CITATIONS.bib`](CITATIONS.bib), and cite each LOC item using its `loc_url`
from the inventory or prediction record.

This is an independent project and is not an official release from Stanford
OVAL, Qwen/Alibaba Cloud, or the Library of Congress.
