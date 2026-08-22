# Data card

## Supervised continuation data

The current adapter continued from the earlier LOC Epoch 19 LoRA using 600
full-page examples from the Abraham Lincoln Papers. Labels were official LOC
per-page transcriptions; model predictions were not used as supervision. The
run processed 593 pages and skipped 7 examples that exceeded the configured
sequence limit. Seventeen additional pages were used for validation loss.

The package does not redistribute supervised training scans or transcripts.
`training/run_config.json`, `training/history.json`, and
`training/summary.json` preserve the run configuration and aggregate results.

## Untranscribed-page inventory

`data/verified_remote_pages.jsonl` is a dated discovery snapshot of 14,637 page
image URLs from 157 Charles S. Hamlin Papers items. The live item-level LOC JSON
audit required image resources and rejected any item advertising online text,
full text, transcript fields, word coordinates, text-service URLs, or textual
resource files. API failures were rejected rather than assumed untranscribed.

The inventory is evidence about what the live API exposed when audited, not a
permanent guarantee that LOC has no transcript. Re-run the audit or verify the
LOC record before publication.

## Generated outputs

Predictions are deliberately marked:

- `model_prediction_is_not_ground_truth: true`
- `eligible_for_supervised_training: false`
- `official_transcript_available: false`

Generated predictions and downloaded images are excluded from Git. Human
review is required before using any prediction as a transcript.
