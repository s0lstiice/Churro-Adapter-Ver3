# Modification notice

This repository is not an unmodified copy of CHURRO or Qwen2.5-VL.

- `adapter/adapter_model.safetensors` contains independently trained rank-8
  LoRA parameters for `stanford-oval/churro-3B`.
- Training continued the earlier LOC Epoch 19 adapter for one epoch on 600
  full-page examples with official LOC page transcripts.
- The continuation objective adds first-occurrence token weighting, corrupted
  prefixes, and image/text visual contrast. Exact parameters are preserved in
  `training/run_config.json`.
- `grounded-faithful` decoding adds a reference-free counterfactual image check
  for unsupported repetitive tails. The base weights and tokenizer vocabulary
  are unchanged.
- `scripts/loc_live/` contains independent LOC audit, download, evaluation, and
  progress utilities.
- `tools/transcript_desk/` is an independent read-only review application.
- Upstream CHURRO/Qwen base-model weights are not redistributed.

See `README.md`, `training/`, and `evaluation/` for configuration, metrics, and
limitations.
