# Autoregression and successor-model roadmap

## Current model

Epoch 19 is a LoRA adapter on CHURRO 3B/Qwen2.5-VL. The base is a causal,
autoregressive vision-language model: visual tokens and the prompt condition the
first output token, and every subsequent token also depends on the output prefix.

Advantages for historical transcription:

- variable-length output without a fixed character grid;
- language context for ambiguous handwriting;
- punctuation, capitalization, and spelling priors;
- one model can emit text, XML structure, and other prompted schemas; and
- the pretrained Qwen language decoder transfers efficiently through a small
  LoRA adapter.

Costs:

- sequential decoding is slower than parallel decoding;
- one early mistake can influence later tokens;
- the model can close XML before visiting every visual region;
- fluent hallucinations are possible; and
- exact visual coverage is not guaranteed by valid syntax.

## Safest near-term improvement

Keep Epoch 19 as the recognizer and add a transcript-independent coverage
controller:

1. propose all visible line polygons with the best image-only line segmenter;
2. run the ordinary full-page CHURRO transcription;
3. test whether the first, last, and intervening line regions have evidence in
   the output;
4. transcribe only suspicious overlapping page bands or line crops; and
5. merge monotonic text while retaining the original result when coverage passes.

This changes the inference pipeline without invalidating the adapter. It should
be enabled only after a source-disjoint A/B evaluation.

## Non-autoregressive experiment

Removing autoregression requires a new model rather than a code switch. A
practical design is:

- the existing polygon line proposer for full-page coverage;
- a visual encoder initialized from a document/handwriting model;
- a CTC line-recognition head that predicts character distributions in parallel;
- explicit blank tokens and monotonic alignment;
- a separate punctuation/capitalization restoration stage only if needed; and
- optional shallow language-model rescoring, kept separate from segmentation.

Train it on verified aligned line images and exact source text. CHURRO predictions
may help rank or align candidates, but should not silently become ground-truth
labels. Evaluate line CER/WER, full-page CER/WER, exact line coverage, first/last
line recall, and hallucination/omission rates.

Expected tradeoff: CTC should make coverage and alignment easier to audit and
decode faster, but it gives up part of CHURRO's powerful generative language and
structured-output prior. It may need substantially more clean line supervision
to match Epoch 19 on difficult handwriting.

## Faster autoregressive retraining

A higher learning rate is not a substitute for every late refinement epoch.
Recommended controlled experiment:

1. start a fresh LoRA from upstream CHURRO;
2. run one warm/high-rate epoch at `1e-5`;
3. continue at `5e-6` with cosine decay;
4. cap the run at 8 epochs with patience 2;
5. evaluate generated CER/WER and visual coverage after each epoch; and
6. compare against the locked Epoch 19 checkpoint, not just validation loss.

Stop or reduce the rate if validation CER/WER worsens, page output becomes
shorter, or omissions increase. The current loss history shows useful but
progressively smaller improvements through Epoch 19, so the shorter schedule is
an experiment rather than a guaranteed equivalent.

