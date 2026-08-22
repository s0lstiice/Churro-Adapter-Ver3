# Library of Congress transcription-draft mode

This package can export CHURRO output as a **LOC By the People–formatted
draft**. It cannot certify an OCR prediction as an accurate or complete LOC
transcription. A human reviewer must still compare every line with the scan.

The implementation was checked against the Library of Congress guidance on
August 21, 2026:

- [Transcription: Basic Rules](https://crowd.loc.gov/get-started/how-to-transcribe/)
- [Transcription: Things to Avoid](https://crowd.loc.gov/get-started/transcription-things-to-avoid/)
- [Transcription: Printed Text & Images](https://crowd.loc.gov/get-started/transcription-printed-text-images/)
- [Transcription: Unusual Text](https://crowd.loc.gov/get-started/transcription-unusual-text/)
- [Transcription Quick Tips (PDF)](https://crowd-media.loc.gov/cm-uploads/resources/BTP_TranscriptionQuickTips.pdf)
- [How to Review](https://crowd.loc.gov/get-started/how-to-review/)

## Rules represented by this export format

- Preserve the source's spelling, grammar, punctuation, abbreviations, word
  order, and physical line breaks. Do not silently correct or paraphrase.
- Put a word broken across physical lines entirely on the line where it starts.
- Put readable deleted text in square brackets. Use `[?]` for illegible text or
  letters, including partial forms such as `s[????]`.
- Put insertions where they would be read aloud.
- Put marginalia in square brackets and asterisks, such as
  `[*Refers to Gettysburg*]`, after the relevant passage or at the end when it is
  unrelated.
- Include typed and printed text, page/catalog numbers, and letterheads when
  relevant. Read newspaper columns in natural order without imitating layout.
- Do not add editorial notes, translations, expanded abbreviations, or visual
  styling markup.
- Ignore images and backward bleed-through. Use LOC's “Nothing to transcribe”
  workflow for genuinely text-free/image-only pages.
- Preserve original non-English characters and common symbols. Represent
  shorthand as `[[shorthand]]` instead of translating it.

## What the exporter changes automatically

The exporter is intentionally lossless with respect to recognized wording. It:

1. removes CHURRO XML/code-fence transport wrappers;
2. preserves prediction wording, capitalization, punctuation, Unicode, and
   line order;
3. normalizes explicit `[illegible]`, `[unclear]`, and similar placeholders to
   LOC's `[?]` convention;
4. emits one UTF-8 text draft per page plus a compatible JSONL manifest;
5. flags obvious omission language, leftover markup, and possible line-broken
   words; and
6. marks every result `human_review_required: true` and prevents it from being
   mistaken for a supervised-training label.

It does **not** invent brackets for deletions, classify marginalia, join a
line-broken word, remove bleed-through, or declare a page complete. Those
decisions require the scan. `automatic_format_checks_passed` only means no
detectable text-format problem was found; it is not an accuracy score or LOC
approval.

## Export

From the repository root:

```bash
python scripts/loc_live/export_loc_btp_drafts.py \
  --predictions outputs/confirmed_untranscribed_loc/predictions_grounded/predictions.jsonl \
  --output outputs/confirmed_untranscribed_loc/loc_btp_drafts
```

Outputs:

```text
loc_btp_drafts/
  loc_drafts.jsonl        compatible with Transcript Desk
  summary.json            counts and audit flags
  drafts/<page-id>.txt    copy-ready LOC-format draft for each scan
```

Review the exported JSONL in Transcript Desk:

```powershell
.\tools\transcript_desk\OPEN_VIEWER.ps1 -Port 8816 `
  -Predictions .\outputs\confirmed_untranscribed_loc\loc_btp_drafts\loc_drafts.jsonl
```

For a live inference run, keep the draft JSONL synchronized as new pages
finish:

```bash
python scripts/loc_live/watch_loc_btp_drafts.py \
  --predictions outputs/confirmed_untranscribed_loc/predictions_grounded/predictions.jsonl \
  --output outputs/confirmed_untranscribed_loc/loc_btp_drafts \
  --total-pages 300
```

Before contributing text to LOC, compare the entire page line-by-line, resolve
all audit flags, and follow any campaign-specific instructions shown by LOC.
