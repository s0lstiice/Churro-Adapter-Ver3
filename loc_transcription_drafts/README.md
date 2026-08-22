# Charles S. Hamlin Papers: LOC transcription drafts

This package pairs **300 reduced-resolution public-domain LOC
scans** with CHURRO-generated transcription drafts. Each image and transcript
has stable provenance, a direct LOC page link, and SHA-256 hashes in
[`manifest.jsonl`](manifest.jsonl).

## Important status

These are **unreviewed OCR drafts**, not official Library of Congress
transcriptions and not archival ground truth. Every row is marked
`human_review_required: true`. Use [`REVIEW_CHECKLIST.md`](REVIEW_CHECKLIST.md)
before submitting or publishing corrected text.

## Layout

```text
pages/<page-id>.jpg        reduced-resolution LOC scan
pages/<page-id>.txt        matching LOC-formatted OCR draft
manifest.jsonl             complete provenance, hashes, audit flags, rights
index.csv                  spreadsheet-friendly index
SOURCE_ITEMS.md            source item links and titles
RIGHTS_AND_ATTRIBUTION.md  public-domain statement and credit line
REVIEW_CHECKLIST.md        required visual-review steps
SHA256SUMS                 integrity hashes for the complete package
```

The drafts follow the text conventions in the [LOC By the People transcription
guide](https://crowd.loc.gov/get-started/how-to-transcribe/) where those conventions can be applied automatically.
Image-dependent decisions remain explicitly assigned to human review.

## Attribution

> Library of Congress, Manuscript Division, Charles S. Hamlin Papers.

See [`RIGHTS_AND_ATTRIBUTION.md`](RIGHTS_AND_ATTRIBUTION.md).
