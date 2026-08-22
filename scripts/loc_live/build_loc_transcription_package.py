#!/usr/bin/env python3
"""Build a GitHub-ready LOC scan plus transcription-draft package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path


RIGHTS_STATEMENT = (
    "The Library of Congress's digital scans of the Charles S. Hamlin Papers "
    "are in the public domain and are free to use and reuse."
)
CREDIT_LINE = "Library of Congress, Manuscript Division, Charles S. Hamlin Papers."
RIGHTS_URL = "https://www.loc.gov/item/mss246610001/#rights-and-access"
GUIDELINES_URL = "https://crowd.loc.gov/get-started/how-to-transcribe/"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    return rows


def local_path(value: object) -> Path:
    raw = str(value or "").strip()
    match = re.match(r"^/mnt/([A-Za-z])/(.*)$", raw)
    if os.name == "nt" and match:
        raw = f"{match.group(1).upper()}:/{match.group(2)}"
    return Path(raw).expanduser().resolve()


def safe_id(value: object, index: int) -> str:
    candidate = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_.")
    return candidate or f"page-{index:06d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.drafts.resolve())
    if args.expected_pages and len(rows) != args.expected_pages and not args.allow_incomplete:
        raise SystemExit(
            f"expected exactly {args.expected_pages} drafts, found {len(rows)}; "
            "wait for inference or use --allow-incomplete"
        )
    if not rows:
        raise SystemExit("no drafts found")

    invalid = [
        str(row.get("id") or row.get("page_id") or "unknown")
        for row in rows
        if row.get("transcription_standard") != "loc-btp-draft"
        or not row.get("model_prediction_is_not_ground_truth")
        or row.get("eligible_for_supervised_training") is not False
    ]
    if invalid:
        raise SystemExit(f"refusing {len(invalid)} rows without draft/non-ground-truth safeguards")

    output = args.output.resolve()
    images_dir = output / "images"
    transcripts_dir = output / "transcripts"
    output.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    transcripts_dir.mkdir(exist_ok=True)

    manifest_rows: list[dict] = []
    flag_counts: Counter[str] = Counter()
    item_sources: dict[str, dict] = {}
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        identifier = safe_id(row.get("id") or row.get("page_id"), index)
        if identifier in seen:
            raise SystemExit(f"duplicate page id: {identifier}")
        seen.add(identifier)
        source_image = local_path(row.get("image"))
        if not source_image.is_file():
            raise FileNotFoundError(f"missing image for {identifier}: {source_image}")
        extension = source_image.suffix.lower() or ".jpg"
        image_relative = Path("images") / f"{identifier}{extension}"
        transcript_relative = Path("transcripts") / f"{identifier}.txt"
        image_destination = output / image_relative
        transcript_destination = output / transcript_relative
        shutil.copy2(source_image, image_destination)
        write_text(transcript_destination, str(row.get("prediction") or ""))

        audit = row.get("loc_btp_audit") if isinstance(row.get("loc_btp_audit"), dict) else {}
        flags = [str(value) for value in audit.get("flags") or []]
        flag_counts.update(flags)
        item_id = str(row.get("item_id") or "")
        item_sources[item_id] = {
            "item_id": item_id,
            "title": str(row.get("title") or ""),
            "item_url": str(row.get("item_url") or ""),
        }
        manifest_rows.append(
            {
                "id": identifier,
                "item_id": item_id,
                "page_number": row.get("page_number"),
                "title": str(row.get("title") or ""),
                "loc_page_url": str(row.get("loc_url") or ""),
                "loc_item_url": str(row.get("item_url") or ""),
                "loc_iiif_image_url": str(row.get("image_url") or ""),
                "image": image_relative.as_posix(),
                "transcript": transcript_relative.as_posix(),
                "image_sha256": sha256(image_destination),
                "transcript_sha256": sha256(transcript_destination),
                "transcription_standard": "loc-btp-draft",
                "automatic_format_checks_passed": bool(audit.get("automatic_format_checks_passed")),
                "loc_review_flags": flags,
                "human_review_required": True,
                "official_transcript_available_at_source_audit": False,
                "model_prediction_is_not_ground_truth": True,
                "rights_statement": RIGHTS_STATEMENT,
                "credit_line": CREDIT_LINE,
            }
        )

    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output / "index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "id",
                "item_id",
                "page_number",
                "title",
                "loc_page_url",
                "image",
                "transcript",
                "automatic_format_checks_passed",
                "loc_review_flags",
                "human_review_required",
            ),
        )
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames if key != "loc_review_flags"},
                    "loc_review_flags": ";".join(row["loc_review_flags"]),
                }
            )

    source_lines = ["# Source items", "", f"Credit line: **{CREDIT_LINE}**", ""]
    for item in sorted(item_sources.values(), key=lambda value: value["item_id"]):
        source_lines.append(f"- [{item['item_id']}: {item['title']}]({item['item_url']})")
    write_text(output / "SOURCE_ITEMS.md", "\n".join(source_lines))

    write_text(
        output / "RIGHTS_AND_ATTRIBUTION.md",
        f"""# Rights and attribution

The Library of Congress states:

> {RIGHTS_STATEMENT}

Required/recommended credit line:

> {CREDIT_LINE}

See the [LOC Rights & Access statement]({RIGHTS_URL}) and each source item in
[`SOURCE_ITEMS.md`](SOURCE_ITEMS.md). The JPEG files in this package are
reduced-resolution copies retrieved from LOC's IIIF image service. No new
copyright is asserted over the machine-generated transcription drafts.

The CHURRO/Qwen model code and weights are not part of this data package and
remain governed by their respective licenses.
""",
    )

    write_text(
        output / "REVIEW_CHECKLIST.md",
        """# Human review checklist

Every transcript in this package is an unreviewed machine-generated draft.
Before representing one as a completed transcription:

- compare the entire scan and transcript line-by-line;
- preserve original spelling, punctuation, capitalization, abbreviations, and line breaks;
- correct omissions, repetitions, hallucinations, names, and reading order;
- resolve every `loc_review_flags` entry in `manifest.jsonl`;
- join words broken across physical lines on the line where the word begins;
- bracket readable deleted text and use `[?]` for illegible characters;
- format marginalia as `[*marginal text*]` in the proper reading position;
- include relevant printed text, page/catalog numbers, and letterheads;
- ignore images and backward bleed-through; and
- follow any campaign-specific LOC instructions and submit through LOC's review workflow.
""",
    )

    write_text(
        output / "README.md",
        f"""# Charles S. Hamlin Papers: LOC transcription drafts

This package pairs **{len(manifest_rows)} reduced-resolution public-domain LOC
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
images/<page-id>.jpg       reduced-resolution LOC scan
transcripts/<page-id>.txt  LOC-formatted OCR draft
manifest.jsonl             complete provenance, hashes, audit flags, rights
index.csv                  spreadsheet-friendly index
SOURCE_ITEMS.md            source item links and titles
RIGHTS_AND_ATTRIBUTION.md  public-domain statement and credit line
REVIEW_CHECKLIST.md        required visual-review steps
SHA256SUMS                 integrity hashes for the complete package
```

The drafts follow the text conventions in the [LOC By the People transcription
guide]({GUIDELINES_URL}) where those conventions can be applied automatically.
Image-dependent decisions remain explicitly assigned to human review.

## Attribution

> {CREDIT_LINE}

See [`RIGHTS_AND_ATTRIBUTION.md`](RIGHTS_AND_ATTRIBUTION.md).
""",
    )

    summary = {
        "schema_version": "loc_transcription_draft_package.v1",
        "pages": len(manifest_rows),
        "items": len(item_sources),
        "images_bytes": sum((output / row["image"]).stat().st_size for row in manifest_rows),
        "transcripts_bytes": sum((output / row["transcript"]).stat().st_size for row in manifest_rows),
        "automatic_format_checks_passed": sum(row["automatic_format_checks_passed"] for row in manifest_rows),
        "human_review_required": len(manifest_rows),
        "loc_review_flag_counts": dict(sorted(flag_counts.items())),
        "rights_statement": RIGHTS_STATEMENT,
        "credit_line": CREDIT_LINE,
        "source_drafts": "local live-draft manifest (machine path intentionally omitted)",
    }
    write_text(output / "package_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

    hashes: list[str] = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    write_text(output / "SHA256SUMS", "\n".join(hashes))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
