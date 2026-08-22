#!/usr/bin/env python3
"""Export CHURRO prediction JSONL as LOC By the People guideline drafts.

The exporter is intentionally conservative. It preserves recognized wording,
spelling, capitalization, punctuation, and line order. It only removes model
transport markup and canonicalizes explicit uncertainty placeholders to the
LOC ``[?]`` convention. Visual rules that cannot be proven from OCR text alone
remain explicit human-review checks.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path


GUIDELINE_URLS = {
    "basic_rules": "https://crowd.loc.gov/get-started/how-to-transcribe/",
    "things_to_avoid": "https://crowd.loc.gov/get-started/transcription-things-to-avoid/",
    "printed_text_and_images": "https://crowd.loc.gov/get-started/transcription-printed-text-images/",
    "unusual_text": "https://crowd.loc.gov/get-started/transcription-unusual-text/",
    "quick_tips": "https://crowd-media.loc.gov/cm-uploads/resources/BTP_TranscriptionQuickTips.pdf",
}

VISUAL_REVIEW_CHECKS = (
    "complete_text_in_natural_reading_order",
    "original_spelling_grammar_punctuation_and_abbreviations_preserved",
    "physical_line_breaks_preserved_and_line_broken_words_joined_on_first_line",
    "deleted_text_bracketed_and_insertions_placed_in_reading_order",
    "marginalia_wrapped_in_[**]_and_positioned_by_meaning",
    "printed_text_letterheads_page_numbers_and_catalog_marks_included",
    "images_bleed_through_and_nontext_features_not_described",
    "non_english_characters_preserved_without_translation",
    "shorthand_represented_as_[[shorthand]]",
)

UNCERTAINTY_PLACEHOLDER = re.compile(
    r"\[(?:unclear|illegible|unreadable|indecipherable)\]"
    r"|\((?:illegible|unreadable|indecipherable)\)",
    re.IGNORECASE,
)
OMISSION_LANGUAGE = re.compile(
    r"omitted\s+for\s+brevity|"
    r"(?:rest|remainder).{0,80}(?:omitted|not\s+transcribed)|"
    r"(?:content|text|page).{0,80}(?:omitted|not\s+transcribed)",
    re.IGNORECASE | re.DOTALL,
)
POSSIBLE_BROKEN_WORD = re.compile(r"(?m)[^\W\d_]+-\n[^\W\d_]", re.UNICODE)
XML_TAG = re.compile(r"<\/?[A-Za-z][^>]*>")
CHURRO_TRANSPORT_XML = re.compile(
    r"<(?:HistoricalDocument|Page|Body|Paragraph|Line|Header|Footer|TextBlock)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    return rows


def plain_text_from_raw(raw: str) -> str:
    """Recover visible text if an older prediction row lacks ``prediction``."""

    value = raw.strip()
    value = re.sub(r"^```(?:xml|markdown|text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(r"<Metadata\b[^>]*>.*?</Metadata>\s*", "", value, flags=re.I | re.S)
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(
        r"</(?:Line|Header|Paragraph|TextBlock|Page)>\s*",
        "\n",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value)


def normalize_loc_draft(value: str) -> tuple[str, dict]:
    original = str(value or "")
    text = unicodedata.normalize("NFC", original)
    transport_markup_removed = bool(CHURRO_TRANSPORT_XML.search(text))
    if transport_markup_removed:
        text = plain_text_from_raw(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"^```(?:xml|markdown|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = UNCERTAINTY_PLACEHOLDER.sub("[?]", text)
    text = re.sub(r"\[\s*\?\s*\]", "[?]", text)
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    text = "\n".join(lines)

    flags: list[str] = []
    if not text.strip():
        flags.append("nothing_to_transcribe_candidate")
    if OMISSION_LANGUAGE.search(text):
        flags.append("explicit_omission_language")
    if XML_TAG.search(text):
        flags.append("xml_or_html_markup_remaining")
    if "```" in text:
        flags.append("markdown_fence_remaining")
    if UNCERTAINTY_PLACEHOLDER.search(text):
        flags.append("non_loc_uncertainty_placeholder")
    if POSSIBLE_BROKEN_WORD.search(text):
        flags.append("possible_line_broken_word_requires_visual_review")

    hard_failures = {
        "explicit_omission_language",
        "xml_or_html_markup_remaining",
        "markdown_fence_remaining",
        "non_loc_uncertainty_placeholder",
    }
    audit = {
        "schema_version": "loc_btp_draft_audit.v1",
        "transcription_standard": "loc-btp-draft",
        "automatic_format_checks_passed": not bool(hard_failures.intersection(flags)),
        "human_review_required": True,
        "flags": flags,
        "visual_review_checks": list(VISUAL_REVIEW_CHECKS),
        "guideline_urls": GUIDELINE_URLS,
        "text_changed_by_exporter": text != original.strip(),
        "transport_markup_removed": transport_markup_removed,
    }
    return text, audit


def safe_id(row: dict, index: int) -> str:
    value = str(row.get("id") or row.get("page_id") or f"page-{index:06d}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or f"page-{index:06d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)
    drafts_dir = args.output / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    export_path = args.output / "loc_drafts.jsonl"
    temporary = export_path.with_suffix(".jsonl.part")
    flag_counts: Counter[str] = Counter()
    automatic_passes = 0

    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, start=1):
            source = str(row.get("prediction") or "")
            if not source.strip():
                source = plain_text_from_raw(str(row.get("raw_prediction") or ""))
            draft, audit = normalize_loc_draft(source)
            identifier = safe_id(row, index)
            (drafts_dir / f"{identifier}.txt").write_text(
                draft.rstrip() + ("\n" if draft else ""), encoding="utf-8"
            )
            flag_counts.update(audit["flags"])
            automatic_passes += int(audit["automatic_format_checks_passed"])
            exported = {
                **row,
                "prediction": draft,
                "transcription_standard": "loc-btp-draft",
                "loc_btp_audit": audit,
                "source_prediction_manifest": str(args.predictions.resolve()),
                "eligible_for_supervised_training": False,
                "model_prediction_is_not_ground_truth": True,
            }
            handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
    os.replace(temporary, export_path)

    summary = {
        "schema_version": "loc_btp_draft_export.v1",
        "source_predictions": str(args.predictions.resolve()),
        "exported_pages": len(rows),
        "automatic_format_checks_passed": automatic_passes,
        "automatic_format_checks_failed": len(rows) - automatic_passes,
        "human_review_required": len(rows),
        "flag_counts": dict(sorted(flag_counts.items())),
        "guideline_urls": GUIDELINE_URLS,
        "loc_drafts": str(export_path.resolve()),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
