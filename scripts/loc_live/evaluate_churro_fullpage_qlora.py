#!/usr/bin/env python3
"""Evaluate base CHURRO or a CHURRO QLoRA adapter on page manifests."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from churro_decode_processors import faithful_logits_processors
from churro_counterfactual_grounding import counterfactual_grounded_tail_guard
from churro_par_qwen25 import PARConfig, install_par, parse_layers
from run_churro_loc_bakeoff import score, visible_text
from universal_progress_monitor.progress_client import ProgressTask


MODEL_ID = "stanford-oval/churro-3B"
DECODE_PROFILES = (
    "legacy",
    "churro-native",
    "faithful",
    "grounded-faithful",
    "faithful-beam2",
    "par-pp",
    "par-full",
)
SYSTEM_PROMPT = "Transcribe the entirety of this historical document to XML format."
RETRY_SYSTEM_PROMPT = (
    "Transcribe every visible word in this entire historical document to XML. "
    "Preserve reading order, spelling, capitalization, and punctuation. Never summarize, "
    "shorten, skip, or replace visible text with comments or placeholders such as "
    "'omitted for brevity'. Continue through the bottom of the page and close the XML only "
    "after all visible text has been transcribed."
    " This is an exhaustive-transcription retry because a prior response skipped content. "
    "Start again from the top. Output every visible line. If an individual word is unreadable, "
    "mark only that word as uncertain; never omit a region, paragraph, or remainder of the page."
)
PLAIN_RETRY_SYSTEM_PROMPT = (
    "Read this entire historical document from top to bottom and return only the complete plain-text "
    "transcription. Transcribe every visible line in reading order. Never summarize, abbreviate, "
    "skip content, add XML, or write comments/placeholders such as 'omitted for brevity'. If one "
    "word is unreadable, mark only that word as [unclear] and continue through the bottom of the page."
)
LINE_SYSTEM_PROMPT = (
    "Transcribe this single handwritten line exactly. Preserve the visible spelling, "
    "capitalization, and punctuation. Return only the transcription."
)


def task_type(row: dict) -> str:
    value = str(row.get("task_type") or row.get("granularity") or "page").strip().lower()
    if value in {"full_page", "full-page", "document"}:
        value = "page"
    if value not in {"page", "line"}:
        raise ValueError(f"{row.get('id', '<unknown>')} has unsupported task_type={value!r}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def messages_for(row: dict, max_pixels: int, retry_variant: int = 0) -> list[dict]:
    if task_type(row) == "line":
        prompt = LINE_SYSTEM_PROMPT
    elif retry_variant == 1:
        prompt = RETRY_SYSTEM_PROMPT
    elif retry_variant == 2:
        prompt = PLAIN_RETRY_SYSTEM_PROMPT
    else:
        prompt = SYSTEM_PROMPT
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"file://{Path(row['image']).resolve()}",
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": max_pixels,
                }
            ],
        },
    ]


def load_model(model_id: str, adapter: Path | None):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor_source = str(adapter) if adapter else model_id
    processor = AutoProcessor.from_pretrained(processor_source, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        device_map={"": 0},
        quantization_config=quantization,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return processor, model


def generation_settings(decode_profile: str) -> dict:
    """Return deterministic generation settings for a named decode profile.

    ``legacy`` preserves this project's historical evaluator exactly.  The
    other profiles remove the global 8-gram ban, which is not present in the
    released CHURRO generation configuration and can force valid repeated
    phrases to be substituted or omitted.  CHURRO's mild repetition penalty is
    retained because fully neutral greedy decoding can enter XML line loops.
    """

    if decode_profile not in DECODE_PROFILES:
        raise ValueError(f"unknown decode profile {decode_profile!r}")
    faithful = decode_profile in {
        "faithful",
        "grounded-faithful",
        "faithful-beam2",
        "par-pp",
        "par-full",
    }
    settings = {
        "do_sample": False,
        "repetition_penalty": 1.0 if faithful else 1.05,
        "no_repeat_ngram_size": (
            8 if decode_profile == "legacy" else (32 if faithful else 0)
        ),
        "use_cache": True,
        "response_only_repetition_penalty": 1.01 if faithful else None,
        "targeted_loop_guard": (
            "visual_grounding_plus_repetitive_xml_tail"
            if decode_profile == "grounded-faithful"
            else ("no_repeat_ngram_size_32" if faithful else None)
        ),
    }
    if decode_profile == "faithful-beam2":
        # Beam width two is a bounded delayed-commitment diagnostic: a recent
        # alternative can survive long enough for later image-conditioned
        # evidence to select it.  It is deliberately not a default because it
        # can be slower and can strengthen the language prior on some pages.
        settings.update(
            {
                "num_beams": 2,
                "num_return_sequences": 1,
                "early_stopping": False,
                "length_penalty": 1.0,
            }
        )
    return settings


def model_generation_settings(decode_profile: str) -> dict:
    """Strip audit-only keys before calling ``model.generate``."""

    return {
        key: value
        for key, value in generation_settings(decode_profile).items()
        if key not in {"response_only_repetition_penalty", "targeted_loop_guard"}
    }


@torch.inference_mode()
def infer(
    processor,
    model,
    row: dict,
    max_pixels: int,
    max_new_tokens: int,
    retry_variant: int = 0,
    decode_profile: str = "legacy",
    par_controller=None,
) -> str:
    messages = messages_for(row, max_pixels, retry_variant=retry_variant)
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    batch = processor(
        text=[rendered],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    if par_controller is not None:
        par_controller.begin_page(batch.input_ids)
    generate_kwargs = model_generation_settings(decode_profile)
    if decode_profile in {
        "faithful",
        "grounded-faithful",
        "faithful-beam2",
        "par-pp",
        "par-full",
    }:
        generate_kwargs["logits_processor"] = faithful_logits_processors(
            prompt_length=int(batch.input_ids.shape[-1])
        )
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        **generate_kwargs,
    )
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(batch.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def counterfactual_negative_for(row: dict, candidates: list[dict]) -> dict | None:
    """Pick a deterministic unrelated page for visual counterfactual scoring."""

    source_image = str(row.get("image") or "")
    source_item = str(row.get("item_id") or row.get("source") or "")
    eligible = [
        candidate
        for candidate in candidates
        if task_type(candidate) == task_type(row)
        and str(candidate.get("image") or "") != source_image
        and str(candidate.get("item_id") or candidate.get("source") or "") != source_item
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda candidate: str(candidate.get("id") or candidate.get("page_id")))[0]


def apply_counterfactual_grounding_guard(
    processor,
    model,
    row: dict,
    negative: dict | None,
    raw: str,
    max_pixels: int,
    retry_variant: int,
) -> tuple[str, dict]:
    if negative is None:
        return raw, {
            "schema_version": "churro_counterfactual_grounding.v1",
            "applied": False,
            "reason": "no_unrelated_counterfactual_page_available",
            "counterfactual_forward_passes": 0,
        }
    negative_row = dict(row)
    negative_row["image"] = negative["image"]
    guarded, audit = counterfactual_grounded_tail_guard(
        processor,
        model,
        messages_for(row, max_pixels, retry_variant=retry_variant),
        messages_for(negative_row, max_pixels, retry_variant=retry_variant),
        raw,
    )
    audit["counterfactual_page_id"] = str(negative.get("id") or negative.get("page_id"))
    return guarded, audit


OMISSION_PATTERNS = (
    re.compile(r"omitted\s+for\s+brevity", re.IGNORECASE),
    # Models sometimes hide an omission in an XML comment.  A closed XML
    # document is not complete when its body contains ``<!-- ... -->``.
    re.compile(
        r"<!--(?:(?!-->).)*(?:\.{3,}|…+|omit(?:ted|s|ting)?|brevity|remainder|continues?)(?:(?!-->).)*-->",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:the\s+)?(?:rest|remainder)\s+of\s+(?:the\s+)?"
        r"(?:page|document|text|content|letter|letter\s+body|body).{0,100}?"
        r"(?:is\s+)?(?:omitted|not\s+transcribed|not\s+included)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:page|document|text|content|letter|letter\s+body|body).{0,80}?"
        r"(?:omitted|not\s+transcribed)\s+(?:for|due\s+to)\s+",
        re.IGNORECASE | re.DOTALL,
    ),
)


def incomplete_generation_reasons(raw: str, row: dict) -> list[str]:
    if task_type(row) != "page":
        return []
    reasons = []
    if "</HistoricalDocument>" not in raw:
        reasons.append("unclosed_historical_document")
    if any(pattern.search(raw) for pattern in OMISSION_PATTERNS):
        reasons.append("explicit_omission_placeholder")
    return reasons


def summarize(rows: list[dict]) -> dict:
    totals = defaultdict(int)
    scored_rows = [
        row for row in rows
        if isinstance(row.get("metrics"), dict) and str(row.get("text") or "").strip()
    ]
    for row in scored_rows:
        for key in (
            "character_edits",
            "target_characters",
            "word_edits",
            "target_words",
            "prediction_characters",
            "prediction_words",
        ):
            totals[key] += row["metrics"][key]
    result = {
        "examples": len(rows),
        "scored_examples": len(scored_rows),
        "unscored_examples": len(rows) - len(scored_rows),
        "pages": sum(task_type(row) == "page" for row in rows),
        "lines": sum(task_type(row) == "line" for row in rows),
        **totals,
        "cer": (
            totals["character_edits"] / max(1, totals["target_characters"])
            if scored_rows else None
        ),
        "wer": (
            totals["word_edits"] / max(1, totals["target_words"])
            if scored_rows else None
        ),
        "output_character_ratio": (
            totals["prediction_characters"] / max(1, totals["target_characters"])
            if scored_rows else None
        ),
        "generation_truncated_examples": sum(bool(row.get("generation_truncated")) for row in rows),
        "generation_truncation_rate": sum(bool(row.get("generation_truncated")) for row in rows)
        / max(1, len(rows)),
        "explicit_omission_examples": sum(
            "explicit_omission_placeholder" in row.get("generation_incomplete_reasons", [])
            for row in rows
        ),
        "generation_incomplete_examples": sum(bool(row.get("generation_incomplete")) for row in rows),
        "retry_attempted_examples": sum(bool(row.get("generation_retry_used")) for row in rows),
        "retry_succeeded_examples": sum(bool(row.get("generation_retry_succeeded")) for row in rows),
    }
    by_task = {}
    for name in ("line", "page"):
        selected = [row for row in rows if task_type(row) == name]
        if selected:
            by_task[name] = summarize(selected) if len(selected) < len(rows) else {
                "examples": len(selected),
                "cer": result["cer"],
                "wer": result["wer"],
            }
    if len(by_task) > 1:
        result["by_task"] = by_task
    return result


def clean_text_path(output: Path, row_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_id).strip("_.") or "prediction"
    return output / "clean_text" / f"{safe_id}.txt"


def write_clean_text(output: Path, row: dict) -> None:
    destination = clean_text_path(output, str(row.get("id") or row.get("page_id") or "prediction"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(str(row.get("prediction") or "").rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("churro_loc_fullpage_dataset_v1/test.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--decode-profile",
        choices=DECODE_PROFILES,
        default="legacy",
        help=(
            "legacy keeps the prior 8-gram ban; churro-native removes it; faithful also "
            "limits repetition handling to generated-text loops; grounded-faithful removes "
            "only visually unsupported repetitive XML tails; par-pp/par-full add the Qwen2.5-VL "
            "OCR grounding intervention"
        ),
    )
    parser.add_argument(
        "--par-layers",
        default="0,1,2,3",
        help="Comma-separated language-model layers for PAR profiles.",
    )
    parser.add_argument("--par-seed", type=int, default=1729)
    parser.add_argument("--id", action="append", default=[], help="Evaluate only these row IDs/page IDs.")
    parser.add_argument(
        "--max-incomplete-retries",
        type=int,
        default=2,
        help="Retry page generations that truncate or explicitly omit visible content.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_manifest_rows = read_jsonl(args.manifest)
    rows = list(all_manifest_rows)
    if args.id:
        requested = set(args.id)
        rows = [row for row in rows if str(row.get("id") or row.get("page_id")) in requested]
        missing = requested - {str(row.get("id") or row.get("page_id")) for row in rows}
        if missing:
            raise KeyError(f"requested IDs not found in manifest: {sorted(missing)}")
    if args.limit:
        rows = rows[: args.limit]
    predictions_path = args.output / "predictions.jsonl"
    existing = read_jsonl(predictions_path) if predictions_path.exists() else []
    existing_changed = False
    for result in existing:
        refreshed_reasons = incomplete_generation_reasons(
            str(result.get("raw_prediction") or ""), result
        )
        if result.get("generation_incomplete_reasons") != refreshed_reasons:
            result["generation_incomplete_reasons"] = refreshed_reasons
            result["generation_incomplete"] = bool(refreshed_reasons)
            existing_changed = True
        if not str(result.get("text") or "").strip() and result.get("metrics") is not None:
            result["metrics"] = None
            existing_changed = True
        write_clean_text(args.output, result)
    if existing_changed:
        with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
            for result in existing:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    done = {row["id"] for row in existing}
    pending = [row for row in rows if row["id"] not in done]
    run_name = "adapter" if args.adapter else "base"
    task_counts = {name: sum(task_type(row) == name for row in rows) for name in ("line", "page")}
    mixed = all(task_counts.values())
    task = ProgressTask(
        f"CHURRO {'mixed-scale' if mixed else ('line' if task_counts['line'] else 'full-page')} {run_name} evaluation",
        total=len(rows),
        unit="examples",
        task_id=f"churro-{'mixed' if mixed else ('line' if task_counts['line'] else 'fullpage')}-{run_name}-eval-{args.output.name}",
        output_dir=args.output,
        metadata={
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "task_counts": task_counts,
            "decode_profile": args.decode_profile,
        },
    )
    try:
        task.update(len(existing), message="loading model")
        processor = model = par_controller = None
        if pending:
            processor, model = load_model(args.model, args.adapter)
            if args.decode_profile in {"par-pp", "par-full"}:
                par_controller = install_par(
                    model,
                    PARConfig(
                        pp_enabled=args.decode_profile in {"par-pp", "par-full"},
                        far_enabled=args.decode_profile == "par-full",
                        target_layers=parse_layers(args.par_layers),
                        seed=args.par_seed,
                    ),
                )
        with predictions_path.open("a", encoding="utf-8") as handle:
            for row in pending:
                counterfactual_negative = counterfactual_negative_for(row, all_manifest_rows)
                grounding_attempts = []
                first_raw = infer(
                    processor,
                    model,
                    row,
                    args.max_pixels,
                    args.max_new_tokens,
                    decode_profile=args.decode_profile,
                    par_controller=par_controller,
                )
                first_audit = {}
                if args.decode_profile == "grounded-faithful":
                    guarded, first_audit = apply_counterfactual_grounding_guard(
                        processor,
                        model,
                        row,
                        counterfactual_negative,
                        first_raw,
                        args.max_pixels,
                        retry_variant=0,
                    )
                    if first_audit.get("applied"):
                        first_audit["_raw_prediction_before_guard"] = first_raw
                    first_raw = guarded
                grounding_attempts.append(first_audit)
                raw_attempts = [
                    first_raw
                ]
                reason_attempts = [incomplete_generation_reasons(raw_attempts[0], row)]
                initial_incomplete = bool(reason_attempts[0])
                for retry_index in range(args.max_incomplete_retries if initial_incomplete else 0):
                    retry_variant = 1 + (retry_index % 2)
                    retry_raw = infer(
                        processor,
                        model,
                        row,
                        args.max_pixels,
                        args.max_new_tokens,
                        retry_variant=retry_variant,
                        decode_profile=args.decode_profile,
                        par_controller=par_controller,
                    )
                    retry_audit = {}
                    if args.decode_profile == "grounded-faithful":
                        guarded, retry_audit = apply_counterfactual_grounding_guard(
                            processor,
                            model,
                            row,
                            counterfactual_negative,
                            retry_raw,
                            args.max_pixels,
                            retry_variant=retry_variant,
                        )
                        if retry_audit.get("applied"):
                            retry_audit["_raw_prediction_before_guard"] = retry_raw
                        retry_raw = guarded
                    raw_attempts.append(retry_raw)
                    grounding_attempts.append(retry_audit)
                    reason_attempts.append(incomplete_generation_reasons(raw_attempts[-1], row))
                # Prefer a complete candidate. If every attempt is incomplete,
                # retain the one with the most visible text rather than hiding
                # the failure by stripping its placeholder.
                complete_indices = [index for index, reasons in enumerate(reason_attempts) if not reasons]
                complete_xml_indices = [
                    index
                    for index in complete_indices
                    if "</HistoricalDocument>" in raw_attempts[index]
                ]
                if complete_xml_indices:
                    # Prefer a complete structured transcription. A longer
                    # plain-text retry can be verbose or hallucinatory, so it
                    # is only a fallback when exhaustive XML also fails.
                    selected_attempt = max(
                        complete_xml_indices,
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                elif complete_indices:
                    selected_attempt = max(
                        complete_indices,
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                else:
                    selected_attempt = max(
                        range(len(raw_attempts)),
                        key=lambda index: len(visible_text(raw_attempts[index])),
                    )
                raw = raw_attempts[selected_attempt]
                selected_grounding_audit = grounding_attempts[selected_attempt]
                raw_before_grounding_guard = selected_grounding_audit.pop(
                    "_raw_prediction_before_guard", None
                )
                prediction = visible_text(raw)
                incomplete_reasons = reason_attempts[selected_attempt]
                result = {
                    **row,
                    "model": args.model,
                    "adapter": str(args.adapter) if args.adapter else None,
                    "decode_profile": args.decode_profile,
                    "generation_settings": generation_settings(args.decode_profile),
                    "par_audit": par_controller.audit() if par_controller is not None else None,
                    "grounding_guard": selected_grounding_audit or None,
                    "raw_prediction_before_grounding_guard": raw_before_grounding_guard,
                    "raw_prediction": raw,
                    "prediction": prediction,
                    "generation_max_new_tokens": args.max_new_tokens,
                    "generation_truncated": task_type(row) == "page"
                    and "</HistoricalDocument>" not in raw,
                    "generation_incomplete": bool(incomplete_reasons),
                    "generation_incomplete_reasons": incomplete_reasons,
                    "generation_attempts": len(raw_attempts),
                    "generation_selected_attempt": selected_attempt + 1,
                    "generation_retry_used": len(raw_attempts) > 1,
                    "generation_retry_succeeded": bool(reason_attempts[0]) and not incomplete_reasons,
                    "generation_attempt_audit": [
                        {
                            "attempt": index + 1,
                            "reasons": reasons,
                            "visible_characters": len(visible_text(candidate)),
                            "format": "xml" if "</HistoricalDocument>" in candidate else "plain_or_unclosed",
                            "grounding_guard_applied": bool(
                                grounding_attempts[index].get("applied")
                            ),
                        }
                        for index, (candidate, reasons) in enumerate(zip(raw_attempts, reason_attempts))
                    ],
                    "metrics": (
                        score(row["text"], prediction)
                        if str(row.get("text") or "").strip() else None
                    ),
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                write_clean_text(args.output, result)
                existing.append(result)
                latest_metrics = result.get("metrics") or {}
                task.update(
                    len(existing),
                    message=row["id"],
                    metrics=(
                        {"latest_cer": latest_metrics["cer"], "latest_wer": latest_metrics["wer"]}
                        if latest_metrics else {"reference_available": False}
                    ),
                )
        summary = summarize(existing)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        finish_metrics = {"scored_examples": summary["scored_examples"]}
        if summary["cer"] is not None:
            finish_metrics.update({"cer": summary["cer"], "wer": summary["wer"]})
        task.finish("evaluation complete", metrics=finish_metrics)
    except Exception as error:
        task.fail(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
