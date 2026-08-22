#!/usr/bin/env python3
"""Run the selected best CHURRO adapter on live-audited, untranscribed LOC pages.

This runner deliberately reuses the fail-closed live LOC inventory created by
``run_epoch19_on_loc_without_official_transcription.py``.  The inventory only
contains items whose LOC item JSON advertised image resources and no official
full text, transcript, word coordinates, or text-service resource.  Predictions
are unreviewed OCR output, never references or supervised-training labels.

Downloads and inference are resumable.  The selected decoder performs the
reference-free counterfactual visual-grounding check while each page is still
loaded, so a stopped run never leaves completed pages waiting for a separate
grounding pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from run_epoch19_on_loc_without_official_transcription import download_pages
from universal_progress_monitor.progress_client import ProgressTask


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE_MANIFEST = (
    REPO_ROOT / "data" / "verified_remote_pages.jsonl"
)
DEFAULT_INVENTORY_SUMMARY = (
    REPO_ROOT / "data" / "inventory_summary.json"
)
DEFAULT_ADAPTER = REPO_ROOT / "adapter"
DEFAULT_OUT = REPO_ROOT / "outputs" / "confirmed_untranscribed_loc"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_remote_manifest(args: argparse.Namespace) -> tuple[Path, int]:
    rows = read_jsonl(args.remote_manifest)
    if args.maximum_pages:
        rows = rows[: args.maximum_pages]
    destination = args.out / "verified_remote_pages.jsonl"
    write_jsonl(destination, rows)
    return destination, len(rows)


def neutralize_manifest(path: Path, inventory_source: Path) -> int:
    rows = read_jsonl(path)
    normalized = []
    for row in rows:
        row = dict(row)
        row.pop("epoch19_prediction_is_not_ground_truth", None)
        row.update(
            {
                "model_prediction_is_not_ground_truth": True,
                "eligible_for_supervised_training": False,
                "official_transcript_available": False,
                "inventory_source": str(inventory_source.resolve()),
                "inference_model_role": "best_first_occurrence_grounded_churro",
            }
        )
        normalized.append(row)
    write_jsonl(path, normalized)
    return len(normalized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-manifest", type=Path, default=DEFAULT_REMOTE_MANIFEST)
    parser.add_argument("--inventory-summary", type=Path, default=DEFAULT_INVENTORY_SUMMARY)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--maximum-pages", type=int, default=0)
    parser.add_argument("--download-delay", type=float, default=0.15)
    parser.add_argument("--max-pixels", type=int, default=1_605_632)
    parser.add_argument("--max-new-tokens", type=int, default=1_536)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    args.out = args.out.resolve()
    args.remote_manifest = args.remote_manifest.resolve()
    args.inventory_summary = args.inventory_summary.resolve()
    args.adapter = args.adapter.resolve()
    if args.maximum_pages < 0:
        parser.error("--maximum-pages must be zero (all pages) or positive")
    if not args.remote_manifest.is_file():
        parser.error(f"verified remote manifest not found: {args.remote_manifest}")
    if not args.inventory_summary.is_file():
        parser.error(f"inventory audit summary not found: {args.inventory_summary}")
    if not args.adapter.is_dir() and not args.download_only:
        parser.error(f"selected adapter not found: {args.adapter}")
    args.out.mkdir(parents=True, exist_ok=True)

    remote_manifest, page_count = selected_remote_manifest(args)
    metadata = {
        "remote_manifest": str(args.remote_manifest),
        "inventory_summary": str(args.inventory_summary),
        "adapter": str(args.adapter),
        "pages": page_count,
        "decode_profile": "grounded-faithful",
    }
    with ProgressTask(
        "Best CHURRO on confirmed untranscribed LOC pages",
        total=3,
        unit="stages",
        task_id="best-churro-confirmed-untranscribed-loc-v1",
        output_dir=args.out,
        metadata=metadata,
    ) as task:
        task.update(1, message=f"verified inventory selected: {page_count} pages")
        local_manifest = download_pages(args, remote_manifest)
        downloaded = neutralize_manifest(local_manifest, args.inventory_summary)
        task.update(2, message=f"download stage complete: {downloaded} pages")
        if args.download_only:
            task.finish("download-only run complete", metrics={"downloaded": downloaded})
            return

        command = [
            sys.executable,
            str(SCRIPT_ROOT / "evaluate_churro_fullpage_qlora.py"),
            "--manifest",
            str(local_manifest),
            "--adapter",
            str(args.adapter),
            "--output",
            str(args.out / "predictions_grounded"),
            "--max-pixels",
            str(args.max_pixels),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--decode-profile",
            "grounded-faithful",
            "--max-incomplete-retries",
            "0",
        ]
        try:
            subprocess.run(command, cwd=SCRIPT_ROOT, check=True)
        except Exception as error:
            task.fail(f"{type(error).__name__}: {error}")
            raise
        task.update(3, message="inference complete")
        task.finish("all confirmed untranscribed LOC pages complete")


if __name__ == "__main__":
    main()
