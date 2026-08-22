#!/usr/bin/env python3
"""Build an inference manifest from LOC page images already on disk.

This lets inference run while a larger sequential downloader is still active.
The output is safe to pass to ``evaluate_churro_fullpage_qlora.py`` because it
contains only complete JPEG files and preserves the live-audit metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-manifest", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--inventory-source", type=Path)
    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be positive")
    if not args.remote_manifest.is_file():
        parser.error(f"remote manifest not found: {args.remote_manifest}")

    selected: list[dict] = []
    with args.remote_manifest.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image = (
                args.images
                / str(row["item_id"])
                / f"page_{int(row['page_number']):04d}.jpg"
            )
            if not image.is_file() or image.stat().st_size < 1024:
                continue
            normalized = dict(row)
            normalized.pop("epoch19_prediction_is_not_ground_truth", None)
            normalized.update(
                {
                    "image": str(image.resolve()),
                    "model_prediction_is_not_ground_truth": True,
                    "eligible_for_supervised_training": False,
                    "official_transcript_available": False,
                    "inference_model_role": "best_first_occurrence_grounded_churro",
                }
            )
            if args.inventory_source:
                normalized["inventory_source"] = str(args.inventory_source.resolve())
            selected.append(normalized)
            if len(selected) >= args.limit:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "pages": len(selected)}))


if __name__ == "__main__":
    main()
