#!/usr/bin/env python3
"""Keep a LOC-formatted draft manifest synchronized with live predictions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from universal_progress_monitor.progress_client import ProgressTask


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exporter", type=Path, default=Path(__file__).with_name("export_loc_btp_drafts.py"))
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--total-pages", type=int)
    parser.add_argument("--task-id", default="loc-btp-live-draft-export")
    args = parser.parse_args()

    predictions = args.predictions.resolve()
    output = args.output.resolve()
    exporter = args.exporter.resolve()
    if not predictions.is_file():
        parser.error(f"predictions file not found: {predictions}")
    if not exporter.is_file():
        parser.error(f"exporter not found: {exporter}")
    output.mkdir(parents=True, exist_ok=True)

    task = ProgressTask(
        "LOC-formatted live drafts",
        total=None,
        unit="pages",
        task_id=args.task_id,
        output_dir=output,
        metadata={"predictions": str(predictions), "exporter": str(exporter)},
    )
    previous_signature: tuple[int, int] | None = None
    try:
        while True:
            stat = predictions.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if signature != previous_signature:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(exporter),
                        "--predictions",
                        str(predictions),
                        "--output",
                        str(output),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                )
                if result.returncode == 0:
                    current = row_count(output / "loc_drafts.jsonl")
                    task.update(
                        current=current,
                        message=f"{current} LOC-format drafts synchronized",
                        metrics={
                            "source_rows": row_count(predictions),
                            "expected_pages": args.total_pages,
                        },
                    )
                    print(json.dumps({"current": current, "total": args.total_pages, "status": "synchronized"}), flush=True)
                    previous_signature = signature
                    if args.total_pages is not None and current >= args.total_pages:
                        task.finish(f"all {current} LOC-format drafts synchronized")
                        return 0
                else:
                    task.message = "source changed during append; retrying"
                    task.metadata["last_export_error"] = result.stdout[-1000:]
                    task.write()
                    print(result.stdout.rstrip(), flush=True)
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        task.finish("live formatter stopped")
        return 0
    except BaseException as error:
        task.fail(f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
