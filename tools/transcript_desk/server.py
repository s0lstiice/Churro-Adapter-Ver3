#!/usr/bin/env python3
"""Local side-by-side viewer for LOC scans and CHURRO draft transcriptions."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS = (
    APP_ROOT.parents[1]
    / "outputs"
    / "confirmed_untranscribed_loc"
    / "predictions_grounded"
    / "predictions.jsonl"
)


def local_path(value: object) -> Path:
    raw = str(value or "").strip()
    if os.name == "nt" and raw.startswith("/mnt/") and len(raw) > 7:
        drive = raw[5]
        raw = f"{drive.upper()}:/{raw[7:]}"
    return Path(raw)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
    return rows


def safe_identifier(value: object, fallback: str) -> str:
    identifier = str(value or fallback).strip()
    return identifier or fallback


@dataclass(frozen=True)
class Record:
    identifier: str
    payload: dict
    image: Path

    @property
    def prediction(self) -> str:
        return str(self.payload.get("prediction") or "").strip()

    @property
    def incomplete(self) -> bool:
        return bool(self.payload.get("generation_incomplete"))

    @property
    def truncated(self) -> bool:
        return bool(self.payload.get("generation_truncated"))

    @property
    def omission(self) -> bool:
        reasons = " ".join(str(value) for value in self.payload.get("generation_incomplete_reasons") or [])
        return "omission" in reasons.casefold()

    @property
    def loc_draft(self) -> bool:
        return self.payload.get("transcription_standard") == "loc-btp-draft"

    @property
    def loc_audit(self) -> dict:
        value = self.payload.get("loc_btp_audit")
        return value if isinstance(value, dict) else {}

    @property
    def loc_audit_flags(self) -> list[str]:
        return [str(value) for value in self.loc_audit.get("flags") or []]

    def summary(self, index: int) -> dict:
        return {
            "id": self.identifier,
            "index": index,
            "item_id": str(self.payload.get("item_id") or ""),
            "page_number": self.payload.get("page_number"),
            "title": str(self.payload.get("title") or ""),
            "incomplete": self.incomplete,
            "truncated": self.truncated,
            "omission": self.omission,
            "characters": len(self.prediction),
            "loc_draft": self.loc_draft,
            "loc_automatic_checks_passed": bool(
                self.loc_audit.get("automatic_format_checks_passed")
            ),
            "loc_audit_flags": self.loc_audit_flags,
        }

    def detail(self, index: int) -> dict:
        reasons = [str(value) for value in self.payload.get("generation_incomplete_reasons") or []]
        loc_url = str(self.payload.get("loc_url") or "").strip()
        if not re.match(r"^https?://", loc_url, flags=re.IGNORECASE):
            loc_url = ""
        return {
            **self.summary(index),
            "prediction": self.prediction,
            "image_url": f"/api/image?id={quote(self.identifier)}",
            "image_path": str(self.image),
            "loc_url": loc_url,
            "writer_name": str(self.payload.get("writer_name") or "unknown"),
            "reasons": reasons,
            "reference_available": bool(self.payload.get("reference_available")),
            "verification": str(self.payload.get("untranscribed_verification") or ""),
            "loc_human_review_required": bool(
                self.loc_audit.get("human_review_required", self.loc_draft)
            ),
        }


class RecordStore:
    def __init__(self, predictions: Path) -> None:
        self.predictions = predictions.resolve()
        self.lock = threading.RLock()
        self.modified_ns = -1
        self.records: list[Record] = []
        self.by_id: dict[str, tuple[int, Record]] = {}
        self.reload(force=True)

    def reload(self, force: bool = False) -> bool:
        try:
            modified_ns = self.predictions.stat().st_mtime_ns
        except OSError:
            return False
        if not force and modified_ns == self.modified_ns:
            return False
        try:
            rows = read_jsonl(self.predictions)
        except (OSError, ValueError):
            # Inference may be replacing/appending the file at this instant.
            # Keep serving the last complete snapshot and retry next request.
            return False
        records: list[Record] = []
        by_id: dict[str, tuple[int, Record]] = {}
        for index, row in enumerate(rows):
            identifier = safe_identifier(row.get("id") or row.get("page_id"), f"page-{index + 1}")
            if identifier in by_id:
                identifier = f"{identifier}-{index + 1}"
            record = Record(identifier, row, local_path(row.get("image")).resolve())
            by_id[identifier] = (index, record)
            records.append(record)
        with self.lock:
            self.records = records
            self.by_id = by_id
            self.modified_ns = modified_ns
        return True

    def counts(self) -> dict:
        return {
            "total": len(self.records),
            "complete": sum(not record.incomplete for record in self.records),
            "incomplete": sum(record.incomplete for record in self.records),
            "truncated": sum(record.truncated for record in self.records),
            "omission": sum(record.omission for record in self.records),
            "loc_flagged": sum(bool(record.loc_audit_flags) for record in self.records),
        }

    def filtered(self, query: str, status: str) -> list[dict]:
        needle = query.casefold().strip()
        output = []
        for index, record in enumerate(self.records):
            if status == "complete" and record.incomplete:
                continue
            if status == "incomplete" and not record.incomplete:
                continue
            if status == "truncated" and not record.truncated:
                continue
            if status == "omission" and not record.omission:
                continue
            if status == "loc_flagged" and not record.loc_audit_flags:
                continue
            if needle:
                haystack = "\n".join(
                    (
                        record.identifier,
                        str(record.payload.get("item_id") or ""),
                        str(record.payload.get("title") or ""),
                        record.prediction,
                    )
                ).casefold()
                if needle not in haystack:
                    continue
            output.append(record.summary(index))
        return output


STORE: RecordStore


class Handler(BaseHTTPRequestHandler):
    server_version = "TranscriptDesk/1.0"

    def send_json(self, value: object, status: int = 200) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        STORE.reload()
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/health":
            self.send_json(
                {
                    "ok": True,
                    "records": len(STORE.records),
                    "predictions": str(STORE.predictions),
                    "live_reload": True,
                }
            )
            return
        if parsed.path == "/api/list":
            status = str(query.get("status", ["all"])[0]).casefold()
            if status not in {
                "all",
                "complete",
                "incomplete",
                "truncated",
                "omission",
                "loc_flagged",
            }:
                self.send_json({"error": "invalid status"}, 400)
                return
            search = str(query.get("q", [""])[0])
            self.send_json({"counts": STORE.counts(), "items": STORE.filtered(search, status)})
            return
        if parsed.path == "/api/item":
            identifier = str(query.get("id", [""])[0])
            found = STORE.by_id.get(identifier)
            if found is None:
                self.send_json({"error": "page not found"}, 404)
                return
            index, record = found
            self.send_json(record.detail(index))
            return
        if parsed.path == "/api/image":
            identifier = str(query.get("id", [""])[0])
            found = STORE.by_id.get(identifier)
            if found is None:
                self.send_error(404)
                return
            self.send_file(found[1].image)
            return
        if parsed.path == "/api/transcript":
            identifier = str(query.get("id", [""])[0])
            found = STORE.by_id.get(identifier)
            if found is None:
                self.send_error(404)
                return
            data = (found[1].prediction + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="{re.sub(r"[^A-Za-z0-9_.-]", "_", identifier)}.txt"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        static_path = APP_ROOT / ("index.html" if parsed.path == "/" else parsed.path.lstrip("/"))
        try:
            static_path = static_path.resolve()
            static_path.relative_to(APP_ROOT)
        except ValueError:
            self.send_error(403)
            return
        self.send_file(static_path)

    def log_message(self, *_args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--port", type=int, default=8813)
    args = parser.parse_args()
    if not args.predictions.is_file():
        parser.error(f"predictions file not found: {args.predictions}")
    global STORE
    STORE = RecordStore(args.predictions)
    print(f"Loaded {len(STORE.records)} pages from {STORE.predictions}", flush=True)
    print(f"Open http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
