#!/usr/bin/env python3
"""Find LOC manuscript items with images and no official transcript, then infer.

Discovery uses the live loc.gov JSON API that powers the public website.  The
audit is item-level and fail-closed: an item is rejected if the API advertises
any online-text format, full-text/transcript/word-coordinate field, text-service
URL, textual resource file, missing item metadata, or missing image resources.

The default collection is the public-domain Charles S. Hamlin Papers.  Model
predictions are unreviewed drafts, never official transcripts or training labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from universal_progress_monitor.progress_client import ProgressTask


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COLLECTION_URL = "https://www.loc.gov/collections/charles-s-hamlin-papers/"
DEFAULT_ADAPTER = REPO_ROOT / "adapter"
DEFAULT_OUT = REPO_ROOT / "outputs" / "live_audited_loc"
USER_AGENT = "LOC-handwriting-research/1.0 (noncommercial research; rate-limited)"

TRANSCRIPT_KEY_MARKERS = (
    "fulltext",
    "full_text",
    "transcript",
    "word_coordinates",
    "wordcoordinates",
    "plain_text",
)
TRANSCRIPT_URL_MARKERS = (
    "text-services",
    "plain_text",
    "fulltext",
    "transcript",
    "word-coordinates",
    "word_coordinates",
)
TEXT_MIMETYPES = {
    "text/plain",
    "text/xml",
    "application/xml",
    "application/tei+xml",
}


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def request_bytes(url: str, *, retries: int = 8, minimum_delay: float = 0.35) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt or minimum_delay:
            time.sleep(minimum_delay if attempt == 0 else min(30.0, 1.5 * (2**attempt)))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {429, 500, 502, 503, 504}:
                raise
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            last_error = error
    raise RuntimeError(f"LOC request failed after {retries} attempts: {url}: {last_error}")


def request_json(url: str, cache_path: Path) -> dict:
    if cache_path.is_file():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            pass
    payload = json.loads(request_bytes(url).decode("utf-8-sig"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def api_url(url: str, **parameters: object) -> str:
    parsed = urllib.parse.urlsplit(url.replace("http://", "https://", 1))
    query = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in parameters.items()})
    return urllib.parse.urlunsplit(
        (parsed.scheme or "https", parsed.netloc, parsed.path, urllib.parse.urlencode(query), "")
    )


def item_slug(item_url: str) -> str:
    path = urllib.parse.urlsplit(item_url).path.rstrip("/")
    slug = path.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", slug).strip("_")


def nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = path + (str(key),)
            yield child_path, child
            yield from walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, path + (str(index),))


def transcript_signals(payload: dict) -> list[str]:
    reasons: set[str] = set()
    item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
    formats = item.get("online_format") or payload.get("online_format") or []
    if isinstance(formats, str):
        formats = [formats]
    normalized_formats = {str(value).strip().lower() for value in formats if str(value).strip()}
    if not any(value == "image" for value in normalized_formats):
        reasons.add("online_format_missing_image")
    if any(value not in {"image"} for value in normalized_formats):
        reasons.add("online_format_includes_nonimage:" + ",".join(sorted(normalized_formats)))

    for path, value in walk(payload):
        key = path[-1].lower() if path else ""
        dotted = ".".join(path)
        if any(marker in key for marker in TRANSCRIPT_KEY_MARKERS) and nonempty(value):
            reasons.add(f"transcript_field:{dotted}")
        if isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in TRANSCRIPT_URL_MARKERS):
                reasons.add(f"transcript_url:{dotted}")
            if key == "mimetype" and lowered in TEXT_MIMETYPES:
                reasons.add(f"textual_resource:{dotted}:{lowered}")
    return sorted(reasons)


def flatten_dicts(value: Any) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        rows: list[dict] = []
        for child in value:
            rows.extend(flatten_dicts(child))
        return rows
    return []


def select_jpeg(variants: Any, target_pixels: int) -> dict | None:
    candidates = []
    for variant in flatten_dicts(variants):
        mimetype = str(variant.get("mimetype") or "").lower()
        url = str(variant.get("url") or "")
        if mimetype not in {"image/jpeg", "image/jpg"} or not url.startswith("https://"):
            continue
        width = int(variant.get("width") or 0)
        height = int(variant.get("height") or 0)
        pixels = width * height
        if pixels <= 0:
            continue
        candidates.append((variant, pixels))
    if not candidates:
        return None
    below = [pair for pair in candidates if pair[1] <= target_pixels]
    chosen = max(below, key=lambda pair: pair[1]) if below else min(candidates, key=lambda pair: pair[1])
    return dict(chosen[0])


def item_pages(payload: dict, item_id: str, target_pixels: int) -> list[dict]:
    pages: list[dict] = []
    seen_urls: set[str] = set()
    page_number = 0
    for resource_index, resource in enumerate(payload.get("resources") or []):
        if not isinstance(resource, dict):
            continue
        files = resource.get("files") or []
        groups = files if isinstance(files, list) else [files]
        if groups and all(isinstance(value, dict) for value in groups):
            groups = [groups]
        for group in groups:
            selected = select_jpeg(group, target_pixels)
            if not selected:
                continue
            url = str(selected["url"])
            if url in seen_urls:
                continue
            seen_urls.add(url)
            page_number += 1
            resource_url = str(resource.get("url") or "").replace("http://", "https://", 1)
            pages.append(
                {
                    "id": f"{item_id}_page_{page_number:04d}",
                    "page_id": f"{item_id}_page_{page_number:04d}",
                    "item_id": item_id,
                    "page_number": page_number,
                    "resource_index": resource_index,
                    "image_url": url,
                    "source_width": int(selected.get("width") or 0),
                    "source_height": int(selected.get("height") or 0),
                    "loc_url": (
                        f"{resource_url.rstrip('/')}?sp={page_number}&st=image"
                        if resource_url
                        else ""
                    ),
                }
            )
    return pages


def collection_items(collection_url: str, cache_dir: Path) -> list[dict]:
    results: list[dict] = []
    page_number = 1
    while True:
        url = api_url(collection_url, fo="json", c=100, sp=page_number)
        payload = request_json(url, cache_dir / f"collection_page_{page_number:03d}.json")
        current = payload.get("results") or []
        results.extend(row for row in current if isinstance(row, dict))
        pagination = payload.get("pagination") or {}
        if not pagination.get("next"):
            break
        page_number += 1
    unique: dict[str, dict] = {}
    for row in results:
        url = str(row.get("id") or "").replace("http://", "https://", 1)
        if "/item/" in url:
            unique[url] = row
    return [unique[key] for key in sorted(unique)]


def discover(args: argparse.Namespace) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out / "loc_api_cache"
    items = (
        [{"id": url.replace("http://", "https://", 1), "title": ""} for url in args.item_url]
        if args.item_url
        else collection_items(args.collection_url, cache_dir)
    )
    if args.maximum_items:
        items = items[: args.maximum_items]

    accepted_items: list[dict] = []
    rejected_items: list[dict] = []
    page_rows: list[dict] = []
    reasons = Counter()
    with ProgressTask(
        "Verify LOC items have no official transcription",
        total=len(items),
        unit="items",
        task_id=f"loc-no-transcript-audit-{args.out.name}",
        output_dir=args.out,
        metadata={"collection_url": args.collection_url, "explicit_item_urls": args.item_url},
    ) as task:
        for index, search_row in enumerate(items, start=1):
            item_url = str(search_row["id"]).replace("http://", "https://", 1)
            slug = item_slug(item_url)
            try:
                payload = request_json(
                    api_url(item_url, fo="json"), cache_dir / "items" / f"{slug}.json"
                )
                signals = transcript_signals(payload)
                pages = item_pages(payload, slug, args.target_image_pixels)
                if not pages:
                    signals.append("no_downloadable_jpeg_pages")
                if signals:
                    for signal in signals:
                        reasons[signal.split(":", 1)[0]] += 1
                    rejected_items.append(
                        {
                            "item_id": slug,
                            "item_url": item_url,
                            "title": str((payload.get("item") or {}).get("title") or search_row.get("title") or ""),
                            "reasons": signals,
                        }
                    )
                else:
                    item_row = {
                        "item_id": slug,
                        "item_url": item_url,
                        "title": str((payload.get("item") or {}).get("title") or search_row.get("title") or ""),
                        "online_format": (payload.get("item") or {}).get("online_format") or [],
                        "page_count": len(pages),
                        "official_transcript_status": "no_transcript_or_fulltext_resource_advertised_by_live_loc_item_api",
                        "api_audited_at_unix": int(time.time()),
                    }
                    accepted_items.append(item_row)
                    for page in pages:
                        page_rows.append(
                            {
                                **page,
                                "title": item_row["title"],
                                "item_url": item_url,
                                "text": "",
                                "task_type": "page",
                                "reference_available": False,
                                "official_transcript_available": False,
                                "epoch19_prediction_is_not_ground_truth": True,
                                "eligible_for_supervised_training": False,
                            }
                        )
            except Exception as error:
                reasons["item_api_error"] += 1
                rejected_items.append(
                    {
                        "item_id": slug,
                        "item_url": item_url,
                        "title": str(search_row.get("title") or ""),
                        "reasons": [f"item_api_error:{type(error).__name__}:{error}"],
                    }
                )
            task.update(
                index,
                message=slug,
                metrics={"accepted_items": len(accepted_items), "candidate_pages": len(page_rows)},
            )

    page_rows.sort(key=lambda row: (row["item_id"], row["page_number"]))
    if args.maximum_pages:
        page_rows = page_rows[: args.maximum_pages]
    write_jsonl(args.out / "accepted_items.jsonl", accepted_items)
    write_jsonl(args.out / "rejected_items.jsonl", rejected_items)
    write_jsonl(args.out / "verified_remote_pages.jsonl", page_rows)
    summary = {
        "schema_version": "loc_live_no_official_transcript_inventory.v1",
        "collection_url": args.collection_url,
        "collection_items_audited": len(items),
        "accepted_items": len(accepted_items),
        "rejected_items": len(rejected_items),
        "verified_candidate_pages": len(page_rows),
        "rejection_reason_counts": dict(reasons),
        "verification_policy": {
            "scope": "entire LOC item",
            "source": "live loc.gov item JSON API",
            "fail_closed": True,
            "requires_image_only_online_format": True,
            "rejects_fulltext_transcript_word_coordinate_and_text_service_signals": True,
        },
        "predictions_are_unreviewed_and_not_ground_truth": True,
    }
    (args.out / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return args.out / "verified_remote_pages.jsonl"


def download_pages(args: argparse.Namespace, remote_manifest: Path) -> Path:
    rows = [json.loads(line) for line in remote_manifest.read_text(encoding="utf-8").splitlines() if line]
    downloaded: list[dict] = []
    failures: list[dict] = []
    with ProgressTask(
        "Download LOC pages verified without official transcripts",
        total=len(rows),
        unit="pages",
        task_id=f"loc-no-transcript-download-{args.out.name}",
        output_dir=args.out,
    ) as task:
        for index, row in enumerate(rows, start=1):
            destination = args.out / "images" / row["item_id"] / f"page_{row['page_number']:04d}.jpg"
            try:
                if not destination.is_file() or destination.stat().st_size < 1024:
                    data = request_bytes(row["image_url"], minimum_delay=args.download_delay)
                    if len(data) < 1024 or not data.startswith(b"\xff\xd8"):
                        raise ValueError(f"download is not a valid JPEG ({len(data)} bytes)")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(".jpg.part")
                    temporary.write_bytes(data)
                    os.replace(temporary, destination)
                downloaded.append({**row, "image": str(destination.resolve())})
            except Exception as error:
                failures.append({**row, "download_error": f"{type(error).__name__}: {error}"})
            task.update(
                index,
                message=row["page_id"],
                metrics={"downloaded": len(downloaded), "failed": len(failures)},
            )
    manifest = args.out / "inference_manifest.jsonl"
    write_jsonl(manifest, downloaded)
    write_jsonl(args.out / "download_failures.jsonl", failures)
    (args.out / "download_summary.json").write_text(
        json.dumps(
            {
                "requested": len(rows),
                "downloaded": len(downloaded),
                "failed": len(failures),
                "manifest": str(manifest.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-url", default=DEFAULT_COLLECTION_URL)
    parser.add_argument(
        "--item-url",
        action="append",
        default=[],
        help="Audit only this exact LOC item URL; may be repeated.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--maximum-items", type=int, default=0)
    parser.add_argument("--maximum-pages", type=int, default=0)
    parser.add_argument("--target-image-pixels", type=int, default=2_200_000)
    parser.add_argument("--download-delay", type=float, default=0.15)
    parser.add_argument("--max-pixels", type=int, default=1605632)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.maximum_items < 0 or args.maximum_pages < 0:
        parser.error("maximum counts must be zero (unlimited) or positive")
    remote_manifest = discover(args)
    if args.inventory_only:
        return
    local_manifest = download_pages(args, remote_manifest)
    if args.download_only:
        return
    if not args.adapter.is_dir():
        parser.error(f"Epoch 19 adapter not found: {args.adapter}")
    command = [
        sys.executable,
        str(SCRIPT_ROOT / "evaluate_churro_fullpage_qlora.py"),
        "--manifest",
        str(local_manifest),
        "--adapter",
        str(args.adapter.resolve()),
        "--output",
        str(args.out / "epoch19_predictions"),
        "--max-pixels",
        str(args.max_pixels),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-incomplete-retries",
        "0",
    ]
    subprocess.run(command, cwd=SCRIPT_ROOT, check=True)


if __name__ == "__main__":
    main()
