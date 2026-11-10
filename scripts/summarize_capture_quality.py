#!/usr/bin/env python3
"""Summarize retained capture quality and runtime metrics without reading payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = (
    "capture_started_at_ns", "capture_stopped_at_ns", "gap_count",
    "disconnect_count", "stop_reason", "closed", "trainable",
)
RUNTIME = (
    "process_user_cpu_seconds", "process_system_cpu_seconds", "process_rss_kib",
    "receiver_queue_limit", "receiver_queue_high_water", "max_interframe_delay_ns",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--stream", default="")
    parser.add_argument("--symbol", default="")
    args = parser.parse_args()
    if not args.archive_root.is_dir():
        raise SystemExit(f"archive root is not a directory: {args.archive_root}")

    entries = []
    for metadata_path in sorted(args.archive_root.rglob("*.jsonl.zst.meta.json")):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if metadata.get("schema") != "td_market_data_archive_metadata_v1":
            continue
        relative = metadata_path.relative_to(args.archive_root)
        if len(relative.parts) != 3:
            continue
        stream, symbol = relative.parts[0], relative.parts[1]
        if args.stream and stream != args.stream:
            continue
        if args.symbol and symbol != args.symbol.upper():
            continue
        entries.append((metadata_path, metadata))

    captures = [(path, metadata) for path, metadata in entries
                if all(key in metadata for key in REQUIRED)]

    runtime_complete = [metadata for _, metadata in captures
                        if all(key in metadata for key in RUNTIME)]
    summary = {
        "archive_root": str(args.archive_root),
        "stream": args.stream or None,
        "symbol": args.symbol.upper() or None,
        "capture_count": len(entries),
        "quality_record_count": len(captures),
        "quality_missing_count": len(entries) - len(captures),
        "trainable_count": sum(bool(row["trainable"]) for _, row in captures),
        "gap_capture_count": sum(int(row["gap_count"]) > 0 for _, row in captures),
        "total_gap_count": sum(int(row["gap_count"]) for _, row in captures),
        "total_disconnect_count": sum(int(row["disconnect_count"]) for _, row in captures),
        "runtime_metrics_complete_count": len(runtime_complete),
        "max_process_rss_kib": max((int(row["process_rss_kib"]) for row in runtime_complete), default=None),
        "max_receiver_queue_high_water": max(
            (int(row["receiver_queue_high_water"]) for row in runtime_complete), default=None
        ),
        "max_interframe_delay_ns": max(
            (int(row["max_interframe_delay_ns"]) for row in runtime_complete), default=None
        ),
        "total_process_user_cpu_seconds": round(
            sum(float(row["process_user_cpu_seconds"]) for row in runtime_complete), 6
        ),
        "total_process_system_cpu_seconds": round(
            sum(float(row["process_system_cpu_seconds"]) for row in runtime_complete), 6
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
