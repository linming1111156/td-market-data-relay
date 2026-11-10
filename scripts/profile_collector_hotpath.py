#!/usr/bin/env python3
"""Measure the collector's local parse/hash/JSON/write hot path without network access."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# Allow `python3 scripts/profile_collector_hotpath.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from collector import CaptureFile, compact_json, sha256_bytes


RAW = json.dumps({
    "e": "aggTrade",
    "E": 1794300000000,
    "a": 123456789,
    "p": "100000.10",
    "q": "0.001000",
    "f": 123456700,
    "l": 123456700,
    "T": 1794300000000,
    "m": False,
    "M": True,
    "s": "BTCUSDT",
}, separators=(",", ":"))


def elapsed(callable_, count: int) -> tuple[float, object]:
    started = time.perf_counter()
    value = callable_()
    return time.perf_counter() - started, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10_000)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("TD_PROFILE_WORK_DIR", "/var/lib/td-market-data/state")),
    )
    args = parser.parse_args()
    if args.frames < 100 or args.frames > 1_000_000:
        raise SystemExit("--frames must be between 100 and 1000000")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="collector-profile-", dir=args.work_dir))
    try:
        raw_bytes = RAW.encode()
        parse_seconds, _ = elapsed(lambda: [json.loads(RAW) for _ in range(args.frames)], args.frames)
        hash_seconds, _ = elapsed(lambda: [sha256_bytes(raw_bytes) for _ in range(args.frames)], args.frames)
        record = {
            "schema": "td_public_market_capture_record", "version": 1,
            "record_type": "market_frame", "received_at_unix_ns": 1794300000000000000,
            "receive_monotonic_ns": 1000000000, "raw_frame": RAW,
            "raw_frame_sha256": sha256_bytes(raw_bytes), "session_id": "profile",
            "event_time_ms": 1794300000000, "trade_id": 123456789,
            "accepted_for_sequence": True,
        }
        json_seconds, _ = elapsed(lambda: [compact_json(record) for _ in range(args.frames)], args.frames)
        capture = CaptureFile(temporary, "usdm-aggtrades", "BTCUSDT", 1 << 30,
                              "profile", 5_000)
        write_started = time.perf_counter()
        for index in range(args.frames):
            capture.frame(1794300000000000000 + index, 1000000000 + index, RAW,
                          event_time_ms=1794300000000, trade_id=123456789 + index,
                          first_trade_id=123456789 + index,
                          last_trade_id=123456789 + index,
                          accepted_for_sequence=True)
        capture.close("profile_complete")
        write_seconds = time.perf_counter() - write_started
        output_bytes = capture.path.stat().st_size
        result = {
            "frames": args.frames,
            "raw_frame_bytes": len(raw_bytes),
            "parse_frames_per_second": round(args.frames / parse_seconds, 2),
            "sha256_frames_per_second": round(args.frames / hash_seconds, 2),
            "compact_json_frames_per_second": round(args.frames / json_seconds, 2),
            "capture_write_frames_per_second": round(args.frames / write_seconds, 2),
            "capture_output_bytes": output_bytes,
            "capture_output_mib": round(output_bytes / 1024 / 1024, 3),
            "work_dir": str(args.work_dir),
            "note": "Synthetic local baseline; run on the VPS during normal operation before deciding on Rust.",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
