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
from collector import write_funding_cursor
import zstandard


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


def usage_snapshot() -> tuple[float, float, int]:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_kib = 0
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss_kib = int(line.split()[1])
                break
    except (OSError, ValueError):
        pass
    return usage.ru_utime, usage.ru_stime, rss_kib


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
        user_before, system_before, rss_before = usage_snapshot()
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
        combined = CaptureFile(temporary, "usdm-aggtrades", "BTCUSDT", 1 << 30,
                               "profile", 5_000)
        combined_started = time.perf_counter()
        for index in range(args.frames):
            parsed = json.loads(RAW)
            combined.frame(1794300000000000000 + index, 1000000000 + index, RAW,
                           event_time_ms=parsed["E"], trade_id=parsed["a"],
                           first_trade_id=parsed["a"], last_trade_id=parsed["a"],
                           accepted_for_sequence=True)
        combined_seconds = time.perf_counter() - combined_started
        combined.close("profile_complete")
        compression_path = temporary / "profile.jsonl.zst"
        compression_started = time.perf_counter()
        with capture.path.open("rb") as source, compression_path.open("xb") as output:
            with zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(
                    output, closefd=False) as writer:
                shutil.copyfileobj(source, writer, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        compression_seconds = time.perf_counter() - compression_started
        state_updates = min(args.frames, 1_000)
        state_path = temporary / "funding-cursor.json"
        state_started = time.perf_counter()
        for index in range(state_updates):
            write_funding_cursor(state_path, 1_700_000_000_000, 1_700_000_000_000 + index)
        state_seconds = time.perf_counter() - state_started
        user_after, system_after, rss_after = usage_snapshot()
        result = {
            "frames": args.frames,
            "raw_frame_bytes": len(raw_bytes),
            "parse_frames_per_second": round(args.frames / parse_seconds, 2),
            "sha256_frames_per_second": round(args.frames / hash_seconds, 2),
            "compact_json_frames_per_second": round(args.frames / json_seconds, 2),
            "capture_write_frames_per_second": round(args.frames / write_seconds, 2),
            "parse_and_capture_write_frames_per_second": round(args.frames / combined_seconds, 2),
            "zstd_compress_frames_per_second": round(args.frames / compression_seconds, 2),
            "zstd_compress_mib_per_second": round(
                output_bytes / 1024 / 1024 / compression_seconds, 2
            ),
            "zstd_output_bytes": compression_path.stat().st_size,
            "funding_cursor_updates": state_updates,
            "funding_cursor_updates_per_second": round(state_updates / state_seconds, 2),
            "capture_output_bytes": output_bytes,
            "capture_output_mib": round(output_bytes / 1024 / 1024, 3),
            "process_user_cpu_seconds": round(user_after - user_before, 6),
            "process_system_cpu_seconds": round(system_after - system_before, 6),
            "process_rss_kib": rss_after,
            "process_rss_delta_kib": max(0, rss_after - rss_before),
            "work_dir": str(args.work_dir),
            "note": "Synthetic local baseline; run on the VPS during normal operation before deciding on Rust.",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
