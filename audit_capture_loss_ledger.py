#!/usr/bin/env python3
"""Report retained VPS quota-loss records intersecting a research window."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path


LOSS_SCHEMA = "td_market_data_relay_early_discard_v1"
MAX_LEDGER_BYTES = 16 * 1024 * 1024
STREAMS = {
    "spot-depth", "spot-trades", "spot-klines-1m", "usdm-depth",
    "usdm-funding-mark", "usdm-finalized-funding", "usdm-aggtrades", "usdm-klines-1m",
}


class CoverageRejected(ValueError):
    pass


def parse_time(value: str) -> int:
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if stamp.tzinfo != dt.timezone.utc:
        raise argparse.ArgumentTypeError("timestamp must use UTC")
    return int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1_000


def verify(ledger: Path, stream: str, symbol: str, start_ns: int, end_ns: int) -> int:
    if not ledger.is_absolute() or ledger.is_symlink() or not ledger.is_file():
        raise CoverageRejected("loss ledger must be an absolute regular non-symlink file")
    if ledger.stat().st_size > MAX_LEDGER_BYTES:
        raise CoverageRejected("loss ledger exceeds the 16 MiB safety bound")
    if start_ns >= end_ns:
        raise CoverageRejected("window must have positive duration")
    if not symbol.isascii() or re.fullmatch(r"[A-Z0-9]{1,32}", symbol) is None:
        raise CoverageRejected("symbol must be uppercase alphanumeric")
    if stream not in STREAMS:
        raise CoverageRejected("unsupported collector stream")
    overlaps = []
    with ledger.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if len(raw) > 16 * 1024 or not raw.endswith(b"\n"):
                raise CoverageRejected(f"ledger line {line_number} is truncated or oversized")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CoverageRejected(f"ledger line {line_number} is invalid JSON") from error
            if not isinstance(row, dict) or row.get("schema") != LOSS_SCHEMA or row.get("version") != 1:
                raise CoverageRejected(f"ledger line {line_number} has an unsupported schema")
            capture_id = row.get("capture_id")
            row_stream, row_symbol = row.get("stream"), row.get("symbol")
            row_path = row.get("path")
            if (not isinstance(capture_id, str) or re.fullmatch(r"[0-9a-f]{32}", capture_id) is None
                    or row_stream not in STREAMS
                    or not isinstance(row_symbol, str) or re.fullmatch(r"[A-Z0-9]{1,32}", row_symbol) is None
                    or not isinstance(row_path, str)
                    or Path(row_path).parts[:2] != (row_stream, row_symbol)
                    or Path(row_path).name.split(".")[0].rsplit("-", 1)[-1] != capture_id
                    or row.get("reason") not in {"archive_quota", "hot_storage_quota"}):
                raise CoverageRejected(f"ledger line {line_number} has malformed capture identity")
            if row.get("stream") != stream or row.get("symbol") != symbol:
                continue
            capture_start = row.get("capture_started_at_ns")
            capture_stop = row.get("capture_stopped_at_ns")
            if (not isinstance(capture_start, int) or isinstance(capture_start, bool)
                    or not isinstance(capture_stop, int) or isinstance(capture_stop, bool)
                    or capture_start < 0 or capture_stop < capture_start):
                raise CoverageRejected(f"matching loss line {line_number} lacks reliable capture bounds")
            if capture_start < end_ns and capture_stop >= start_ns:
                overlaps.append(row)
    if overlaps:
        details = ", ".join(f"{item['capture_id']}({item['reason']})" for item in overlaps[:20])
        raise CoverageRejected(f"window intersects {len(overlaps)} remotely discarded capture(s): {details}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-ledger", required=True, type=Path)
    parser.add_argument("--stream", required=True, choices=sorted(STREAMS))
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    args = parser.parse_args()
    try:
        count = verify(args.loss_ledger, args.stream, args.symbol, args.start, args.end)
    except (CoverageRejected, OSError) as error:
        raise SystemExit(f"capture-loss ledger audit rejected: {error}") from error
    print(f"no recorded quota losses overlap: stream={args.stream} symbol={args.symbol} "
          f"window=[{args.start},{args.end})")


if __name__ == "__main__":
    main()
