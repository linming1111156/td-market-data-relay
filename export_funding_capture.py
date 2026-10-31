#!/usr/bin/env python3
"""Validate one clean Python VPS Mark Price capture and export Rust-compatible JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
SOURCE = "binance_usdm_mark_price_websocket"


class InvalidCapture(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_line(raw: bytes, line_number: int) -> dict[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > 2 * 1024 * 1024:
        raise InvalidCapture(f"line {line_number}: truncated or oversized JSONL record")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise InvalidCapture(f"line {line_number}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InvalidCapture(f"line {line_number}: malformed JSON") from error
    if not isinstance(value, dict):
        raise InvalidCapture(f"line {line_number}: record must be a JSON object")
    if value.get("schema") != "td_public_market_capture_record" or value.get("version") != 1:
        raise InvalidCapture(f"line {line_number}: unsupported capture schema")
    return value


def require_int(record: dict[str, Any], key: str, line: int) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InvalidCapture(f"line {line}: {key} must be a nonnegative integer")
    return value


def require_finite(value: Any, field: str, line: int, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise InvalidCapture(f"line {line}: invalid {field}") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise InvalidCapture(f"line {line}: invalid {field}")
    return number


def source_url(symbol: str) -> str:
    return f"wss://fstream.binance.com/market/ws/{symbol.lower()}@markPrice@1s"


def export_capture(source: Path, output: Path, symbol: str, max_gap_ms: int | None) -> tuple[str, int]:
    source = source.resolve(strict=True)
    if source.stat().st_size > MAX_CAPTURE_BYTES:
        raise InvalidCapture("source capture exceeds the 256 MiB validation bound")
    source_hash_before = sha256_file(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    converter_hash = sha256_file(Path(__file__).resolve())
    fd, temp_name = tempfile.mkstemp(prefix=".funding-export-", suffix=".jsonl", dir=output.parent)
    temp = Path(temp_name)
    count = 0
    previous_event_ms: int | None = None
    previous_receive_ns: int | None = None
    previous_mono_ns: int | None = None
    capture_gap_ms: int | None = None
    capture_start_wall_ns: int | None = None
    capture_start_mono_ns: int | None = None
    connect_wall_ns: int | None = None
    last_recorded_wall_ns: int | None = None
    input_bytes_before_stop: int | None = None
    started: dict[str, Any] | None = None
    session_id: str | None = None
    connected = False
    disconnected = False
    stopped = False
    output_bytes = 0

    def write_record(stream: Any, record: dict[str, Any]) -> None:
        nonlocal output_bytes
        encoded = (json.dumps(record, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
        stream.write(encoded)
        output_bytes += len(encoded)

    try:
        with os.fdopen(fd, "wb", buffering=0) as target, source.open("rb") as reader:
            line_number = 0
            input_bytes = 0
            for raw in reader:
                line_number += 1
                input_bytes += len(raw)
                record = parse_line(raw, line_number)
                kind = record.get("record_type")
                if line_number == 1:
                    if kind != "capture_started" or record.get("stream") != "usdm-funding-mark":
                        raise InvalidCapture("capture must start with a USD-M markPrice header")
                    if record.get("symbol") != symbol or record.get("source_host_id") is None:
                        raise InvalidCapture("capture symbol/source identity mismatch")
                    started = record
                    capture_start_wall_ns = require_int(record, "recorded_at_unix_ns", line_number)
                    capture_start_mono_ns = require_int(record, "recorded_at_monotonic_ns", line_number)
                    last_recorded_wall_ns = capture_start_wall_ns
                    capture_code_hash = record.get("collector_executable_sha256")
                    capture_code_path = record.get("collector_executable_path")
                    capture_max_gap = record.get("max_gap_ms")
                    if (not isinstance(capture_code_hash, str) or len(capture_code_hash) != 64
                            or not isinstance(capture_code_path, str)
                            or not isinstance(capture_max_gap, int) or capture_max_gap <= 0):
                        raise InvalidCapture("collector code fingerprint or max gap policy is missing")
                    if any(c not in "0123456789abcdef" for c in capture_code_hash):
                        raise InvalidCapture("collector code fingerprint is malformed")
                    if max_gap_ms is not None and max_gap_ms != capture_max_gap:
                        raise InvalidCapture("requested max gap differs from capture policy")
                    if record.get("source_url") != source_url(symbol):
                        raise InvalidCapture("capture header source URL mismatch")
                    if not isinstance(record.get("source_revision"), str) or not record["source_revision"]:
                        raise InvalidCapture("capture source revision is missing")
                    capture_gap_ms = capture_max_gap
                    write_record(target, {
                        "record_type": "capture_started",
                        "recorded_at_unix_ns": require_int(record, "recorded_at_unix_ns", line_number),
                        "process_id": os.getpid(),
                        "collector_executable_path": capture_code_path,
                        "collector_executable_sha256": capture_code_hash,
                        "converter_executable_path": str(Path(__file__).resolve()),
                        "converter_executable_sha256": converter_hash,
                        "symbol": symbol,
                        "source_url": source_url(symbol),
                        "max_gap_ms": capture_max_gap,
                        "max_output_bytes": MAX_CAPTURE_BYTES,
                        "source_capture_sha256": source_hash_before,
                        "source_capture_code_revision": record.get("source_revision", "unknown"),
                    })
                elif kind == "session_connected":
                    if not started or connected or disconnected or stopped:
                        raise InvalidCapture(f"line {line_number}: invalid/reconnected session")
                    session_id = record.get("session_id")
                    if not isinstance(session_id, str) or not session_id:
                        raise InvalidCapture(f"line {line_number}: session ID missing")
                    if record.get("url") != source_url(symbol):
                        raise InvalidCapture(f"line {line_number}: connected WebSocket URL mismatch")
                    connect_wall_ns = require_int(record, "recorded_at_unix_ns", line_number)
                    if connect_wall_ns < (capture_start_wall_ns or 0):
                        raise InvalidCapture(f"line {line_number}: connection predates capture start")
                    last_recorded_wall_ns = connect_wall_ns
                    connected = True
                    write_record(target, {
                        "record_type": "session_connected",
                        "recorded_at_unix_ns": require_int(record, "recorded_at_unix_ns", line_number),
                        "session_id": session_id,
                        "attempt": 1,
                        "source_url": source_url(symbol),
                    })
                elif kind == "market_frame":
                    if not connected or disconnected or stopped or record.get("session_id") != session_id:
                        raise InvalidCapture(f"line {line_number}: observation outside its session")
                    raw_frame = record.get("raw_frame")
                    if not isinstance(raw_frame, str):
                        raise InvalidCapture(f"line {line_number}: raw frame missing")
                    payload_hash = hashlib.sha256(raw_frame.encode()).hexdigest()
                    if record.get("raw_frame_sha256") != payload_hash:
                        raise InvalidCapture(f"line {line_number}: raw frame hash mismatch")
                    try:
                        frame = json.loads(raw_frame)
                    except json.JSONDecodeError as error:
                        raise InvalidCapture(f"line {line_number}: invalid exchange frame JSON") from error
                    event_ms = record.get("event_time_ms")
                    if (not isinstance(frame, dict) or frame.get("e") != "markPriceUpdate" or frame.get("s") != symbol
                            or frame.get("E") != event_ms):
                        raise InvalidCapture(f"line {line_number}: exchange event identity mismatch")
                    received_ns = require_int(record, "received_at_unix_ns", line_number)
                    received_mono_ns = require_int(record, "receive_monotonic_ns", line_number)
                    if not isinstance(event_ms, int) or event_ms < 0:
                        raise InvalidCapture(f"line {line_number}: invalid exchange event time")
                    event_ns = event_ms * 1_000_000
                    if event_ns > received_ns + 2_000_000_000:
                        raise InvalidCapture(f"line {line_number}: exchange event is ahead of receive time")
                    if received_ns < (connect_wall_ns or 0):
                        raise InvalidCapture(f"line {line_number}: observation predates connection")
                    if (previous_event_ms is not None and event_ms <= previous_event_ms
                            or previous_receive_ns is not None and received_ns <= previous_receive_ns
                            or previous_mono_ns is not None and received_mono_ns <= previous_mono_ns):
                        raise InvalidCapture(f"line {line_number}: timestamps are not strictly increasing")
                    if previous_receive_ns is not None and previous_mono_ns is not None:
                        wall_delta = received_ns - previous_receive_ns
                        mono_delta = received_mono_ns - previous_mono_ns
                        if mono_delta > (capture_gap_ms or 0) * 1_000_000:
                            raise InvalidCapture(f"line {line_number}: receive-time gap exceeds capture policy")
                        if abs(wall_delta - mono_delta) > 2_000_000_000:
                            raise InvalidCapture(f"line {line_number}: wall/monotonic clock discontinuity")
                    next_funding_ms = frame.get("T")
                    if not isinstance(next_funding_ms, int) or next_funding_ms <= 0:
                        raise InvalidCapture(f"line {line_number}: invalid next funding time")
                    mark = require_finite(frame.get("p"), "mark price", line_number, positive=True)
                    index = require_finite(frame.get("i"), "index price", line_number, positive=True)
                    rate = require_finite(frame.get("r"), "funding rate", line_number)
                    settle = frame.get("P")
                    if settle in (None, ""):
                        estimated_settle = None
                    else:
                        estimated_settle = require_finite(settle, "estimated settle price", line_number, positive=True)
                    write_record(target, {
                        "record_type": "mark_price_funding_observation",
                        "session_id": session_id,
                        "sequence": count + 1,
                        "source": SOURCE,
                        "symbol": symbol,
                        "exchange_event_time_ms": event_ms,
                        "received_at_unix_ns": received_ns,
                        "received_monotonic_ns": received_mono_ns,
                        "next_funding_time_ms": next_funding_ms,
                        "mark_price": mark,
                        "index_price": index,
                        "estimated_settle_price": estimated_settle,
                        "funding_rate": rate,
                        "payload_sha256": payload_hash,
                        "raw_frame": raw_frame,
                    })
                    count += 1
                    previous_event_ms = event_ms
                    previous_receive_ns = received_ns
                    previous_mono_ns = received_mono_ns
                    last_recorded_wall_ns = received_ns
                elif kind == "session_disconnected":
                    if not connected or disconnected or stopped or record.get("session_id") != session_id:
                        raise InvalidCapture(f"line {line_number}: invalid session end")
                    if record.get("reason") not in {"hourly_rotation", "local_shutdown"}:
                        raise InvalidCapture(f"line {line_number}: capture did not end cleanly")
                    disconnected_at_ns = require_int(record, "recorded_at_unix_ns", line_number)
                    if disconnected_at_ns < (last_recorded_wall_ns or 0):
                        raise InvalidCapture(f"line {line_number}: disconnect predates prior record")
                    last_recorded_wall_ns = disconnected_at_ns
                    disconnected = True
                    write_record(target, {
                        "record_type": "session_disconnected",
                        "recorded_at_unix_ns": require_int(record, "recorded_at_unix_ns", line_number),
                        "session_id": session_id,
                        "reason": "local_shutdown",
                    })
                elif kind == "capture_stopped":
                    input_bytes_before_stop = input_bytes - len(raw)
                    if (not disconnected or stopped or record.get("reason") not in {"duration_elapsed", "sigterm"}
                            or record.get("observation_count") != count
                            or record.get("bytes_before_stop_record") != input_bytes_before_stop):
                        raise InvalidCapture(f"line {line_number}: capture stop summary/integrity mismatch")
                    stopped = True
                    stopped_wall_ns = require_int(record, "recorded_at_unix_ns", line_number)
                    if stopped_wall_ns < (last_recorded_wall_ns or 0):
                        raise InvalidCapture(f"line {line_number}: capture stop predates prior record")
                    final_record = {
                        "record_type": "capture_stopped",
                        "recorded_at_unix_ns": require_int(record, "recorded_at_unix_ns", line_number),
                        "elapsed_monotonic_ns": max(0, require_int(record, "recorded_at_monotonic_ns", line_number)
                                                     - (capture_start_mono_ns or 0)),
                        "observation_count": count,
                        "bytes_written": output_bytes,
                    }
                    write_record(target, final_record)
                elif kind in {"gap_detected", "clock_discontinuity"}:
                    raise InvalidCapture(f"line {line_number}: capture contains {kind}")
                else:
                    raise InvalidCapture(f"line {line_number}: unsupported capture record type {kind!r}")
            target.flush()
            os.fsync(target.fileno())
        if not (started and connected and disconnected and stopped and count > 0):
            raise InvalidCapture("capture is incomplete, empty, or has no single clean session")
        if sha256_file(source) != source_hash_before:
            raise InvalidCapture("source capture changed while it was being converted")
        output_hash = sha256_file(temp)
        os.link(temp, output)
        return output_hash, count
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--max-gap-ms", type=int, help="must match the max-gap policy recorded by the collector")
    args = parser.parse_args()
    symbol = args.symbol.upper()
    if not symbol.isascii() or not symbol.isalnum() or (args.max_gap_ms is not None and args.max_gap_ms <= 0):
        raise SystemExit("symbol or max-gap value is invalid")
    try:
        digest, observations = export_capture(args.capture, args.output, symbol, args.max_gap_ms)
    except (InvalidCapture, OSError) as error:
        raise SystemExit(f"Funding capture rejected: {error}") from error
    print(f"Funding capture accepted: observations={observations} sha256={digest} output={args.output}")


if __name__ == "__main__":
    main()
