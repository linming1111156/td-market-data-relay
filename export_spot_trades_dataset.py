#!/usr/bin/env python3
"""Publish a source-bound, point-in-time Spot public-trades Dataset."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from export_spot_depth_dataset import canonical, file_sha256, source_revision


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_DATASET_BYTES = 512 * 1024 * 1024


class SpotTradesRejected(ValueError):
    pass


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None or renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def parse_time(value: str) -> int:
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if stamp.tzinfo != dt.timezone.utc:
        raise argparse.ArgumentTypeError("timestamp must use UTC")
    ns = int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1_000
    if ns < 0 or ns % 1_000_000:
        raise argparse.ArgumentTypeError("timestamp must be nonnegative and millisecond aligned")
    return ns


def decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, str):
        raise SpotTradesRejected(f"{field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise SpotTradesRejected(f"invalid {field}") from error
    if not parsed.is_finite() or parsed <= 0:
        raise SpotTradesRejected(f"{field} must be finite and positive")
    return parsed


def uint(value: Any, field: str, line: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise SpotTradesRejected(f"line {line}: {field} must be a nonnegative integer")
    return value


def publish(capture: Path, output: Path, symbol: str, start_ns: int, end_ns: int) -> dict[str, Any]:
    capture = capture.resolve(strict=True)
    if not output.is_absolute() or start_ns >= end_ns:
        raise SpotTradesRejected("output must be absolute and the requested window positive")
    if capture.stat().st_size > MAX_CAPTURE_BYTES:
        raise SpotTradesRejected("capture exceeds 256 MiB")
    if not symbol.isascii() or not symbol.isalnum():
        raise SpotTradesRejected("invalid Spot symbol")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if output.exists():
        raise SpotTradesRejected("Dataset output already exists")
    repository = Path(__file__).resolve().parent
    if output == repository or repository in output.parents:
        raise SpotTradesRejected("Dataset output must be outside the source repository")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        with (stage / "events.jsonl").open("xb", buffering=0) as events_writer:
            return _publish_staged(capture, output, symbol, start_ns, end_ns, parent, stage, events_writer)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _publish_staged(capture: Path, output: Path, symbol: str, start_ns: int,
                    end_ns: int, parent: Path, stage: Path, events_writer: Any) -> dict[str, Any]:

    capture_hash = file_sha256(capture)
    header: dict[str, Any] | None = None
    footer: dict[str, Any] | None = None
    session: str | None = None
    connected = disconnected = False
    start_ns_recorded = stop_ns_recorded = None
    start_mono_ns = stop_mono_ns = None
    connect_ns = None
    max_gap_ms = 0
    previous_trade_id: int | None = None
    previous_event_ns: int | None = None
    previous_trade_ns: int | None = None
    previous_receive_ns: int | None = None
    previous_receive_mono_ns: int | None = None
    observation_count = 0
    event_count = 0
    event_bytes = 0
    events_hash = hashlib.sha256()
    logical_hash = hashlib.sha256(b"td-market-data-spot-trades-dataset-v1\0" + canonical({"start_ns": start_ns, "end_ns": end_ns}))
    has_pre_window = False
    has_post_window = False
    first_event: dict[str, Any] | None = None
    last_event: dict[str, Any] | None = None
    capture_bytes_seen = 0

    with capture.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            bytes_before_record = capture_bytes_seen
            capture_bytes_seen += len(raw)
            if not raw.endswith(b"\n") or len(raw) > 4 * 1024 * 1024:
                raise SpotTradesRejected(f"line {line_number}: truncated or oversized JSONL")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SpotTradesRejected(f"line {line_number}: malformed JSON") from error
            if not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record" or row.get("version") != 1:
                raise SpotTradesRejected(f"line {line_number}: unsupported capture record")
            kind = row.get("record_type")
            if line_number == 1:
                if kind != "capture_started" or row.get("stream") != "spot-trades" or row.get("symbol") != symbol:
                    raise SpotTradesRejected("capture header does not identify this Spot trades stream")
                header = row
                start_ns_recorded = uint(row.get("recorded_at_unix_ns"), "capture start time", line_number)
                start_mono_ns = uint(row.get("recorded_at_monotonic_ns"), "capture monotonic start", line_number)
                max_gap_ms = uint(row.get("max_gap_ms"), "max gap policy", line_number)
                if max_gap_ms <= 0:
                    raise SpotTradesRejected("max gap policy must be positive")
                code_hash = row.get("collector_executable_sha256")
                if (not isinstance(code_hash, str) or len(code_hash) != 64
                        or any(char not in "0123456789abcdef" for char in code_hash)
                        or not isinstance(row.get("source_revision"), str) or not row["source_revision"]):
                    raise SpotTradesRejected("collector code hash or source revision missing")
            elif kind == "session_connected":
                if connected or disconnected:
                    raise SpotTradesRejected(f"line {line_number}: reconnects are not admissible in one Dataset")
                session = row.get("session_id")
                expected_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
                connect_ns = uint(row.get("recorded_at_unix_ns"), "session connect time", line_number)
                if not isinstance(session, str) or not session or row.get("url") != expected_url:
                    raise SpotTradesRejected(f"line {line_number}: invalid Spot trade session identity")
                connected = True
            elif kind == "market_frame":
                observation_count += 1
                if not connected or disconnected or row.get("session_id") != session:
                    raise SpotTradesRejected(f"line {line_number}: trade frame outside connected session")
                raw_frame = row.get("raw_frame")
                if not isinstance(raw_frame, str) or hashlib.sha256(raw_frame.encode()).hexdigest() != row.get("raw_frame_sha256"):
                    raise SpotTradesRejected(f"line {line_number}: raw trade frame hash mismatch")
                try:
                    frame = json.loads(raw_frame)
                except json.JSONDecodeError as error:
                    raise SpotTradesRejected(f"line {line_number}: invalid raw trade frame") from error
                trade_id = uint(frame.get("t"), "trade ID", line_number)
                event_ms = uint(frame.get("E"), "exchange event time", line_number)
                trade_ms = uint(frame.get("T"), "trade time", line_number)
                receive_ns = uint(row.get("received_at_unix_ns"), "receive time", line_number)
                receive_mono_ns = uint(row.get("receive_monotonic_ns"), "monotonic receive time", line_number)
                if (frame.get("e") != "trade" or frame.get("s") != symbol
                        or event_ms != row.get("event_time_ms") or trade_ms != row.get("trade_time_ms")
                        or trade_id != row.get("trade_id")):
                    raise SpotTradesRejected(f"line {line_number}: raw frame fields disagree with parsed capture")
                if not isinstance(frame.get("m"), bool):
                    raise SpotTradesRejected(f"line {line_number}: buyer-maker flag is invalid")
                if not isinstance(frame.get("M"), bool) or row.get("accepted_for_sequence") is not True:
                    raise SpotTradesRejected(f"line {line_number}: trade frame is not sequence-accepted")
                if event_ms < trade_ms or receive_ns < max(event_ms, trade_ms) * 1_000_000:
                    raise SpotTradesRejected(f"line {line_number}: event/receive time order is invalid")
                event_ns, trade_ns = event_ms * 1_000_000, trade_ms * 1_000_000
                if previous_trade_id is not None and trade_id != previous_trade_id + 1:
                    raise SpotTradesRejected(f"line {line_number}: Spot trade ID gap or duplicate")
                if ((previous_event_ns is not None and event_ns < previous_event_ns)
                        or (previous_trade_ns is not None and trade_ns < previous_trade_ns)
                        or (previous_receive_ns is not None and receive_ns <= previous_receive_ns)
                        or (previous_receive_mono_ns is not None and receive_mono_ns <= previous_receive_mono_ns)):
                    raise SpotTradesRejected(f"line {line_number}: event, trade, or receive time regressed")
                if previous_receive_ns is not None:
                    wall_gap = receive_ns - previous_receive_ns
                    mono_gap = receive_mono_ns - (previous_receive_mono_ns or 0)
                    if wall_gap > max_gap_ms * 1_000_000 or abs(wall_gap - mono_gap) > 2_000_000_000:
                        raise SpotTradesRejected(f"line {line_number}: receive gap/clock discontinuity exceeds policy")
                price, quantity = decimal(frame.get("p"), "trade price"), decimal(frame.get("q"), "trade quantity")
                available_ns = max(event_ns, trade_ns, receive_ns)
                event = {"schema": "td_market_data_spot_trade_event", "version": 1,
                         "exchange": "BINANCE", "market": "SPOT", "symbol": symbol,
                         "session_id": session, "trade_id": trade_id, "event_time_ns": event_ns,
                         "trade_time_ns": trade_ns, "receive_time_ns": receive_ns,
                         "available_at_ns": available_ns,
                         "price": format(price, "f"), "quantity": format(quantity, "f"),
                         "quote_quantity": format(price * quantity, "f"),
                         "buyer_is_maker": frame["m"],
                         "aggressor_side": "SELL" if frame["m"] else "BUY",
                         "payload_sha256": row["raw_frame_sha256"]}
                if available_ns < start_ns:
                    has_pre_window = True
                elif available_ns >= end_ns:
                    has_post_window = True
                else:
                    if first_event is None:
                        first_event = event
                    last_event = event
                    encoded = canonical(event) + b"\n"
                    event_bytes += len(encoded)
                    if event_bytes > MAX_DATASET_BYTES:
                        raise SpotTradesRejected("normalized Dataset exceeds 512 MiB")
                    events_writer.write(encoded)
                    events_hash.update(encoded)
                    logical_hash.update(encoded)
                    event_count += 1
                previous_trade_id, previous_event_ns, previous_trade_ns = trade_id, event_ns, trade_ns
                previous_receive_ns, previous_receive_mono_ns = receive_ns, receive_mono_ns
                if observation_count > MAX_EVENTS:
                    raise SpotTradesRejected("trade count exceeds Dataset event limit")
            elif kind in {"gap_detected", "clock_discontinuity"}:
                raise SpotTradesRejected(f"line {line_number}: capture contains {kind}")
            elif kind == "session_disconnected":
                if (not connected or disconnected or row.get("session_id") != session
                        or row.get("reason") not in {"hourly_rotation", "local_shutdown"}):
                    raise SpotTradesRejected(f"line {line_number}: unclean session end")
                disconnected = True
            elif kind == "capture_stopped":
                if (not disconnected or footer is not None
                        or row.get("reason") not in {"duration_elapsed", "sigterm"}
                        or row.get("observation_count") != observation_count
                        or row.get("bytes_before_stop_record") != bytes_before_record):
                    raise SpotTradesRejected(f"line {line_number}: capture stop summary mismatch")
                footer = row
                stop_ns_recorded = uint(row.get("recorded_at_unix_ns"), "capture stop time", line_number)
                stop_mono_ns = uint(row.get("recorded_at_monotonic_ns"), "capture monotonic stop", line_number)
            else:
                raise SpotTradesRejected(f"line {line_number}: unsupported record type {kind!r}")

    events_writer.flush()
    os.fsync(events_writer.fileno())
    if not (header and footer and connected and disconnected and session and event_count):
        raise SpotTradesRejected("capture is incomplete or has no valid trade observations")
    if start_mono_ns is None or stop_mono_ns is None or stop_mono_ns <= start_mono_ns:
        raise SpotTradesRejected("monotonic capture lifecycle is invalid")
    if start_ns_recorded is None or stop_ns_recorded is None or connect_ns is None:
        raise SpotTradesRejected("capture lifecycle timestamps are missing")
    if start_ns_recorded > start_ns or connect_ns > start_ns or stop_ns_recorded < end_ns:
        raise SpotTradesRejected("capture/session lifecycle does not cover requested window")
    if not has_pre_window:
        raise SpotTradesRejected("capture has no pre-window market observation")
    if not has_post_window:
        raise SpotTradesRejected("capture has no observation at/after window end")
    if file_sha256(capture) != capture_hash:
        raise SpotTradesRejected("source capture changed during conversion")
    window = {"start_ns": start_ns, "end_ns": end_ns}
    logical = logical_hash.hexdigest()
    builder = source_revision()
    manifest = {"schema": "td_market_data_spot_trades_dataset", "version": 1,
                "exchange": "BINANCE", "market": "SPOT", "symbol": symbol,
                "session_id": session, "window": window,
                "source_host_id": header.get("source_host_id"),
                "capture_started_at_ns": start_ns_recorded,
                "capture_stopped_at_ns": stop_ns_recorded,
                "max_receive_gap_ms": max_gap_ms,
                "event_count": event_count, "first_trade_id": first_event["trade_id"],
                "last_trade_id": last_event["trade_id"],
                "first_event_time_ns": first_event["event_time_ns"],
                "last_event_time_ns": last_event["event_time_ns"],
                "first_receive_time_ns": first_event["receive_time_ns"],
                "last_receive_time_ns": last_event["receive_time_ns"],
                "events_file": "events.jsonl", "events_sha256": events_hash.hexdigest(),
                "source_capture_sha256": capture_hash,
                "collector_code_sha256": header["collector_executable_sha256"],
                "collector_source_revision": header["source_revision"],
                "converter_code_sha256": file_sha256(Path(__file__).resolve()),
                "builder_source": builder, "logical_fingerprint_sha256": logical,
                "replay_usable": builder["tree_sha256"] is not None,
                "warnings": [] if builder["tree_sha256"] else ["builder source tree fingerprint is unavailable"]}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode() + b"\n"
    with (stage / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    stage_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(stage_fd)
    finally:
        os.close(stage_fd)
    rename_noreplace(stage, output)
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    args = parser.parse_args()
    try:
        manifest = publish(args.capture, args.output, args.symbol.upper(), args.start, args.end)
    except (SpotTradesRejected, OSError, ValueError) as error:
        raise SystemExit(f"Spot trades Dataset rejected: {error}") from error
    print(f"Spot trades Dataset published: events={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
