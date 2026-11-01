#!/usr/bin/env python3
"""Publish a strict, replay-usable USD-M depth Dataset from one VPS capture."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


DOMAIN = b"td-gateway-usdm-depth-dataset-v2\0"
SOURCE_DOMAIN = b"td-gateway-usdm-depth-source-v2\0"
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 50_000
MAX_LEVEL_UPDATES = 500_000
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024


class CaptureRejected(ValueError):
    pass


def compact(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little"))
    digest.update(value)


def hash_json_field(digest: Any, value: Any) -> None:
    hash_field(digest, compact(value))


def parse_ns(value: str) -> int:
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


def parse_capture_line(raw: bytes, number: int) -> dict[str, Any]:
    if not raw.endswith(b"\n") or len(raw) > 4 * 1024 * 1024:
        raise CaptureRejected(f"line {number}: truncated or oversized record")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureRejected(f"line {number}: malformed JSON") from error
    if (not isinstance(value, dict) or value.get("schema") != "td_public_market_capture_record"
            or value.get("version") != 1):
        raise CaptureRejected(f"line {number}: unsupported capture record")
    return value


def integer(value: Any, field: str, line: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CaptureRejected(f"line {line}: {field} must be a nonnegative integer")
    return value


def decimal(value: Any, field: str) -> Decimal:
    if not isinstance(value, str):
        raise CaptureRejected(f"invalid {field}: expected a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise CaptureRejected(f"invalid {field}") from error
    if not parsed.is_finite():
        raise CaptureRejected(f"invalid {field}: non-finite value")
    return parsed


def normalize_levels(rows: Any, field: str, allow_zero: bool) -> tuple[list[list[str]], int]:
    if not isinstance(rows, list):
        raise CaptureRejected(f"{field} must be an array")
    result: list[list[str]] = []
    seen: dict[str, str] = {}
    duplicates = 0
    for row in rows:
        if not isinstance(row, list) or len(row) != 2:
            raise CaptureRejected(f"{field} contains a malformed level")
        price, quantity = row
        price_value = decimal(price, f"{field} price")
        quantity_value = decimal(quantity, f"{field} quantity")
        if price_value <= 0 or quantity_value < 0 or (not allow_zero and quantity_value == 0):
            raise CaptureRejected(f"{field} contains an out-of-range price/quantity")
        if price in seen:
            if seen[price] != quantity:
                raise CaptureRejected(f"{field} repeats a price with conflicting quantities")
            duplicates += 1
            continue
        seen[price] = quantity
        result.append([price, quantity])
    return result, duplicates


def source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parent

    def git(*args: str) -> bytes | None:
        result = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, check=False)
        return result.stdout if result.returncode == 0 else None

    revision_bytes = git("rev-parse", "HEAD")
    if not revision_bytes:
        return {"committed_revision": None, "working_tree_state": "unknown",
                "source_tree_fingerprint": None}
    revision = revision_bytes.decode().strip()
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    diff = git("diff", "--binary", "HEAD", "--")
    if status is None or diff is None:
        return {"committed_revision": revision, "working_tree_state": "unknown",
                "source_tree_fingerprint": None}
    untracked = git("ls-files", "--others", "--exclude-standard", "-z")
    if untracked is None:
        return {"committed_revision": revision, "working_tree_state": "unknown",
                "source_tree_fingerprint": None}
    state = "clean" if not status else "dirty"
    digest = hashlib.sha256()
    digest.update(b"td-gateway-source-tree-v1\0")
    digest.update(len(revision.encode()).to_bytes(8, "little"))
    digest.update(revision.encode())
    digest.update(len(diff).to_bytes(8, "little"))
    digest.update(diff)
    for relative in filter(None, untracked.split(b"\0")):
        candidate = root / os.fsdecode(relative)
        try:
            info = candidate.lstat()
            if not candidate.is_file() or candidate.is_symlink() or info.st_size > 16 * 1024 * 1024:
                return {"committed_revision": revision, "working_tree_state": state,
                        "source_tree_fingerprint": None}
            contents = candidate.read_bytes()
        except OSError:
            return {"committed_revision": revision, "working_tree_state": state,
                    "source_tree_fingerprint": None}
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return {"committed_revision": revision, "working_tree_state": state,
            "source_tree_fingerprint": digest.hexdigest()}


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic no-replace dataset publication requires Linux renameat2")
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), str(destination))


def write_line(path: Path, value: Any) -> bytes:
    line = compact(value)
    with path.open("ab") as stream:
        stream.write(line + b"\n")
    return line


def publish_dataset(capture: Path, output: Path, symbol: str, start_ns: int, end_ns: int) -> dict[str, Any]:
    capture = capture.resolve(strict=True)
    if not output.is_absolute():
        raise CaptureRejected("Dataset output path must be absolute")
    output.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output.parent.resolve(strict=True)
    output = output_parent / output.name
    repository = Path(__file__).resolve().parent
    if output == repository or repository in output.parents:
        raise CaptureRejected("Dataset output must be outside the Python source repository")
    if capture.stat().st_size > MAX_CAPTURE_BYTES:
        raise CaptureRejected("capture exceeds 256 MiB")
    if start_ns >= end_ns or start_ns % 1_000_000 or end_ns % 1_000_000:
        raise CaptureRejected("Dataset window must be positive and millisecond aligned")
    start_ms, end_ms = start_ns // 1_000_000, end_ns // 1_000_000
    capture_sha = hash_file(capture)
    session_id: str | None = None
    started: dict[str, Any] | None = None
    stopped: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_ns = -1
    snapshot_id = -1
    window_snapshot: dict[str, Any] | None = None
    book_bids: dict[str, Decimal] = {}
    book_asks: dict[str, Decimal] = {}
    connected = disconnected = False
    raw_observation_count = 0
    capture_bytes_before_stop: int | None = None
    bridge_receive_ns: int | None = None
    accepted_events: list[dict[str, Any]] = []
    accepted_frame_count = 0
    previous_id: int | None = None
    previous_event_ms: int | None = None
    previous_receive_ns: int | None = None
    previous_receive_mono_ns: int | None = None
    last_ws_receive_ns: int | None = None
    last_ws_receive_mono_ns: int | None = None
    previous_effective_ns: int | None = None
    max_gap_ms = 0
    expected_source_url = f"wss://fstream.binance.com/public/ws/{symbol.lower()}@depth@100ms"

    capture_bytes = 0
    with capture.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            bytes_before_record = capture_bytes
            capture_bytes += len(raw)
            row = parse_capture_line(raw, line_number)
            kind = row.get("record_type")
            if line_number == 1:
                if kind != "capture_started" or row.get("stream") != "usdm-depth" or row.get("symbol") != symbol:
                    raise CaptureRejected("capture header does not identify this USD-M symbol")
                started = row
                max_gap_ms = integer(row.get("max_gap_ms"), "max_gap_ms", line_number)
                collector_hash = row.get("collector_executable_sha256")
                if (not isinstance(row.get("collector_executable_path"), str)
                        or not isinstance(collector_hash, str) or len(collector_hash) != 64
                        or any(char not in "0123456789abcdef" for char in collector_hash)
                        or not isinstance(row.get("source_revision"), str)):
                    raise CaptureRejected("capture has no valid collector code/source fingerprint")
            elif kind == "session_connected":
                if connected or disconnected or snapshot is not None:
                    raise CaptureRejected(f"line {line_number}: reconnects are not admissible in one Dataset")
                if row.get("url") != expected_source_url:
                    raise CaptureRejected(f"line {line_number}: WebSocket URL mismatch")
                session_id = row.get("session_id")
                if not isinstance(session_id, str) or not session_id:
                    raise CaptureRejected(f"line {line_number}: session ID missing")
                connected = True
            elif kind == "depth_snapshot":
                if not connected or snapshot is not None or row.get("session_id") != session_id:
                    raise CaptureRejected(f"line {line_number}: invalid or duplicate depth snapshot")
                received_ns = integer(row.get("received_at_unix_ns"), "snapshot receive time", line_number)
                raw_snapshot = row.get("raw_snapshot")
                if not isinstance(raw_snapshot, str):
                    raise CaptureRejected(f"line {line_number}: raw REST snapshot missing")
                raw_sha = hashlib.sha256(raw_snapshot.encode()).hexdigest()
                if raw_sha != row.get("raw_snapshot_sha256"):
                    raise CaptureRejected(f"line {line_number}: REST snapshot hash mismatch")
                try:
                    parsed = json.loads(raw_snapshot)
                except json.JSONDecodeError as error:
                    raise CaptureRejected(f"line {line_number}: invalid REST snapshot JSON") from error
                if not isinstance(parsed, dict):
                    raise CaptureRejected(f"line {line_number}: REST snapshot must be an object")
                snapshot_id = integer(parsed.get("lastUpdateId"), "snapshot update ID", line_number)
                bids, bid_duplicates = normalize_levels(parsed.get("bids"), "snapshot bids", False)
                asks, ask_duplicates = normalize_levels(parsed.get("asks"), "snapshot asks", False)
                if bid_duplicates or ask_duplicates or parsed.get("bids") != row.get("bids") or parsed.get("asks") != row.get("asks"):
                    raise CaptureRejected(f"line {line_number}: REST snapshot levels disagree with raw evidence")
                snapshot_ns = received_ns
                snapshot = {"last_update_id": snapshot_id, "bids": bids, "asks": asks,
                            "received_at_ns": received_ns}
                book_bids = {price: Decimal(quantity) for price, quantity in bids}
                book_asks = {price: Decimal(quantity) for price, quantity in asks}
            elif kind == "market_frame":
                raw_observation_count += 1
                if not connected or disconnected or snapshot is None or row.get("session_id") != session_id:
                    raise CaptureRejected(f"line {line_number}: depth frame outside initialized session")
                try:
                    frame = json.loads(row["raw_frame"])
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise CaptureRejected(f"line {line_number}: invalid raw depth frame") from error
                raw_frame = row["raw_frame"].encode()
                if hashlib.sha256(raw_frame).hexdigest() != row.get("raw_frame_sha256"):
                    raise CaptureRejected(f"line {line_number}: raw depth frame hash mismatch")
                if (not isinstance(frame, dict) or frame.get("e") != "depthUpdate"
                        or frame.get("s") != symbol
                        or frame.get("U") != row.get("first_update_id")
                        or frame.get("u") != row.get("final_update_id")
                        or frame.get("pu") != row.get("previous_final_update_id")):
                    raise CaptureRejected(f"line {line_number}: raw frame update IDs or identity mismatch")
                if ((row.get("accepted_for_book") is True or row.get("event_time_ms") is not None)
                        and frame.get("E") != row.get("event_time_ms")):
                    raise CaptureRejected(f"line {line_number}: raw frame event time disagrees with parsed field")
                first_id = integer(frame.get("U"), "first update ID", line_number)
                final_id = integer(frame.get("u"), "final update ID", line_number)
                event_ms = integer(frame.get("E"), "event time", line_number)
                receive_ns = integer(row.get("received_at_unix_ns"), "receive time", line_number)
                receive_mono_ns = integer(row.get("receive_monotonic_ns"), "monotonic receive time", line_number)
                if last_ws_receive_ns is not None and last_ws_receive_mono_ns is not None:
                    wall_gap = receive_ns - last_ws_receive_ns
                    mono_gap = receive_mono_ns - last_ws_receive_mono_ns
                    if wall_gap <= 0 or mono_gap <= 0 or mono_gap > max_gap_ms * 1_000_000:
                        raise CaptureRejected(f"line {line_number}: raw WebSocket receive gap exceeds policy")
                    if abs(wall_gap - mono_gap) > 2_000_000_000:
                        raise CaptureRejected(f"line {line_number}: raw frame clock discontinuity")
                last_ws_receive_ns = receive_ns
                last_ws_receive_mono_ns = receive_mono_ns
                if previous_receive_mono_ns is not None and receive_mono_ns <= previous_receive_mono_ns:
                    raise CaptureRejected(f"line {line_number}: monotonic receive time regressed")
                if frame.get("b") is None or frame.get("a") is None:
                    raise CaptureRejected(f"line {line_number}: raw frame lacks changed bid/ask levels")
                if row.get("accepted_for_book") is False:
                    if row.get("reject_reason") != "older_than_snapshot_or_duplicate" or final_id > (previous_id or snapshot_id):
                        raise CaptureRejected(f"line {line_number}: rejected frame proves an invalid depth sequence")
                    continue
                if row.get("accepted_for_book") is not True or final_id <= (previous_id or snapshot_id):
                    raise CaptureRejected(f"line {line_number}: invalid accepted update ordering")
                if previous_id is None:
                    if not (first_id <= snapshot_id + 1 <= final_id):
                        raise CaptureRejected(f"line {line_number}: first update does not bridge the REST snapshot")
                    bridge_receive_ns = receive_ns
                else:
                    previous_final = frame.get("pu")
                    if not isinstance(previous_final, int) or previous_final != previous_id:
                        raise CaptureRejected(f"line {line_number}: USD-M previous update ID discontinuity")
                if previous_event_ms is not None and event_ms < previous_event_ms:
                    raise CaptureRejected(f"line {line_number}: event time regressed in update order")
                if previous_receive_ns is not None:
                    receive_gap = receive_ns - previous_receive_ns
                    if receive_gap <= 0 or receive_gap > max_gap_ms * 1_000_000:
                        raise CaptureRejected(f"line {line_number}: receive gap exceeds capture policy")
                    if abs(receive_gap - (receive_mono_ns - (previous_receive_mono_ns or 0))) > 2_000_000_000:
                        raise CaptureRejected(f"line {line_number}: wall/monotonic clock discontinuity")
                event_ns = event_ms * 1_000_000
                if receive_ns < event_ns:
                    raise CaptureRejected(f"line {line_number}: exchange event is ahead of receive time")
                effective_ns = max(event_ns, receive_ns)
                if previous_effective_ns is not None and effective_ns < previous_effective_ns:
                    raise CaptureRejected(f"line {line_number}: data availability time regressed")
                levels: list[dict[str, str]] = []
                seen: dict[tuple[str, str], str] = {}
                duplicate_count = 0
                for side, key in (("BID", "b"), ("ASK", "a")):
                    rows, side_duplicates = normalize_levels(frame[key], key, True)
                    duplicate_count += side_duplicates
                    for price, quantity in rows:
                        identity = (side, price)
                        if identity in seen:
                            if seen[identity] != quantity:
                                raise CaptureRejected(f"line {line_number}: repeated update level conflicts")
                            duplicate_count += 1
                            continue
                        seen[identity] = quantity
                        levels.append({"side": side, "price": price, "quantity": quantity})
                if not levels:
                    raise CaptureRejected(f"line {line_number}: depth update has no changed levels")
                if effective_ns >= start_ns and window_snapshot is None:
                    window_snapshot = {
                        "last_update_id": previous_id if previous_id is not None else snapshot_id,
                        "bids": [[price, format(quantity, "f")] for price, quantity in
                                 sorted(book_bids.items(), key=lambda item: Decimal(item[0]), reverse=True)],
                        "asks": [[price, format(quantity, "f")] for price, quantity in
                                 sorted(book_asks.items(), key=lambda item: Decimal(item[0]))],
                        "received_at_ns": previous_receive_ns if previous_receive_ns is not None else snapshot_ns,
                    }
                if effective_ns >= start_ns:
                    accepted_events.append({"event_ts_ms": event_ms, "first_update_id": first_id,
                                            "final_update_id": final_id, "prev_final_update_id": frame.get("pu"),
                                            "duplicate_level_rows": duplicate_count, "levels": levels,
                                            "receive_ts_ms": receive_ns // 1_000_000,
                                            "receive_mono_ns": receive_mono_ns})
                for change in levels:
                    book = book_bids if change["side"] == "BID" else book_asks
                    quantity = Decimal(change["quantity"])
                    if quantity == 0:
                        book.pop(change["price"], None)
                    else:
                        book[change["price"]] = quantity
                if (not book_bids or not book_asks
                        or max(map(Decimal, book_bids)) >= min(map(Decimal, book_asks))):
                    raise CaptureRejected(f"line {line_number}: depth replay produces an empty or crossed book")
                previous_id = final_id
                previous_event_ms = event_ms
                previous_receive_ns = receive_ns
                previous_receive_mono_ns = receive_mono_ns
                previous_effective_ns = effective_ns
                accepted_frame_count += 1
                if accepted_frame_count > MAX_EVENTS:
                    raise CaptureRejected("depth event count exceeds Dataset limit")
            elif kind == "gap_detected" or kind == "clock_discontinuity":
                raise CaptureRejected(f"line {line_number}: capture contains {kind}")
            elif kind == "session_disconnected":
                if not connected or disconnected or row.get("session_id") != session_id:
                    raise CaptureRejected(f"line {line_number}: invalid session termination")
                if row.get("reason") not in {"hourly_rotation", "local_shutdown"}:
                    raise CaptureRejected(f"line {line_number}: capture did not end cleanly")
                disconnected = True
            elif kind == "capture_stopped":
                if (not disconnected or stopped is not None
                        or row.get("reason") not in {"duration_elapsed", "sigterm"}):
                    raise CaptureRejected(f"line {line_number}: capture was not cleanly finalized")
                if (row.get("bytes_before_stop_record") != bytes_before_record
                        or row.get("observation_count") != raw_observation_count):
                    raise CaptureRejected(f"line {line_number}: stop summary disagrees with capture contents")
                capture_bytes_before_stop = bytes_before_record
                stopped = row
            else:
                raise CaptureRejected(f"line {line_number}: unsupported record type {kind!r}")

    if not (started and connected and disconnected and stopped and snapshot and window_snapshot and session_id):
        raise CaptureRejected("capture is incomplete")
    if hash_file(capture) != capture_sha:
        raise CaptureRejected("source capture changed while it was being converted")
    if bridge_receive_ns is None or previous_id is None or not accepted_events:
        raise CaptureRejected("capture has no snapshot-bridged depth events")
    snapshot = window_snapshot
    snapshot_id = snapshot["last_update_id"]
    snapshot_ns = snapshot["received_at_ns"]
    stopped_ns = integer(stopped.get("recorded_at_unix_ns"), "capture stop time", 0)
    stop_mono = integer(stopped.get("recorded_at_monotonic_ns"), "capture stop monotonic time", 0)
    start_mono = integer(started.get("recorded_at_monotonic_ns"), "capture start monotonic time", 1)
    if stop_mono <= start_mono:
        raise CaptureRejected("monotonic capture lifecycle is invalid")
    if snapshot_ns > start_ns or bridge_receive_ns > start_ns:
        raise CaptureRejected("window starts before the snapshot bridge; no synchronized warmup is available")
    if stopped_ns >= end_ns or stopped_ns < start_ns:
        raise CaptureRejected("window must include the clean capture commit time")
    if capture_bytes_before_stop is None:
        raise CaptureRejected("capture byte summary is missing")
    if len(accepted_events) == 0 or stopped_ns <= snapshot_ns:
        raise CaptureRejected("capture contains no usable temporal depth coverage")

    total_levels = sum(len(event["levels"]) for event in accepted_events)
    if total_levels > MAX_LEVEL_UPDATES:
        raise CaptureRejected("depth level updates exceed Dataset limit")
    if any(event["event_ts_ms"] * 1_000_000 >= end_ns or event["receive_ts_ms"] * 1_000_000 >= end_ns
           for event in accepted_events):
        raise CaptureRejected("capture has an event outside the Dataset window")
    # Replay all levels to ensure the snapshot, update chain, and book remain valid.
    bids = {price: decimal(quantity, "snapshot bid quantity") for price, quantity in snapshot["bids"]}
    asks = {price: decimal(quantity, "snapshot ask quantity") for price, quantity in snapshot["asks"]}
    for event in accepted_events:
        for level in event["levels"]:
            book = bids if level["side"] == "BID" else asks
            quantity = decimal(level["quantity"], "depth quantity")
            if quantity == 0:
                book.pop(level["price"], None)
            else:
                book[level["price"]] = quantity
        if not bids or not asks or max(map(Decimal, bids)) >= min(map(Decimal, asks)):
            raise CaptureRejected("depth replay produces an empty or crossed book")

    source_tables = ["usdm_orderbook_updates_vps_capture", "usdm_orderbook_sessions_vps_capture",
                    "usdm_orderbook_snapshots_vps_capture", "usdm_orderbook_commits_vps_capture",
                    "usdm_orderbook_checkpoints_vps_capture", "usdm_orderbook_health_vps_capture"]
    health_state = "SYNCED"
    health_session = session_id
    health_rows: list[dict[str, Any]] = []
    if start_ns <= bridge_receive_ns < end_ns:
        health_rows.append({"schema": "td_gateway_usdm_depth_health_event", "version": 2,
                            "exchange": "BINANCE", "market": "USD_M", "contract": "PERPETUAL",
                            "symbol": symbol, "session_id": session_id,
                            "observed_at_ns": bridge_receive_ns, "status": "SYNCED",
                            "reason": "REST snapshot bridge validated from raw VPS capture",
                            "reconnect_count": 0, "resync_count": 0, "retry_delay_ms": 0})

    snapshot_record = {"schema": "td_gateway_usdm_depth_snapshot", "version": 2,
                       "exchange": "BINANCE", "market": "USD_M", "contract": "PERPETUAL",
                       "symbol": symbol, "session_id": session_id,
                       "received_at_ns": snapshot_ns, "last_update_id": snapshot_id,
                       "bids": [{"price": price, "quantity": quantity} for price, quantity in snapshot["bids"]],
                       "asks": [{"price": price, "quantity": quantity} for price, quantity in snapshot["asks"]]}
    snapshot_bytes = compact(snapshot_record)
    event_records = []
    for event in accepted_events:
        event_records.append({"schema": "td_gateway_usdm_depth_event", "version": 2,
                              "exchange": "BINANCE", "market": "USD_M", "contract": "PERPETUAL",
                              "symbol": symbol, "session_id": session_id,
                              "event_ts_ns": event["event_ts_ms"] * 1_000_000,
                              "receive_ts_ns": event["receive_ts_ms"] * 1_000_000,
                              "first_update_id": event["first_update_id"],
                              "final_update_id": event["final_update_id"],
                              "prev_final_update_id": event["prev_final_update_id"],
                              "duplicate_level_rows": event["duplicate_level_rows"],
                              "levels": event["levels"]})
    event_lines = [compact(row) for row in event_records]
    health_rows.sort(key=lambda item: (item["observed_at_ns"], item["status"], item["session_id"] or ""))
    health_lines = [compact(row) for row in health_rows]
    event_bytes = b"".join(line + b"\n" for line in event_lines)
    health_bytes = b"".join(line + b"\n" for line in health_lines)
    source_hash = hashlib.sha256()
    source_hash.update(SOURCE_DOMAIN)
    window = {"start_ns": start_ns, "end_ns": end_ns}
    commit_ms = stopped_ns // 1_000_000
    source_tables_tuple = source_tables
    hash_json_field(source_hash, [symbol, session_id, snapshot_ns // 1_000_000, commit_ms,
                                  previous_id, source_tables_tuple, window, health_state, health_session])
    hash_json_field(source_hash, [snapshot_id, snapshot["bids"], snapshot["asks"]])
    for health in health_rows:
        hash_json_field(source_hash, health)
    for event in event_records:
        hash_json_field(source_hash, event)
    source_fingerprint = source_hash.hexdigest()

    logical_hash = hashlib.sha256()
    logical_hash.update(DOMAIN)
    logical_hash.update(start_ns.to_bytes(8, "little", signed=True))
    logical_hash.update(end_ns.to_bytes(8, "little", signed=True))
    hash_json_field(logical_hash, health_state)
    hash_json_field(logical_hash, health_session)
    hash_field(logical_hash, snapshot_bytes)
    for line in event_lines + health_lines:
        hash_field(logical_hash, line)

    revision = source_revision()
    health_issue_count = 0
    warmup_count = 0
    manifest = {"schema": "td_gateway_usdm_depth_dataset", "version": 2,
                "exchange": "BINANCE", "market": "USD_M", "contract": "PERPETUAL",
                "symbol": symbol, "session_id": session_id, "window": window,
                "source_tables": source_tables, "snapshot_file": "snapshot.json",
                "events_file": "events.jsonl", "health_file": "health.jsonl",
                "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
                "events_sha256": hashlib.sha256(event_bytes).hexdigest(),
                "health_sha256": hashlib.sha256(health_bytes).hexdigest(),
                "source_fingerprint_sha256": source_fingerprint,
                "logical_fingerprint_sha256": logical_hash.hexdigest(),
                "source_revision": revision, "event_count": len(event_records),
                "level_update_count": sum(len(item["levels"]) for item in event_records),
                "first_update_id": event_records[0]["first_update_id"],
                "last_update_id": previous_id, "snapshot_update_id": snapshot_id,
                "committed_through_update_id": previous_id, "committed_at_ns": commit_ms * 1_000_000,
                "receive_time_known_event_count": len(event_records),
                "receive_time_unknown_event_count": 0, "clock_order_violation_event_count": 0,
                "availability_order_violation_event_count": 0, "warmup_event_count": warmup_count,
                "health_event_count": len(health_rows), "health_issue_count": health_issue_count,
                "health_state_at_window_start": health_state,
                "health_session_id_at_window_start": health_session,
                "replay_usable": revision["source_tree_fingerprint"] is not None,
                "warnings": [] if revision["source_tree_fingerprint"] is not None
                else ["builder source tree fingerprint is unavailable"]}
    manifest["source_capture_sha256"] = capture_sha
    manifest["collector_code_sha256"] = started["collector_executable_sha256"]
    manifest["collector_source_revision"] = started["source_revision"]
    manifest["converter_code_sha256"] = hash_file(Path(__file__).resolve())

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (stage / "snapshot.json").write_bytes(snapshot_bytes)
        (stage / "events.jsonl").write_bytes(event_bytes)
        (stage / "health.jsonl").write_bytes(health_bytes)
        (stage / "manifest.json").write_bytes(json.dumps(manifest, ensure_ascii=False, indent=2).encode())
        for path in stage.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        dir_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        rename_noreplace(stage, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, type=parse_ns, help="RFC3339 UTC window start")
    parser.add_argument("--end", required=True, type=parse_ns, help="RFC3339 UTC window end")
    args = parser.parse_args()
    symbol = args.symbol.upper()
    if not symbol.isascii() or not symbol.isalnum():
        raise SystemExit("invalid USD-M symbol")
    try:
        manifest = publish_dataset(args.capture, args.output, symbol, args.start, args.end)
    except (CaptureRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M depth Dataset rejected: {error}") from error
    print(f"USD-M depth Dataset published: events={manifest['event_count']} "
          f"replay_usable={manifest['replay_usable']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
