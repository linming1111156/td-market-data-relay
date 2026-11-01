#!/usr/bin/env python3
"""Publish a point-in-time Spot depth Dataset from one continuous capture.

This deliberately uses a standalone Python schema. It does not read or alter
any frozen historical datasets or import the private td-gateway repository.
"""

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


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_EVENTS = 100_000


class CaptureRejected(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def uint(value: Any, name: str, line: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CaptureRejected(f"line {line}: {name} must be a nonnegative integer")
    return value


def decimal_string(value: Any, name: str) -> Decimal:
    if not isinstance(value, str):
        raise CaptureRejected(f"{name} must be encoded as a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise CaptureRejected(f"invalid {name}") from error
    if not number.is_finite():
        raise CaptureRejected(f"non-finite {name}")
    return number


def levels(value: Any, name: str, zero_allowed: bool, allow_empty: bool = False) -> list[list[str]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise CaptureRejected(f"{name} must be an {'array' if allow_empty else 'non-empty array'}")
    out: list[list[str]] = []
    seen: dict[str, str] = {}
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise CaptureRejected(f"malformed {name} level")
        price, quantity = row
        p, q = decimal_string(price, f"{name} price"), decimal_string(quantity, f"{name} quantity")
        if p <= 0 or q < 0 or (q == 0 and not zero_allowed):
            raise CaptureRejected(f"out-of-range {name} level")
        if price in seen and seen[price] != quantity:
            raise CaptureRejected(f"conflicting duplicate {name} price")
        if price not in seen:
            out.append([price, quantity])
            seen[price] = quantity
    return out


def source_revision() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           stderr=subprocess.DEVNULL).decode().strip()
        state = subprocess.check_output(["git", "status", "--porcelain=v1", "-z",
                                         "--untracked-files=all"], cwd=root,
                                         stderr=subprocess.DEVNULL)
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--"], cwd=root,
                                       stderr=subprocess.DEVNULL)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                             "-z"], cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tree_sha256": None, "tree_state": "unknown"}
    tree = hashlib.sha256(b"td-market-data-source-tree-v1\0" + revision.encode() + diff)
    for name in filter(None, untracked.split(b"\0")):
        path = root / os.fsdecode(name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            return {"revision": revision, "tree_sha256": None,
                    "tree_state": "dirty" if state else "clean"}
        raw = path.read_bytes()
        tree.update(len(name).to_bytes(8, "little")); tree.update(name)
        tree.update(len(raw).to_bytes(8, "little")); tree.update(raw)
    return {"revision": revision, "tree_sha256": tree.hexdigest(),
            "tree_state": "dirty" if state else "clean"}


def publish(capture: Path, output: Path, symbol: str, start_ns: int, end_ns: int) -> dict[str, Any]:
    capture = capture.resolve(strict=True)
    if not output.is_absolute() or start_ns >= end_ns:
        raise CaptureRejected("output must be absolute and the window must be positive")
    if capture.stat().st_size > MAX_CAPTURE_BYTES:
        raise CaptureRejected("capture exceeds 256 MiB")
    if not symbol.isascii() or not symbol.isalnum():
        raise CaptureRejected("invalid Spot symbol")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    if output.exists():
        raise CaptureRejected("Dataset output already exists")
    repo = Path(__file__).resolve().parent
    if output == repo or repo in output.parents:
        raise CaptureRejected("Dataset output must be outside the source repository")

    source_hash = file_sha256(capture)
    start: dict[str, Any] | None = None
    stop: dict[str, Any] | None = None
    session: str | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_ns = -1
    snapshot_id = -1
    connected = disconnected = False
    last_id: int | None = None
    last_event_ns: int | None = None
    last_receive_ns: int | None = None
    last_mono_ns: int | None = None
    last_frame_receive_ns: int | None = None
    last_frame_mono_ns: int | None = None
    bridge_ns: int | None = None
    frame_count = 0
    accepted: list[dict[str, Any]] = []
    max_gap_ms = 0

    with capture.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 4 * 1024 * 1024:
                raise CaptureRejected(f"line {number}: truncated or oversized JSONL record")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureRejected(f"line {number}: invalid JSON") from error
            if not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record" or row.get("version") != 1:
                raise CaptureRejected(f"line {number}: unsupported capture record")
            kind = row.get("record_type")
            if number == 1:
                if kind != "capture_started" or row.get("stream") != "spot-depth" or row.get("symbol") != symbol:
                    raise CaptureRejected("capture header does not identify this Spot depth symbol")
                start = row
                max_gap_ms = uint(row.get("max_gap_ms"), "max_gap_ms", number)
                if not isinstance(row.get("collector_executable_sha256"), str) or len(row["collector_executable_sha256"]) != 64:
                    raise CaptureRejected("capture has no collector code fingerprint")
            elif kind == "session_connected":
                if connected or disconnected or snapshot is not None or row.get("url") != f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth@100ms":
                    raise CaptureRejected(f"line {number}: unexpected connection or endpoint")
                session = row.get("session_id")
                if not isinstance(session, str) or not session:
                    raise CaptureRejected(f"line {number}: session ID missing")
                connected = True
            elif kind == "depth_snapshot":
                if not connected or snapshot is not None or row.get("session_id") != session:
                    raise CaptureRejected(f"line {number}: invalid REST snapshot lifecycle")
                raw_snapshot = row.get("raw_snapshot")
                if not isinstance(raw_snapshot, str) or sha256(raw_snapshot.encode()) != row.get("raw_snapshot_sha256"):
                    raise CaptureRejected(f"line {number}: REST snapshot evidence hash mismatch")
                try:
                    evidence = json.loads(raw_snapshot)
                except json.JSONDecodeError as error:
                    raise CaptureRejected(f"line {number}: invalid REST snapshot JSON") from error
                snapshot_id = uint(evidence.get("lastUpdateId"), "snapshot update ID", number)
                bids, asks = levels(evidence.get("bids"), "snapshot bids", False), levels(evidence.get("asks"), "snapshot asks", False)
                if bids != row.get("bids") or asks != row.get("asks"):
                    raise CaptureRejected(f"line {number}: snapshot levels disagree with raw REST response")
                snapshot_ns = uint(row.get("received_at_unix_ns"), "snapshot receive time", number)
                snapshot = {"last_update_id": snapshot_id, "received_at_ns": snapshot_ns,
                            "bids": bids, "asks": asks}
            elif kind == "market_frame":
                frame_count += 1
                if not connected or disconnected or snapshot is None or row.get("session_id") != session:
                    raise CaptureRejected(f"line {number}: frame outside initialized session")
                raw_frame = row.get("raw_frame")
                if not isinstance(raw_frame, str) or sha256(raw_frame.encode()) != row.get("raw_frame_sha256"):
                    raise CaptureRejected(f"line {number}: raw frame hash mismatch")
                try:
                    frame = json.loads(raw_frame)
                except json.JSONDecodeError as error:
                    raise CaptureRejected(f"line {number}: invalid depth frame") from error
                if (frame.get("e") != "depthUpdate" or frame.get("s") != symbol
                        or frame.get("U") != row.get("first_update_id")
                        or frame.get("u") != row.get("final_update_id")):
                    raise CaptureRejected(f"line {number}: raw frame identity/update IDs disagree")
                if ((row.get("accepted_for_book") is True or row.get("event_time_ms") is not None)
                        and frame.get("E") != row.get("event_time_ms")):
                    raise CaptureRejected(f"line {number}: raw frame event time disagrees with parsed field")
                first_id = uint(frame.get("U"), "first update ID", number)
                final_id = uint(frame.get("u"), "final update ID", number)
                event_ns = uint(frame.get("E"), "exchange event time", number) * 1_000_000
                receive_ns = uint(row.get("received_at_unix_ns"), "receive time", number)
                mono_ns = uint(row.get("receive_monotonic_ns"), "monotonic receive time", number)
                if event_ns > receive_ns:
                    raise CaptureRejected(f"line {number}: exchange event occurs after local receipt")
                if last_frame_receive_ns is not None:
                    wall_gap = receive_ns - last_frame_receive_ns
                    mono_gap = mono_ns - (last_frame_mono_ns or 0)
                    if wall_gap <= 0 or mono_gap <= 0 or mono_gap > max_gap_ms * 1_000_000 or abs(wall_gap - mono_gap) > 2_000_000_000:
                        raise CaptureRejected(f"line {number}: receive gap or wall/monotonic clock discontinuity")
                last_frame_receive_ns, last_frame_mono_ns = receive_ns, mono_ns
                if row.get("accepted_for_book") is False:
                    if row.get("reject_reason") != "older_than_snapshot_or_duplicate" or final_id > (last_id or snapshot_id):
                        raise CaptureRejected(f"line {number}: rejected depth frame is not a harmless duplicate")
                    continue
                if row.get("accepted_for_book") is not True or final_id <= (last_id or snapshot_id):
                    raise CaptureRejected(f"line {number}: invalid accepted depth update")
                if last_id is None:
                    if not (first_id <= snapshot_id + 1 <= final_id):
                        raise CaptureRejected(f"line {number}: first event does not bridge REST snapshot")
                    bridge_ns = receive_ns
                elif first_id > last_id + 1:
                    raise CaptureRejected(f"line {number}: Spot depth update ID gap")
                if last_receive_ns is not None and (receive_ns <= last_receive_ns or receive_ns - last_receive_ns > max_gap_ms * 1_000_000):
                    raise CaptureRejected(f"line {number}: accepted frame receive gap exceeds policy")
                if last_mono_ns is not None and mono_ns <= last_mono_ns:
                    raise CaptureRejected(f"line {number}: monotonic clock regressed")
                if last_event_ns is not None and event_ns < last_event_ns:
                    raise CaptureRejected(f"line {number}: exchange event time regressed")
                changes: list[dict[str, str]] = []
                for side, key in (("BID", "b"), ("ASK", "a")):
                    for price, quantity in levels(frame.get(key), key, True, allow_empty=True):
                        changes.append({"side": side, "price": price, "quantity": quantity})
                if not changes:
                    raise CaptureRejected(f"line {number}: empty depth change")
                accepted.append({"event_ts_ns": event_ns, "receive_ts_ns": receive_ns,
                                 "first_update_id": first_id, "final_update_id": final_id,
                                 "changes": changes})
                last_id, last_event_ns, last_receive_ns, last_mono_ns = final_id, event_ns, receive_ns, mono_ns
                if len(accepted) > MAX_EVENTS:
                    raise CaptureRejected("depth event count exceeds limit")
            elif kind in {"gap_detected", "clock_discontinuity"}:
                raise CaptureRejected(f"line {number}: capture contains {kind}")
            elif kind == "session_disconnected":
                if not connected or disconnected or row.get("session_id") != session or row.get("reason") not in {"hourly_rotation", "local_shutdown"}:
                    raise CaptureRejected(f"line {number}: unclean session termination")
                disconnected = True
            elif kind == "capture_stopped":
                if not disconnected or stop is not None or row.get("reason") not in {"duration_elapsed", "sigterm"}:
                    raise CaptureRejected(f"line {number}: capture was not finalized cleanly")
                if row.get("observation_count") != frame_count:
                    raise CaptureRejected(f"line {number}: stop summary disagrees with frame count")
                stop = row
            else:
                raise CaptureRejected(f"line {number}: unsupported record type {kind!r}")

    if not (start and stop and connected and disconnected and snapshot and session and accepted and bridge_ns is not None):
        raise CaptureRejected("capture is incomplete or contains no snapshot-bridged events")
    if file_sha256(capture) != source_hash:
        raise CaptureRejected("source capture changed during conversion")
    start_ns_recorded = uint(start.get("recorded_at_unix_ns"), "capture start time", 1)
    stop_ns = uint(stop.get("recorded_at_unix_ns"), "capture stop time", 0)
    if uint(start.get("recorded_at_monotonic_ns"), "start monotonic time", 1) >= uint(stop.get("recorded_at_monotonic_ns"), "stop monotonic time", 0):
        raise CaptureRejected("monotonic capture lifecycle is invalid")
    if snapshot_ns > start_ns or start_ns_recorded > start_ns or stop_ns < end_ns:
        raise CaptureRejected("capture snapshot must precede the window and capture must continue through its end")
    bids = {p: Decimal(q) for p, q in snapshot["bids"]}
    asks = {p: Decimal(q) for p, q in snapshot["asks"]}
    selected: list[dict[str, Any]] = []
    for event in accepted:
        available_ns = max(event["event_ts_ns"], event["receive_ts_ns"])
        for change in event["changes"]:
            book = bids if change["side"] == "BID" else asks
            if Decimal(change["quantity"]) == 0:
                book.pop(change["price"], None)
            else:
                book[change["price"]] = Decimal(change["quantity"])
        if not bids or not asks or max(map(Decimal, bids)) >= min(map(Decimal, asks)):
            raise CaptureRejected("replayed order book is empty or crossed")
        # An exchange event timestamp can precede local receipt. If receipt
        # happens after the window opens, preserve the update as an in-window
        # event and apply it only at its point-in-time availability.
        if start_ns <= available_ns < end_ns:
            selected.append(event)
    if not selected:
        raise CaptureRejected("requested window contains no available depth events")

    # Roll the REST snapshot forward with only pre-window information. The
    # Dataset snapshot is thus the book available at the start boundary.
    warm_bids = {p: Decimal(q) for p, q in snapshot["bids"]}
    warm_asks = {p: Decimal(q) for p, q in snapshot["asks"]}
    warm_last_id = snapshot_id
    for event in accepted:
        if max(event["event_ts_ns"], event["receive_ts_ns"]) >= start_ns:
            break
        for change in event["changes"]:
            book = warm_bids if change["side"] == "BID" else warm_asks
            if Decimal(change["quantity"]) == 0:
                book.pop(change["price"], None)
            else:
                book[change["price"]] = Decimal(change["quantity"])
        warm_last_id = event["final_update_id"]
    snapshot = {**snapshot,
                "last_update_id": warm_last_id,
                "bids": [[p, format(q, "f")] for p, q in sorted(warm_bids.items(), key=lambda item: Decimal(item[0]), reverse=True)],
                "asks": [[p, format(q, "f")] for p, q in sorted(warm_asks.items(), key=lambda item: Decimal(item[0]))]}

    snapshot_record = {"schema": "td_market_data_spot_depth_snapshot", "version": 1,
                       "exchange": "BINANCE", "market": "SPOT", "symbol": symbol,
                       "session_id": session, **snapshot}
    event_records = [{"schema": "td_market_data_spot_depth_event", "version": 1,
                      "exchange": "BINANCE", "market": "SPOT", "symbol": symbol,
                      "session_id": session, **event} for event in selected]
    snapshot_bytes = canonical(snapshot_record)
    events_bytes = b"".join(canonical(row) + b"\n" for row in event_records)
    window = {"start_ns": start_ns, "end_ns": end_ns}
    logical = hashlib.sha256(b"td-market-data-spot-depth-dataset-v1\0" + canonical(window)
                             + snapshot_bytes + events_bytes).hexdigest()
    builder = source_revision()
    manifest = {"schema": "td_market_data_spot_depth_dataset", "version": 1,
                "exchange": "BINANCE", "market": "SPOT", "symbol": symbol,
                "session_id": session, "window": window, "event_count": len(event_records),
                "snapshot_file": "snapshot.json", "events_file": "events.jsonl",
                "snapshot_sha256": sha256(snapshot_bytes), "events_sha256": sha256(events_bytes),
                "source_capture_sha256": source_hash, "collector_code_sha256": start["collector_executable_sha256"],
                "collector_source_revision": start.get("source_revision"),
                "converter_code_sha256": file_sha256(Path(__file__).resolve()),
                "builder_source": builder, "logical_fingerprint_sha256": logical,
                "replay_usable": builder["tree_sha256"] is not None,
                "warnings": [] if builder["tree_sha256"] else ["builder source tree fingerprint is unavailable"]}
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        (staging / "snapshot.json").write_bytes(snapshot_bytes)
        (staging / "events.jsonl").write_bytes(events_bytes)
        (staging / "manifest.json").write_bytes(json.dumps(manifest, indent=2, ensure_ascii=False).encode() + b"\n")
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None or renameat2(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), str(output))
        fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
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
    except (CaptureRejected, OSError, ValueError) as error:
        raise SystemExit(f"Spot depth Dataset rejected: {error}") from error
    print(f"Spot depth Dataset published: events={manifest['event_count']} "
          f"replay_usable={manifest['replay_usable']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
