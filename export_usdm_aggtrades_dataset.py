#!/usr/bin/env python3
"""Publish a point-in-time USD-M aggregate-trades Dataset from raw captures."""

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
from capture_recovery import capture_id_for, recovery_provenance


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_DATASET_BYTES = 512 * 1024 * 1024
MAX_EVENTS = 2_000_000
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_OVERLAP_IDS = 100_000


class AggregateTradesRejected(ValueError):
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
    return int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1_000


def uint(value: Any, field: str, line: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AggregateTradesRejected(f"line {line}: {field} must be a nonnegative integer")
    return value


def decimal(value: Any, field: str, line: int, positive: bool = True) -> str:
    if not isinstance(value, str):
        raise AggregateTradesRejected(f"line {line}: {field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise AggregateTradesRejected(f"line {line}: invalid {field}") from error
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise AggregateTradesRejected(f"line {line}: invalid {field}")
    return value


def publish(captures: list[Path], output: Path, symbol: str,
            start_ns: int, end_ns: int) -> dict[str, Any]:
    if not captures or not output.is_absolute() or start_ns >= end_ns:
        raise AggregateTradesRejected("supply captures, an absolute output, and a positive time window")
    if (start_ns % 1_000_000 or end_ns % 1_000_000
            or not symbol.isascii() or not symbol.isalnum()):
        raise AggregateTradesRejected("window must be millisecond-aligned and symbol alphanumeric")
    resolved = [path.resolve(strict=True) for path in captures]
    if len(set(resolved)) != len(resolved):
        raise AggregateTradesRejected("duplicate capture paths are not allowed")
    for path in resolved:
        if path.stat().st_size > MAX_CAPTURE_BYTES:
            raise AggregateTradesRejected(f"capture exceeds 256 MiB: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    repository = Path(__file__).resolve().parent
    if output.exists() or output == repository or repository in output.parents:
        raise AggregateTradesRejected("output exists or is inside the source repository")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        return _publish(resolved, output, parent, stage, symbol.upper(), start_ns, end_ns)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _publish(captures: list[Path], output: Path, parent: Path, stage: Path,
             symbol: str, start_ns: int, end_ns: int) -> dict[str, Any]:
    start_ms, end_ms = start_ns // 1_000_000, end_ns // 1_000_000
    logical = hashlib.sha256(b"td-market-data-usdm-aggtrades-v1\0" + canonical({
        "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns}))
    events_hash = hashlib.sha256()
    event_count = event_bytes = observation_count = 0
    previous_id: int | None = None
    previous_event_ms: int | None = None
    previous_trade_ms: int | None = None
    previous_received_ns: int | None = None
    previous_mono_ns: int | None = None
    previous_available_ns: int | None = None
    previous_inside_capture_id: str | None = None
    recent_frame_hashes: dict[int, str] = {}
    first_inside: dict[str, Any] | None = None
    last_inside: dict[str, Any] | None = None
    pre_window_evidence: dict[str, Any] | None = None
    post_window_evidence: dict[str, Any] | None = None
    has_pre = has_post = False
    max_receive_gap_ms = 0
    max_dataset_receive_gap_ms = 0
    previous_inside_receive_ns: int | None = None
    capture_sources: list[dict[str, Any]] = []
    chain_start_ns: int | None = None
    chain_stop_ns: int | None = None
    last_capture_started_ns: int | None = None
    deduplicated_overlap_count = 0

    with (stage / "events.jsonl").open("xb", buffering=0) as target:
        for capture in captures:
            source_hash = file_sha256(capture)
            capture_id = capture_id_for(capture, source_hash)
            recovery = recovery_provenance(capture, source_hash, capture_id)
            recovered_at_ns = recovery["recovered_at_unix_ns"] if recovery else 0
            start: dict[str, Any] | None = None
            stop: dict[str, Any] | None = None
            session: str | None = None
            connected = disconnected = False
            connect_ns: int | None = None
            disconnect_ns: int | None = None
            capture_max_gap_ms = 0
            capture_previous_received_ns: int | None = None
            capture_previous_mono_ns: int | None = None
            capture_previous_event_ms: int | None = None
            capture_previous_trade_ms: int | None = None
            capture_previous_id: int | None = None
            frames = 0
            bytes_seen = 0
            with capture.open("rb") as source:
                for line_number, raw in enumerate(source, 1):
                    before_record = bytes_seen
                    bytes_seen += len(raw)
                    if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                        raise AggregateTradesRejected(f"{capture}:{line_number}: truncated/oversized JSONL")
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise AggregateTradesRejected(f"{capture}:{line_number}: invalid JSON") from error
                    if (not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record"
                            or row.get("version") != 1):
                        raise AggregateTradesRejected(f"{capture}:{line_number}: unsupported capture schema")
                    kind = row.get("record_type")
                    if line_number == 1:
                        if (kind != "capture_started" or row.get("stream") != "usdm-aggtrades"
                                or row.get("symbol") != symbol
                                or row.get("source_url") != f"wss://fstream.binance.com/market/ws/{symbol.lower()}@aggTrade"):
                            raise AggregateTradesRejected(f"capture header identity/source mismatch: {capture}")
                        start = row
                        started_ns = uint(row.get("recorded_at_unix_ns"), "capture start", line_number)
                        capture_max_gap_ms = uint(row.get("max_gap_ms"), "capture max gap", line_number)
                        code_hash = row.get("collector_executable_sha256")
                        revision = row.get("source_revision")
                        if capture_max_gap_ms <= 0 or not isinstance(revision, str) or not revision:
                            raise AggregateTradesRejected(f"capture source policy/revision invalid: {capture}")
                        if (not isinstance(code_hash, str) or len(code_hash) != 64
                                or any(char not in "0123456789abcdef" for char in code_hash)):
                            raise AggregateTradesRejected(f"capture code hash invalid: {capture}")
                        if last_capture_started_ns is not None and started_ns <= last_capture_started_ns:
                            raise AggregateTradesRejected("capture chain is duplicated or out of order")
                        if chain_start_ns is None or started_ns < chain_start_ns:
                            chain_start_ns = started_ns
                        last_capture_started_ns = started_ns
                    elif kind == "session_connected":
                        if connected or disconnected:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: reconnect in capture")
                        session = row.get("session_id")
                        connect_ns = uint(row.get("recorded_at_unix_ns"), "connect time", line_number)
                        if (not isinstance(session, str) or not session
                                or row.get("url") != start["source_url"]):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: invalid session identity")
                        connected = True
                    elif kind == "market_frame":
                        frames += 1
                        observation_count += 1
                        if not connected or disconnected or row.get("session_id") != session:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: frame outside session")
                        raw_frame = row.get("raw_frame")
                        if (not isinstance(raw_frame, str)
                                or hashlib.sha256(raw_frame.encode()).hexdigest() != row.get("raw_frame_sha256")):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: raw frame hash mismatch")
                        try:
                            frame = json.loads(raw_frame)
                        except json.JSONDecodeError as error:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: invalid raw frame") from error
                        if not isinstance(frame, dict) or frame.get("e") != "aggTrade" or frame.get("s") != symbol:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: exchange frame identity mismatch")
                        trade_id = uint(frame.get("a"), "aggregate trade ID", line_number)
                        first_trade_id = uint(frame.get("f"), "first trade ID", line_number)
                        last_trade_id = uint(frame.get("l"), "last trade ID", line_number)
                        event_ms = uint(frame.get("E"), "exchange event time", line_number)
                        trade_ms = uint(frame.get("T"), "exchange trade time", line_number)
                        receive_ns = uint(row.get("received_at_unix_ns"), "local receive time", line_number)
                        receive_mono = uint(row.get("receive_monotonic_ns"), "monotonic receive time", line_number)
                        if (first_trade_id > last_trade_id
                                or row.get("trade_id") != trade_id or row.get("first_trade_id") != first_trade_id
                                or row.get("last_trade_id") != last_trade_id or row.get("event_time_ms") != event_ms
                                or row.get("trade_time_ms") != trade_ms or not isinstance(frame.get("m"), bool)
                                or row.get("buyer_is_maker") != frame["m"]
                                or row.get("accepted_for_sequence") is not True):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: parsed fields differ from raw frame")
                        if receive_ns < (connect_ns or 0) or event_ms * 1_000_000 > receive_ns + 2_000_000_000:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: invalid event/receive ordering")
                        if capture_previous_id is not None and trade_id <= capture_previous_id:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: IDs regress within capture")
                        if capture_previous_event_ms is not None and event_ms < capture_previous_event_ms:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: exchange event time regressed")
                        if capture_previous_trade_ms is not None and trade_ms < capture_previous_trade_ms:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: trade time regressed")
                        if capture_previous_received_ns is not None:
                            if receive_ns <= capture_previous_received_ns or receive_mono <= capture_previous_mono_ns:
                                raise AggregateTradesRejected(f"{capture}:{line_number}: receive clock regressed")
                            wall_delta = receive_ns - capture_previous_received_ns
                            mono_delta = receive_mono - capture_previous_mono_ns
                            max_receive_gap_ms = max(max_receive_gap_ms, wall_delta // 1_000_000)
                            if (mono_delta > capture_max_gap_ms * 1_000_000
                                    or abs(wall_delta - mono_delta) > 2_000_000_000):
                                raise AggregateTradesRejected(f"{capture}:{line_number}: receive gap/clock discontinuity")
                        price = decimal(frame.get("p"), "price", line_number)
                        quantity = decimal(frame.get("q"), "quantity", line_number)
                        if frame["m"]:
                            aggressor = "SELL"
                        else:
                            aggressor = "BUY"
                        event_ns = trade_ms * 1_000_000
                        frame_hash = row["raw_frame_sha256"]
                        if previous_id is not None and trade_id <= previous_id:
                            known_hash = recent_frame_hashes.get(trade_id)
                            if known_hash is None:
                                raise AggregateTradesRejected(f"{capture}:{line_number}: overlap exceeds the bounded deduplication window")
                            if known_hash != frame_hash:
                                raise AggregateTradesRejected(f"{capture}:{line_number}: conflicting duplicate aggregate trade ID")
                            deduplicated_overlap_count += 1
                            capture_previous_id = trade_id
                            capture_previous_event_ms = event_ms
                            capture_previous_trade_ms = trade_ms
                            capture_previous_received_ns = receive_ns
                            capture_previous_mono_ns = receive_mono
                            previous_max_gap_ms = capture_max_gap_ms
                            continue
                        if previous_id is not None and trade_id != previous_id + 1:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: aggregate trade ID gap")
                        if previous_event_ms is not None and event_ms < previous_event_ms:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: exchange event time regressed")
                        if previous_trade_ms is not None and trade_ms < previous_trade_ms:
                            raise AggregateTradesRejected(f"{capture}:{line_number}: trade time regressed")
                        normalized = {"record_type": "aggregate_trade", "symbol": symbol,
                                      "capture_id": capture_id,
                                      "aggregate_trade_id": trade_id, "first_trade_id": first_trade_id,
                                      "last_trade_id": last_trade_id, "exchange_event_time_ns": event_ms * 1_000_000,
                                      "trade_time_ns": event_ns, "received_at_ns": receive_ns,
                                      "available_at_ns": max(receive_ns, event_ms * 1_000_000, event_ns, recovered_at_ns,
                                                              previous_available_ns or 0),
                                      "received_monotonic_ns": receive_mono, "price": price,
                                      "quantity": quantity, "buyer_is_maker": frame["m"],
                                      "aggressor_side": aggressor,
                                      "raw_frame_sha256": row["raw_frame_sha256"]}
                        if trade_ms < start_ms:
                            has_pre = True
                            pre_window_evidence = pre_window_evidence or {
                                "aggregate_trade_id": trade_id, "trade_time_ns": event_ns,
                                "received_at_ns": receive_ns, "raw_frame_sha256": row["raw_frame_sha256"]}
                        elif trade_ms >= end_ms:
                            has_post = True
                            post_window_evidence = post_window_evidence or {
                                "aggregate_trade_id": trade_id, "trade_time_ns": event_ns,
                                "received_at_ns": receive_ns, "raw_frame_sha256": row["raw_frame_sha256"]}
                        else:
                            if previous_inside_receive_ns is not None and previous_inside_capture_id == capture_id:
                                max_dataset_receive_gap_ms = max(
                                    max_dataset_receive_gap_ms,
                                    (receive_ns - previous_inside_receive_ns) // 1_000_000)
                            previous_inside_receive_ns = receive_ns
                            previous_inside_capture_id = capture_id
                            encoded = canonical(normalized) + b"\n"
                            event_bytes += len(encoded)
                            if event_bytes > MAX_DATASET_BYTES:
                                raise AggregateTradesRejected("normalized Dataset exceeds 512 MiB")
                            target.write(encoded)
                            events_hash.update(encoded)
                            logical.update(encoded)
                            first_inside = first_inside or normalized
                            last_inside = normalized
                            event_count += 1
                            previous_available_ns = normalized["available_at_ns"]
                        recent_frame_hashes[trade_id] = frame_hash
                        cutoff = trade_id - MAX_OVERLAP_IDS
                        if len(recent_frame_hashes) > MAX_OVERLAP_IDS * 2:
                            recent_frame_hashes = {key: value for key, value in recent_frame_hashes.items()
                                                   if key >= cutoff}
                        previous_id, previous_event_ms, previous_trade_ms = trade_id, event_ms, trade_ms
                        previous_received_ns, previous_mono_ns = receive_ns, receive_mono
                        capture_previous_id = trade_id
                        capture_previous_event_ms = event_ms
                        capture_previous_trade_ms = trade_ms
                        capture_previous_received_ns = receive_ns
                        capture_previous_mono_ns = receive_mono
                        previous_max_gap_ms = capture_max_gap_ms
                    elif kind == "session_disconnected":
                        if (not connected or disconnected or row.get("session_id") != session
                                or row.get("reason") not in {"hourly_rotation", "local_shutdown"}):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: unclean session stop")
                        disconnect_ns = uint(row.get("recorded_at_unix_ns"), "disconnect time", line_number)
                        if (capture_previous_received_ns is not None
                                and disconnect_ns < capture_previous_received_ns):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: disconnect predates last frame")
                        disconnected = True
                    elif kind == "capture_stopped":
                        stop = row
                        stopped_ns = uint(row.get("recorded_at_unix_ns"), "capture stop", line_number)
                        if (not disconnected or row.get("reason") not in {"duration_elapsed", "sigterm"}
                                or row.get("observation_count") != frames or row.get("gap_count") != 0
                                or row.get("disconnect_count") != 0 or row.get("bytes_before_stop_record") != before_record
                                or stopped_ns < (disconnect_ns or 0)):
                            raise AggregateTradesRejected(f"{capture}:{line_number}: capture footer mismatch")
                        if chain_stop_ns is None or stopped_ns > chain_stop_ns:
                            chain_stop_ns = stopped_ns
                    elif kind in {"gap_detected", "clock_discontinuity"}:
                        raise AggregateTradesRejected(f"{capture}:{line_number}: capture contains {kind}")
                    else:
                        raise AggregateTradesRejected(f"{capture}:{line_number}: unsupported record {kind!r}")
            if not (start and stop and connected and disconnected and frames > 0):
                raise AggregateTradesRejected(f"capture incomplete or empty: {capture}")
            if file_sha256(capture) != source_hash:
                raise AggregateTradesRejected(f"capture changed while exporting: {capture}")
            capture_sources.append({"path": str(capture), "sha256": source_hash,
                                    "capture_id": capture_id,
                                    "capture_started_at_ns": uint(start["recorded_at_unix_ns"], "capture start", 1),
                                    "capture_stopped_at_ns": uint(stop["recorded_at_unix_ns"], "capture stop", 1),
                                    "collector_code_sha256": start["collector_executable_sha256"],
                            "collector_source_revision": start["source_revision"],
                            "origin": "vps_recovered" if recovery else "local",
                            "recovered_at_unix_ns": recovered_at_ns or None,
                                    "observation_count": frames})
        target.flush()
        os.fsync(target.fileno())

    if not event_count or not has_pre or not has_post:
        raise AggregateTradesRejected("captures lack in-window events or pre/post window trade coverage")
    if chain_start_ns is None or chain_stop_ns is None or chain_start_ns > start_ns or chain_stop_ns < end_ns:
        raise AggregateTradesRejected("capture chain lifecycle does not cover the requested window")
    if previous_received_ns is None or previous_max_gap_ms is None:
        raise AggregateTradesRejected("capture chain has no aggregate trade observations")
    builder = source_revision()
    manifest = {"schema": "td_market_data_usdm_aggtrades_dataset", "version": 1,
                "exchange": "BINANCE", "market": "USD_M", "symbol": symbol,
                "window": {"start_ns": start_ns, "end_ns": end_ns},
                "event_count": event_count, "first_aggregate_trade_id": first_inside["aggregate_trade_id"],
                "last_aggregate_trade_id": last_inside["aggregate_trade_id"],
                "first_trade_time_ns": first_inside["trade_time_ns"],
                "last_trade_time_ns": last_inside["trade_time_ns"],
                "first_receive_time_ns": first_inside["received_at_ns"],
                "last_receive_time_ns": last_inside["received_at_ns"],
                "max_receive_gap_ms": max_dataset_receive_gap_ms,
                "max_capture_chain_receive_gap_ms": max_receive_gap_ms,
                "deduplicated_overlap_count": deduplicated_overlap_count,
                "overlap_policy": "identical_raw_frame_sha256_required_bounded_100000_ids",
                "boundary_coverage": {"pre_window_trade": pre_window_evidence,
                                      "post_window_trade": post_window_evidence},
                "events_file": "events.jsonl", "events_sha256": events_hash.hexdigest(),
                "logical_fingerprint_sha256": logical.hexdigest(),
                "capture_sources": capture_sources,
                "source_capture_set_sha256": hashlib.sha256(canonical(
                    [item["sha256"] for item in capture_sources])).hexdigest(),
                "collector_code_sha256": sorted({item["collector_code_sha256"] for item in capture_sources}),
                "converter_code_sha256": file_sha256(Path(__file__).resolve()),
                "builder_source": builder, "replay_usable": builder["tree_sha256"] is not None}
    with (stage / "manifest.json").open("xb") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    for source in capture_sources:
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise AggregateTradesRejected(f"source capture changed during publication: {source['path']}")
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
    parser.add_argument("--capture", action="append", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = publish(args.capture, args.output, args.symbol.upper(), args.start, args.end)
    except (AggregateTradesRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M aggregate-trades Dataset rejected: {error}") from error
    print(f"USD-M aggregate-trades Dataset published: events={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
