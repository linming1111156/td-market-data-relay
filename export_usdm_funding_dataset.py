#!/usr/bin/env python3
"""Publish finalized USD-M Funding matched to point-in-time Mark schedules."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from export_spot_depth_dataset import canonical, file_sha256, source_revision


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 10_000
FUNDING_SCHEDULE_ALIGNMENT_TOLERANCE_MS = 1_000


class FundingDatasetRejected(ValueError):
    pass


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None or renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def uint(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FundingDatasetRejected(f"{field} must be a nonnegative integer")
    return value


def finite_decimal(value: Any, field: str, *, positive: bool = False) -> str:
    if not isinstance(value, str):
        raise FundingDatasetRejected(f"{field} must be a decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise FundingDatasetRejected(f"invalid {field}") from error
    if not number.is_finite() or (positive and number <= 0):
        raise FundingDatasetRejected(f"invalid {field}")
    return value


def capture_records(path: Path) -> tuple[list[dict[str, Any]], str, int]:
    path = path.resolve(strict=True)
    if path.stat().st_size > MAX_CAPTURE_BYTES:
        raise FundingDatasetRejected(f"capture exceeds 256 MiB: {path}")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    byte_count = 0
    bytes_before_footer = 0
    with path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                raise FundingDatasetRejected(f"{path}: line {line} is truncated or oversized")
            digest.update(raw)
            bytes_before_footer = byte_count
            byte_count += len(raw)
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FundingDatasetRejected(f"{path}: line {line} is invalid JSON") from error
            if (not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record"
                    or row.get("version") != 1):
                raise FundingDatasetRejected(f"{path}: line {line} has unsupported capture schema")
            rows.append(row)
            if len(rows) > 2_000_000:
                raise FundingDatasetRejected(f"capture record count exceeds safety bound: {path}")
    if not rows:
        raise FundingDatasetRejected(f"capture is empty: {path}")
    return rows, digest.hexdigest(), bytes_before_footer


def lifecycle(rows: list[dict[str, Any]], stream: str, symbol: str, path: Path,
              bytes_before_footer: int) -> dict[str, Any]:
    header, footer = rows[0], rows[-1]
    if (header.get("record_type") != "capture_started" or header.get("stream") != stream
            or header.get("symbol") != symbol):
        raise FundingDatasetRejected(f"capture identity mismatch: {path}")
    expected_source = ("https://fapi.binance.com/fapi/v1/fundingRate" if stream == "usdm-finalized-funding"
                       else f"wss://fstream.binance.com/market/ws/{symbol.lower()}@markPrice@1s")
    if header.get("source_url") != expected_source:
        raise FundingDatasetRejected(f"capture source URL mismatch: {path}")
    if footer.get("record_type") != "capture_stopped" or footer.get("reason") not in {"duration_elapsed", "sigterm"}:
        raise FundingDatasetRejected(f"capture lacks a clean stop: {path}")
    started_ns = uint(header.get("recorded_at_unix_ns"), "capture start")
    stopped_ns = uint(footer.get("recorded_at_unix_ns"), "capture stop")
    if stopped_ns < started_ns or not isinstance(header.get("source_revision"), str):
        raise FundingDatasetRejected(f"capture lifecycle/source revision is invalid: {path}")
    source_hash = header.get("collector_executable_sha256")
    if (not isinstance(source_hash, str) or len(source_hash) != 64
            or any(char not in "0123456789abcdef" for char in source_hash)):
        raise FundingDatasetRejected(f"capture code fingerprint is invalid: {path}")
    if footer.get("bytes_before_stop_record") != bytes_before_footer:
        raise FundingDatasetRejected(f"capture stop byte count mismatch: {path}")
    return {"header": header, "footer": footer, "started_ns": started_ns,
            "stopped_ns": stopped_ns, "collector_code_sha256": source_hash,
            "source_revision": header["source_revision"]}


def mark_observations(paths: list[Path], symbol: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for path in sorted((item.resolve(strict=True) for item in paths), key=lambda item: item.name):
        rows, capture_hash, bytes_before_footer = capture_records(path)
        life = lifecycle(rows, "usdm-funding-mark", symbol, path, bytes_before_footer)
        connected = disconnected = False
        session: str | None = None
        previous_received_ns = previous_mono_ns = previous_event_ms = None
        connect_ns: int | None = None
        disconnect_ns: int | None = None
        count = 0
        max_gap_ms = uint(life["header"].get("max_gap_ms"), "Mark max gap")
        if max_gap_ms <= 0:
            raise FundingDatasetRejected("Mark capture max gap must be positive")
        for index, row in enumerate(rows[1:-1], 2):
            kind = row.get("record_type")
            if kind == "session_connected":
                if connected or disconnected:
                    raise FundingDatasetRejected(f"Mark capture reconnect is not admissible: {path}")
                session = row.get("session_id")
                if not isinstance(session, str) or not session:
                    raise FundingDatasetRejected(f"Mark session identity missing: {path}")
                expected_url = f"wss://fstream.binance.com/market/ws/{symbol.lower()}@markPrice@1s"
                if row.get("url") != expected_url:
                    raise FundingDatasetRejected(f"Mark WebSocket URL mismatch: {path}")
                connect_ns = uint(row.get("recorded_at_unix_ns"), "Mark connect time")
                if connect_ns < life["started_ns"]:
                    raise FundingDatasetRejected(f"Mark connection predates capture start: {path}")
                connected = True
            elif kind == "market_frame":
                if not connected or disconnected or row.get("session_id") != session:
                    raise FundingDatasetRejected(f"Mark event outside its session: {path}:{index}")
                raw = row.get("raw_frame")
                if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_frame_sha256"):
                    raise FundingDatasetRejected(f"Mark raw frame hash mismatch: {path}:{index}")
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise FundingDatasetRejected(f"Mark raw frame is invalid JSON: {path}:{index}") from error
                if not isinstance(frame, dict) or frame.get("e") != "markPriceUpdate" or frame.get("s") != symbol:
                    raise FundingDatasetRejected(f"Mark raw frame identity mismatch: {path}:{index}")
                event_ms = uint(frame.get("E"), "Mark event time")
                funding_ms = uint(frame.get("T"), "Mark next Funding time")
                received_ns = uint(row.get("received_at_unix_ns"), "Mark receive time")
                mono_ns = uint(row.get("receive_monotonic_ns"), "Mark monotonic receive time")
                if connect_ns is None or received_ns < connect_ns:
                    raise FundingDatasetRejected(f"Mark event predates its session connection: {path}:{index}")
                if row.get("event_time_ms") != event_ms or row.get("next_funding_time_ms") != funding_ms:
                    raise FundingDatasetRejected(f"Mark parsed fields differ from raw frame: {path}:{index}")
                if event_ms * 1_000_000 > received_ns + 2_000_000_000:
                    raise FundingDatasetRejected(f"Mark event is ahead of local receive time: {path}:{index}")
                if previous_event_ms is not None and event_ms <= previous_event_ms:
                    raise FundingDatasetRejected(f"Mark exchange event time regressed: {path}:{index}")
                if previous_received_ns is not None:
                    if received_ns <= previous_received_ns or mono_ns <= previous_mono_ns:
                        raise FundingDatasetRejected(f"Mark receive clocks regressed: {path}:{index}")
                    wall_delta = received_ns - previous_received_ns
                    mono_delta = mono_ns - previous_mono_ns
                    if mono_delta > max_gap_ms * 1_000_000 or abs(wall_delta - mono_delta) > 2_000_000_000:
                        raise FundingDatasetRejected(f"Mark receive gap/clock discontinuity: {path}:{index}")
                for key in ("p", "i"):
                    finite_decimal(frame.get(key), f"Mark {key}", positive=True)
                finite_decimal(frame.get("r"), "Mark funding rate")
                observations.append({"event_time_ms": event_ms, "funding_time_ms": funding_ms,
                                     "received_at_ns": received_ns, "session_id": session,
                                     "payload_sha256": row["raw_frame_sha256"]})
                previous_event_ms, previous_received_ns, previous_mono_ns = event_ms, received_ns, mono_ns
                count += 1
            elif kind == "session_disconnected":
                if (not connected or disconnected or row.get("session_id") != session
                        or row.get("reason") not in {"hourly_rotation", "local_shutdown"}):
                    raise FundingDatasetRejected(f"Mark session did not stop cleanly: {path}")
                disconnect_ns = uint(row.get("recorded_at_unix_ns"), "Mark disconnect time")
                if previous_received_ns is None or disconnect_ns < previous_received_ns:
                    raise FundingDatasetRejected(f"Mark disconnect predates its last event: {path}")
                disconnected = True
            else:
                raise FundingDatasetRejected(f"Mark capture has inadmissible record {kind!r}: {path}")
        if (not connected or not disconnected or count <= 0
                or life["footer"].get("observation_count") != count
                or disconnect_ns is None or life["stopped_ns"] < disconnect_ns):
            raise FundingDatasetRejected(f"Mark capture is incomplete or count-mismatched: {path}")
        provenance.append({"path": str(path), "sha256": capture_hash,
                           "capture_started_at_ns": life["started_ns"],
                           "capture_stopped_at_ns": life["stopped_ns"],
                           "collector_code_sha256": life["collector_code_sha256"],
                           "source_revision": life["source_revision"], "record_count": count})
    observations.sort(key=lambda item: item["received_at_ns"])
    if not observations:
        raise FundingDatasetRejected("no Mark Price observations were supplied")
    for previous, current in zip(observations, observations[1:]):
        if current["received_at_ns"] <= previous["received_at_ns"]:
            raise FundingDatasetRejected("Mark observations overlap or duplicate receive times")
    return observations, provenance


def expected_schedule(observations: list[dict[str, Any]], start_ns: int, end_ns: int,
                      max_pre_settlement_age_ms: int) -> list[dict[str, Any]]:
    max_gap_ns = max_pre_settlement_age_ms * 1_000_000
    if (observations[0]["received_at_ns"] > start_ns
            or observations[-1]["received_at_ns"] < end_ns - max_gap_ns):
        raise FundingDatasetRejected(
            "Mark observations do not cover the requested window edges within the receive-gap bound"
        )
    receive_gaps = [right["received_at_ns"] - left["received_at_ns"]
                    for left, right in zip(observations, observations[1:])]
    largest_receive_gap_ns = max(receive_gaps, default=0)
    if largest_receive_gap_ns > max_gap_ns:
        raise FundingDatasetRejected("Mark capture chain has a receive-time gap in the requested window")
    by_funding_time: dict[int, list[dict[str, Any]]] = {}
    for observation in observations:
        settlement_ns = observation["funding_time_ms"] * 1_000_000
        if start_ns <= settlement_ns < end_ns and observation["received_at_ns"] < settlement_ns:
            by_funding_time.setdefault(observation["funding_time_ms"], []).append(observation)
    schedule: list[dict[str, Any]] = []
    for funding_time_ms, evidence in sorted(by_funding_time.items()):
        latest = max(evidence, key=lambda item: item["received_at_ns"])
        age_ns = funding_time_ms * 1_000_000 - latest["received_at_ns"]
        if age_ns <= 0 or age_ns > max_pre_settlement_age_ms * 1_000_000:
            raise FundingDatasetRejected(f"pre-settlement Mark evidence is stale for {funding_time_ms}")
        schedule.append({"funding_time_ms": funding_time_ms,
                         "latest_mark_received_at_ns": latest["received_at_ns"],
                         "mark_observation_age_ns": age_ns,
                         "mark_session_id": latest["session_id"],
                         "mark_payload_sha256": latest["payload_sha256"]})
    # A valid short research window can contain no scheduled settlement. The
    # continuous Mark and REST capture checks below still prove that the window
    # was observed; publish an empty event set instead of rejecting it.
    return schedule


def funding_rows(paths: list[Path], symbol: str, schedule: list[dict[str, Any]],
                 max_poll_gap_ms: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {row["funding_time_ms"] for row in schedule}
    received_rows: dict[int, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    pages: list[tuple[int, str]] = []
    first_cursor: int | None = None
    last_cursor_ms: int | None = None
    for path in sorted((item.resolve(strict=True) for item in paths), key=lambda item: item.name):
        rows, capture_hash, bytes_before_footer = capture_records(path)
        life = lifecycle(rows, "usdm-finalized-funding", symbol, path, bytes_before_footer)
        cursor_record: dict[str, Any] | None = None
        page_count = 0
        last_page_received_ns: int | None = None
        for index, row in enumerate(rows[1:-1], 2):
            kind = row.get("record_type")
            if kind == "funding_cursor_started":
                if cursor_record is not None or row.get("future_only") is not True:
                    raise FundingDatasetRejected(f"Funding future-only cursor record is invalid: {path}")
                cursor_record = row
                start_cursor = uint(row.get("request_start_time_ms"), "Funding start cursor")
                cursor_recorded_ns = uint(row.get("recorded_at_unix_ns"), "Funding cursor record time")
                if cursor_recorded_ns < life["started_ns"] or cursor_recorded_ns > life["stopped_ns"]:
                    raise FundingDatasetRejected(f"Funding cursor record is outside capture lifecycle: {path}")
                if first_cursor is None:
                    first_cursor = start_cursor
                elif start_cursor < last_cursor_ms:
                    raise FundingDatasetRejected("Funding cursor regressed across capture rotation")
                last_cursor_ms = start_cursor
            elif kind == "funding_api_page":
                if cursor_record is None:
                    raise FundingDatasetRejected(f"Funding API page precedes cursor record: {path}")
                raw = row.get("raw_response")
                if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != row.get("raw_response_sha256"):
                    raise FundingDatasetRejected(f"Funding REST response hash mismatch: {path}:{index}")
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise FundingDatasetRejected(f"Funding REST response is invalid JSON: {path}:{index}") from error
                if decoded != row.get("funding_rows") or not isinstance(decoded, list):
                    raise FundingDatasetRejected(f"Funding parsed rows differ from raw response: {path}:{index}")
                received_ns = uint(row.get("received_at_unix_ns"), "Funding API receive time")
                if (received_ns < life["started_ns"] or received_ns > life["stopped_ns"]
                        or (last_page_received_ns is not None and received_ns <= last_page_received_ns)):
                    raise FundingDatasetRejected(f"Funding API page time is outside capture lifecycle: {path}:{index}")
                last_page_received_ns = received_ns
                cursor_ms = uint(row.get("request_start_time_ms"), "Funding request cursor")
                if cursor_ms < cursor_record["request_start_time_ms"] or (
                    last_cursor_ms is not None and cursor_ms < last_cursor_ms
                ):
                    raise FundingDatasetRejected(f"Funding API request cursor regressed: {path}:{index}")
                last_cursor_ms = cursor_ms
                expected_url = ("https://fapi.binance.com/fapi/v1/fundingRate?"
                                + urllib.parse.urlencode(
                                    {"symbol": symbol, "startTime": cursor_ms, "limit": 1000}))
                if row.get("request_url") != expected_url or row.get("record_count") != len(decoded):
                    raise FundingDatasetRejected(f"Funding request identity/count mismatch: {path}:{index}")
                for item in decoded:
                    if not isinstance(item, dict) or item.get("symbol") != symbol:
                        raise FundingDatasetRejected(f"Funding settlement symbol mismatch: {path}:{index}")
                    time_ms = uint(item.get("fundingTime"), "Funding settlement time")
                    if time_ms < cursor_ms or received_ns < time_ms * 1_000_000:
                        raise FundingDatasetRejected(f"Funding row violates cursor or availability time: {path}:{index}")
                    rate = finite_decimal(item.get("fundingRate"), "finalized funding rate")
                    mark = item.get("markPrice")
                    if mark is not None:
                        finite_decimal(mark, "settlement mark price", positive=True)
                    matches = [scheduled_ms for scheduled_ms in expected
                               if abs(time_ms - scheduled_ms) <= FUNDING_SCHEDULE_ALIGNMENT_TOLERANCE_MS]
                    if len(matches) > 1:
                        raise FundingDatasetRejected(
                            f"finalized Funding timestamp ambiguously matches Mark schedule: {time_ms}"
                        )
                    if matches:
                        scheduled_ms = matches[0]
                        normalized = {"funding_time_ms": time_ms,
                                      "scheduled_funding_time_ms": scheduled_ms,
                                      "funding_time_offset_ms": time_ms - scheduled_ms,
                                      "available_at_ns": received_ns,
                                      "funding_rate": rate, "mark_price": mark,
                                      "rate_type": item.get("rateType", "unknown"),
                                      "api_response_sha256": row["raw_response_sha256"],
                                      "raw_funding_row": item}
                        prior = received_rows.get(scheduled_ms)
                        if prior is not None and prior["raw_funding_row"] != item:
                            raise FundingDatasetRejected(
                                f"conflicting finalized Funding rows for scheduled time {scheduled_ms}"
                            )
                        if prior is None or received_ns < prior["available_at_ns"]:
                            received_rows[scheduled_ms] = normalized
                pages.append((received_ns, row["raw_response_sha256"]))
                page_count += 1
            else:
                raise FundingDatasetRejected(f"Funding capture has inadmissible record {kind!r}: {path}")
        if cursor_record is None or page_count <= 0:
            raise FundingDatasetRejected(f"Funding capture lacks a cursor or API pages: {path}")
        if life["footer"].get("gap_count") != 0 or life["footer"].get("disconnect_count") != 0:
            raise FundingDatasetRejected(f"Funding capture stop reports gaps/disconnects: {path}")
        provenance.append({"path": str(path), "sha256": capture_hash,
                           "capture_started_at_ns": life["started_ns"],
                           "capture_stopped_at_ns": life["stopped_ns"],
                           "collector_code_sha256": life["collector_code_sha256"],
                           "source_revision": life["source_revision"],
                           "initial_request_start_time_ms": cursor_record["request_start_time_ms"],
                           "page_count": page_count})
    if not pages:
        raise FundingDatasetRejected("no finalized Funding API pages were supplied")
    pages.sort()
    if any(right[0] <= left[0] for left, right in zip(pages, pages[1:])):
        raise FundingDatasetRejected("Funding API receive timestamps are duplicated or regressed")
    max_poll_gap_ns = max((right[0] - left[0] for left, right in zip(pages, pages[1:])), default=0)
    if max_poll_gap_ns > max_poll_gap_ms * 1_000_000:
        raise FundingDatasetRejected("Funding API poll chain has a receive-time gap")
    missing = sorted(expected - set(received_rows))
    extra = sorted(set(received_rows) - expected)
    if missing or extra:
        raise FundingDatasetRejected(f"finalized Funding does not exactly match Mark schedule; missing={missing}, extra={extra}")
    return [received_rows[key] for key in sorted(received_rows)], provenance


def publish(mark_paths: list[Path], funding_paths: list[Path], output: Path, symbol: str,
            start_ns: int, end_ns: int, max_pre_settlement_age_ms: int = 5_000,
            max_poll_gap_ms: int = 120_000) -> dict[str, Any]:
    if not output.is_absolute() or start_ns >= end_ns or start_ns % 1_000_000 or end_ns % 1_000_000:
        raise FundingDatasetRejected("output must be absolute and the millisecond window positive")
    if not symbol.isascii() or not symbol.isalnum() or max_pre_settlement_age_ms <= 0 or max_poll_gap_ms <= 0:
        raise FundingDatasetRejected("symbol and observation/poll bounds are invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    repo = Path(__file__).resolve().parent
    if output.exists() or output == repo or repo in output.parents:
        raise FundingDatasetRejected("Dataset output exists or is inside the source repository")
    mark_obs, mark_sources = mark_observations(mark_paths, symbol)
    schedule = expected_schedule(mark_obs, start_ns, end_ns, max_pre_settlement_age_ms)
    settlements, funding_sources = funding_rows(funding_paths, symbol, schedule, max_poll_gap_ms)
    if len(settlements) > MAX_EVENTS:
        raise FundingDatasetRejected("Funding settlement count exceeds safety bound")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        events_path = stage / "events.jsonl"
        events_hash = hashlib.sha256()
        logical = hashlib.sha256(b"td-market-data-usdm-funding-dataset-v1\0" + canonical({
            "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns, "schedule": schedule}))
        with events_path.open("xb", buffering=0) as stream:
            for settlement in settlements:
                record = {"record_type": "funding_settlement", "symbol": symbol,
                          "funding_time_ms": settlement["funding_time_ms"],
                          "funding_time_ns": settlement["funding_time_ms"] * 1_000_000,
                          "scheduled_funding_time_ms": settlement["scheduled_funding_time_ms"],
                          "funding_time_offset_ms": settlement["funding_time_offset_ms"],
                          "available_at_ns": settlement["available_at_ns"],
                          "funding_rate": settlement["funding_rate"],
                          "mark_price": settlement["mark_price"], "rate_type": settlement["rate_type"],
                          "api_response_sha256": settlement["api_response_sha256"],
                          "raw_funding_row": settlement["raw_funding_row"]}
                encoded = canonical(record) + b"\n"
                stream.write(encoded)
                events_hash.update(encoded)
                logical.update(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        source_hashes = [item["sha256"] for item in mark_sources + funding_sources]
        builder = source_revision()
        manifest = {"schema": "td_market_data_usdm_finalized_funding_dataset", "version": 1,
                    "exchange": "BINANCE", "market": "USD_M", "symbol": symbol,
                    "window": {"start_ns": start_ns, "end_ns": end_ns},
                    "expected_settlement_times_ms": [row["funding_time_ms"] for row in schedule],
                    "settlement_observation_evidence": schedule,
                    "event_count": len(settlements), "events_file": "events.jsonl",
                    "events_sha256": events_hash.hexdigest(),
                    "logical_fingerprint_sha256": logical.hexdigest(),
                    "mark_capture_sources": mark_sources, "funding_capture_sources": funding_sources,
                    "source_capture_set_sha256": hashlib.sha256(canonical(source_hashes)).hexdigest(),
                    "collector_code_sha256": sorted({item["collector_code_sha256"]
                                                       for item in mark_sources + funding_sources}),
                    "converter_code_sha256": file_sha256(Path(__file__).resolve()),
                    "builder_source": builder, "replay_usable": builder["tree_sha256"] is not None,
                    "max_pre_settlement_observation_age_ms": max_pre_settlement_age_ms,
                    "max_funding_poll_gap_ms": max_poll_gap_ms,
                    "funding_time_alignment_tolerance_ms": FUNDING_SCHEDULE_ALIGNMENT_TOLERANCE_MS}
        manifest_path = stage / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        for source in mark_sources + funding_sources:
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise FundingDatasetRejected(f"source capture changed during publication: {source['path']}")
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
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parse_time(value: str) -> int:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise argparse.ArgumentTypeError("timestamp must use UTC")
    return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mark-capture", action="append", required=True, type=Path)
    parser.add_argument("--funding-capture", action="append", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    parser.add_argument("--max-pre-settlement-age-ms", type=int, default=5_000)
    parser.add_argument("--max-poll-gap-ms", type=int, default=120_000)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = publish(args.mark_capture, args.funding_capture, args.output,
                           args.symbol.upper(), args.start, args.end,
                           args.max_pre_settlement_age_ms, args.max_poll_gap_ms)
    except (FundingDatasetRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M Funding Dataset rejected: {error}") from error
    print(f"USD-M Funding Dataset published: settlements={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
