#!/usr/bin/env python3
"""Independently verify a published USD-M finalized Funding Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class DatasetRejected(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_uint(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetRejected(f"{field} must be a nonnegative integer")
    return value


def require_hash(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise DatasetRejected(f"{field} must be a lowercase SHA-256")
    return value


def require_decimal(value: Any, field: str, positive: bool = False) -> str:
    if not isinstance(value, str):
        raise DatasetRejected(f"{field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise DatasetRejected(f"invalid {field}") from error
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise DatasetRejected(f"invalid {field}")
    return value


def verify_dataset(dataset: Path, source_captures: list[Path] | None = None) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    if not dataset.is_dir() or dataset.is_symlink():
        raise DatasetRejected("Dataset path must be a real directory")
    children = {item.name for item in dataset.iterdir()}
    if children != {"manifest.json", "events.jsonl"}:
        raise DatasetRejected("Dataset must contain exactly manifest.json and events.jsonl")
    manifest_path, events_path = dataset / "manifest.json", dataset / "events.jsonl"
    if manifest_path.is_symlink() or events_path.is_symlink():
        raise DatasetRejected("Dataset files cannot be symlinks")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetRejected("manifest.json is unreadable or invalid JSON") from error
    if (not isinstance(manifest, dict) or manifest.get("schema") != "td_market_data_usdm_finalized_funding_dataset"
            or manifest.get("version") != 1 or manifest.get("exchange") != "BINANCE"
            or manifest.get("market") != "USD_M" or manifest.get("events_file") != "events.jsonl"):
        raise DatasetRejected("unsupported USD-M Funding Dataset manifest")
    symbol = manifest.get("symbol")
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum():
        raise DatasetRejected("manifest symbol is invalid")
    window = manifest.get("window")
    if not isinstance(window, dict):
        raise DatasetRejected("manifest window missing")
    start_ns = require_uint(window.get("start_ns"), "window start")
    end_ns = require_uint(window.get("end_ns"), "window end")
    if start_ns >= end_ns or start_ns % 1_000_000 or end_ns % 1_000_000:
        raise DatasetRejected("manifest window is invalid")
    expected = manifest.get("expected_settlement_times_ms")
    schedule = manifest.get("settlement_observation_evidence")
    if not isinstance(expected, list) or not isinstance(schedule, list) or len(schedule) != len(expected):
        raise DatasetRejected("settlement schedule is absent or malformed")
    max_age_ms = require_uint(manifest.get("max_pre_settlement_observation_age_ms"), "maximum Mark age")
    timestamp_tolerance_ms = require_uint(
        manifest.get("funding_time_alignment_tolerance_ms"), "Funding schedule alignment tolerance"
    )
    if timestamp_tolerance_ms != 1_000:
        raise DatasetRejected("Funding schedule alignment tolerance is not the approved 1000 ms bound")
    scheduled: list[int] = []
    for item in schedule:
        if not isinstance(item, dict):
            raise DatasetRejected("schedule evidence row must be an object")
        funding_ms = require_uint(item.get("funding_time_ms"), "scheduled Funding time")
        received_ns = require_uint(item.get("latest_mark_received_at_ns"), "pre-settlement Mark receive time")
        age_ns = require_uint(item.get("mark_observation_age_ns"), "pre-settlement Mark age")
        if (age_ns != funding_ms * 1_000_000 - received_ns or age_ns <= 0
                or age_ns > max_age_ms * 1_000_000
                or not isinstance(item.get("mark_session_id"), str) or not item["mark_session_id"]):
            raise DatasetRejected("pre-settlement Mark evidence is invalid/stale")
        require_hash(item.get("mark_payload_sha256"), "Mark payload hash")
        scheduled.append(funding_ms)
    if scheduled != expected or expected != sorted(set(expected)):
        raise DatasetRejected("expected settlement list differs from ordered schedule evidence")
    if any(time_ms * 1_000_000 < start_ns or time_ms * 1_000_000 >= end_ns for time_ms in expected):
        raise DatasetRejected("scheduled Funding time falls outside manifest window")

    event_hasher = hashlib.sha256()
    logical = hashlib.sha256(b"td-market-data-usdm-funding-dataset-v1\0" + canonical({
        "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns, "schedule": schedule}))
    events: list[dict[str, Any]] = []
    with events_path.open("rb") as stream:
        for line, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 2 * 1024 * 1024:
                raise DatasetRejected(f"events.jsonl line {line} is truncated or oversized")
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetRejected(f"events.jsonl line {line} is invalid JSON") from error
            if (not isinstance(event, dict) or event.get("record_type") != "funding_settlement"
                    or event.get("symbol") != symbol):
                raise DatasetRejected(f"events.jsonl line {line} has invalid event identity")
            time_ms = require_uint(event.get("funding_time_ms"), "Funding time")
            time_ns = require_uint(event.get("funding_time_ns"), "Funding time ns")
            scheduled_ms = require_uint(event.get("scheduled_funding_time_ms"), "scheduled Funding time")
            offset_ms = event.get("funding_time_offset_ms")
            available_ns = require_uint(event.get("available_at_ns"), "Funding availability time")
            raw_row = event.get("raw_funding_row")
            if not isinstance(raw_row, dict):
                raise DatasetRejected(f"events.jsonl line {line} raw Funding row is missing")
            if (time_ns != time_ms * 1_000_000 or available_ns < time_ns
                    or raw_row.get("fundingTime") != time_ms
                    or scheduled_ms not in expected
                    or isinstance(offset_ms, bool) or not isinstance(offset_ms, int)
                    or offset_ms != time_ms - scheduled_ms
                    or abs(offset_ms) > timestamp_tolerance_ms
                    or raw_row.get("symbol") != symbol
                    or raw_row.get("fundingRate") != event.get("funding_rate")
                    or raw_row.get("markPrice") != event.get("mark_price")
                    or event.get("rate_type") != raw_row.get("rateType", "unknown")):
                raise DatasetRejected(f"events.jsonl line {line} has inconsistent event/availability fields")
            require_decimal(event.get("funding_rate"), "funding rate")
            if event.get("mark_price") is not None:
                require_decimal(event.get("mark_price"), "Funding mark price", positive=True)
            require_hash(event.get("api_response_sha256"), "Funding REST response hash")
            events.append(event)
            event_hasher.update(raw)
            logical.update(raw)
    if [event["scheduled_funding_time_ms"] for event in events] != expected:
        raise DatasetRejected("event rows do not exactly match the expected settlement schedule")
    if manifest.get("event_count") != len(events):
        raise DatasetRejected("manifest event count mismatch")
    if manifest.get("events_sha256") != event_hasher.hexdigest():
        raise DatasetRejected("events.jsonl hash mismatch")
    if manifest.get("logical_fingerprint_sha256") != logical.hexdigest():
        raise DatasetRejected("logical Dataset fingerprint mismatch")

    mark_sources, funding_sources = manifest.get("mark_capture_sources"), manifest.get("funding_capture_sources")
    if not isinstance(mark_sources, list) or not mark_sources or not isinstance(funding_sources, list) or not funding_sources:
        raise DatasetRejected("capture source provenance is missing")
    all_hashes: list[str] = []
    for source in mark_sources + funding_sources:
        if not isinstance(source, dict):
            raise DatasetRejected("capture source entry is invalid")
        all_hashes.append(require_hash(source.get("sha256"), "source capture hash"))
        require_hash(source.get("collector_code_sha256"), "collector code hash")
        require_uint(source.get("capture_started_at_ns"), "source capture start")
        require_uint(source.get("capture_stopped_at_ns"), "source capture stop")
        if not isinstance(source.get("source_revision"), str) or not source["source_revision"]:
            raise DatasetRejected("source revision is missing")
    expected_sources_sha = hashlib.sha256(canonical(all_hashes)).hexdigest()
    if manifest.get("source_capture_set_sha256") != expected_sources_sha:
        raise DatasetRejected("source capture-set fingerprint mismatch")
    for key in ("converter_code_sha256",):
        require_hash(manifest.get(key), key)
    code_hashes = manifest.get("collector_code_sha256")
    if not isinstance(code_hashes, list) or not code_hashes:
        raise DatasetRejected("collector code fingerprint list is missing")
    for value in code_hashes:
        require_hash(value, "collector code fingerprint")
    expected_code_hashes = sorted({source["collector_code_sha256"]
                                   for source in mark_sources + funding_sources})
    if code_hashes != expected_code_hashes:
        raise DatasetRejected("collector code fingerprint inventory mismatch")
    builder = manifest.get("builder_source")
    if not isinstance(builder, dict) or builder.get("tree_state") not in {"clean", "dirty", "unknown"}:
        raise DatasetRejected("builder source provenance is missing")
    if manifest.get("replay_usable") is not (builder.get("tree_sha256") is not None):
        raise DatasetRejected("replay usability flag disagrees with builder source fingerprint")

    if source_captures:
        provided = {sha256_file(path.resolve(strict=True)) for path in source_captures}
        if provided != set(all_hashes):
            raise DatasetRejected("provided source captures do not match the manifest source set")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-capture", action="append", type=Path,
                        help="optional original source capture; repeat for all inputs")
    args = parser.parse_args()
    try:
        manifest = verify_dataset(args.dataset, args.source_capture)
    except (DatasetRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M Funding Dataset rejected: {error}") from error
    print(f"USD-M Funding Dataset valid: settlements={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
