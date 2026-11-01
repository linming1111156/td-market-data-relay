#!/usr/bin/env python3
"""Independently recheck an immutable USD-M aggregate-trades Dataset."""

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


def uint(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetRejected(f"{field} must be a nonnegative integer")
    return value


def require_hash(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise DatasetRejected(f"{field} must be a lowercase SHA-256")
    return value


def verify_dataset(dataset: Path, source_captures: list[Path] | None = None) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    if dataset.is_symlink() or not dataset.is_dir() or {p.name for p in dataset.iterdir()} != {"manifest.json", "events.jsonl"}:
        raise DatasetRejected("Dataset must be a real directory containing only manifest.json and events.jsonl")
    manifest_path, events_path = dataset / "manifest.json", dataset / "events.jsonl"
    if manifest_path.is_symlink() or events_path.is_symlink():
        raise DatasetRejected("Dataset files cannot be symlinks")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetRejected("manifest.json is invalid") from error
    if (not isinstance(manifest, dict) or manifest.get("schema") != "td_market_data_usdm_aggtrades_dataset"
            or manifest.get("version") != 1 or manifest.get("market") != "USD_M"
            or manifest.get("exchange") != "BINANCE" or manifest.get("events_file") != "events.jsonl"):
        raise DatasetRejected("unsupported aggregate-trades Dataset manifest")
    symbol = manifest.get("symbol")
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum():
        raise DatasetRejected("manifest symbol invalid")
    window = manifest.get("window")
    if not isinstance(window, dict):
        raise DatasetRejected("manifest window missing")
    start_ns, end_ns = uint(window.get("start_ns"), "window start"), uint(window.get("end_ns"), "window end")
    if start_ns >= end_ns or start_ns % 1_000_000 or end_ns % 1_000_000:
        raise DatasetRejected("manifest window invalid")
    logical = hashlib.sha256(b"td-market-data-usdm-aggtrades-v1\0" + canonical({
        "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns}))
    content_hash = hashlib.sha256()
    event_count = 0
    previous_id = None
    previous_event_ns = previous_trade_ns = previous_receive_ns = previous_mono_ns = None
    previous_available_ns = previous_capture_id = None
    first = last = None
    max_receive_gap_ms = 0
    with events_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 2 * 1024 * 1024:
                raise DatasetRejected(f"events.jsonl line {line_number} truncated/oversized")
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetRejected(f"events.jsonl line {line_number} invalid JSON") from error
            if not isinstance(event, dict) or event.get("record_type") != "aggregate_trade" or event.get("symbol") != symbol:
                raise DatasetRejected(f"events.jsonl line {line_number} invalid identity")
            trade_id = uint(event.get("aggregate_trade_id"), "aggregate trade ID")
            event_ns = uint(event.get("exchange_event_time_ns"), "exchange event time")
            trade_ns = uint(event.get("trade_time_ns"), "trade time")
            receive_ns = uint(event.get("received_at_ns"), "receive time")
            available_ns = uint(event.get("available_at_ns"), "availability time")
            mono_ns = uint(event.get("received_monotonic_ns"), "monotonic receive time")
            first_id = uint(event.get("first_trade_id"), "first underlying trade ID")
            last_id = uint(event.get("last_trade_id"), "last underlying trade ID")
            if (first_id > last_id or event_ns > receive_ns + 2_000_000_000
                    or not start_ns <= trade_ns < end_ns
                    or available_ns < max(receive_ns, event_ns, trade_ns, previous_available_ns or 0)
                    or not isinstance(event.get("buyer_is_maker"), bool)
                    or event.get("aggressor_side") != ("SELL" if event["buyer_is_maker"] else "BUY")):
                raise DatasetRejected(f"events.jsonl line {line_number} invalid timestamps/trade fields")
            if previous_id is not None and trade_id != previous_id + 1:
                raise DatasetRejected(f"events.jsonl line {line_number} aggregate trade ID gap")
            if previous_event_ns is not None and event_ns < previous_event_ns:
                raise DatasetRejected(f"events.jsonl line {line_number} exchange event time regressed")
            if previous_trade_ns is not None and trade_ns < previous_trade_ns:
                raise DatasetRejected(f"events.jsonl line {line_number} trade time regressed")
            capture_id = event.get("capture_id")
            if not isinstance(capture_id, str):
                raise DatasetRejected(f"events.jsonl line {line_number} capture ID missing")
            if previous_capture_id == capture_id and previous_receive_ns is not None:
                if receive_ns <= previous_receive_ns or mono_ns <= previous_mono_ns:
                    raise DatasetRejected(f"events.jsonl line {line_number} receive clock regressed within capture")
                max_receive_gap_ms = max(max_receive_gap_ms, (receive_ns - previous_receive_ns) // 1_000_000)
            for field in ("price", "quantity"):
                value = event.get(field)
                if not isinstance(value, str):
                    raise DatasetRejected(f"events.jsonl line {line_number} {field} must be a decimal string")
                try:
                    number = Decimal(value)
                except InvalidOperation as error:
                    raise DatasetRejected(f"events.jsonl line {line_number} invalid {field}") from error
                if not number.is_finite() or number <= 0:
                    raise DatasetRejected(f"events.jsonl line {line_number} invalid {field}")
            require_hash(event.get("raw_frame_sha256"), "raw frame hash")
            previous_id, previous_event_ns = trade_id, event_ns
            previous_trade_ns, previous_receive_ns, previous_mono_ns = trade_ns, receive_ns, mono_ns
            previous_available_ns, previous_capture_id = available_ns, capture_id
            first = first or event
            last = event
            event_count += 1
            content_hash.update(raw)
            logical.update(raw)
    if event_count == 0 or manifest.get("event_count") != event_count:
        raise DatasetRejected("event count mismatch or empty Dataset")
    if (manifest.get("events_sha256") != content_hash.hexdigest()
            or manifest.get("logical_fingerprint_sha256") != logical.hexdigest()):
        raise DatasetRejected("event file or logical fingerprint mismatch")
    for manifest_key, event_key in (("first_aggregate_trade_id", "aggregate_trade_id"),
                                    ("last_aggregate_trade_id", "aggregate_trade_id"),
                                    ("first_trade_time_ns", "trade_time_ns"),
                                    ("last_trade_time_ns", "trade_time_ns"),
                                    ("first_receive_time_ns", "received_at_ns"),
                                    ("last_receive_time_ns", "received_at_ns")):
        expected = first[event_key] if manifest_key.startswith("first_") else last[event_key]
        if manifest.get(manifest_key) != expected:
            raise DatasetRejected(f"manifest {manifest_key} mismatch")
    captures = manifest.get("capture_sources")
    if not isinstance(captures, list) or not captures:
        raise DatasetRejected("capture provenance missing")
    hashes = []
    code_hashes = set()
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in captures:
        if (not isinstance(source, dict)
                or not isinstance(source.get("collector_source_revision"), str)
                or not source["collector_source_revision"]):
            raise DatasetRejected("capture provenance row invalid")
        hashes.append(require_hash(source.get("sha256"), "source capture hash"))
        code_hashes.add(require_hash(source.get("collector_code_sha256"), "collector code hash"))
        capture_id = source.get("capture_id")
        if not isinstance(capture_id, str) or len(capture_id) != 32 or capture_id in source_by_id:
            raise DatasetRejected("capture provenance ID is invalid or duplicated")
        if source.get("origin") == "vps_recovered":
            recovered = uint(source.get("recovered_at_unix_ns"), "capture recovery time")
        elif source.get("origin") == "local" and source.get("recovered_at_unix_ns") is None:
            recovered = 0
        else:
            raise DatasetRejected("capture origin/recovery time is invalid")
        source_by_id[capture_id] = {"recovered_at_unix_ns": recovered}
    last_available_ns = 0
    with events_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            event = json.loads(raw)
            capture_id = event.get("capture_id")
            source = source_by_id.get(capture_id)
            if source is None:
                raise DatasetRejected(f"events.jsonl line {line_number} references an unknown capture")
            # Availability is clamped in event order so a multi-host merge remains
            # point-in-time monotone even when host wall clocks differ slightly.
            expected_available = max(event["received_at_ns"], event["exchange_event_time_ns"], event["trade_time_ns"],
                                     source["recovered_at_unix_ns"], last_available_ns)
            if event.get("available_at_ns") != expected_available:
                raise DatasetRejected(f"events.jsonl line {line_number} availability time mismatch")
            last_available_ns = expected_available
    if manifest.get("source_capture_set_sha256") != hashlib.sha256(canonical(hashes)).hexdigest():
        raise DatasetRejected("source capture-set fingerprint mismatch")
    if manifest.get("collector_code_sha256") != sorted(code_hashes):
        raise DatasetRejected("collector code hash inventory mismatch")
    if manifest.get("max_receive_gap_ms") != max_receive_gap_ms:
        raise DatasetRejected("manifest max receive gap does not match event records")
    boundary = manifest.get("boundary_coverage")
    if not isinstance(boundary, dict):
        raise DatasetRejected("pre/post window coverage evidence missing")
    pre, post = boundary.get("pre_window_trade"), boundary.get("post_window_trade")
    if (not isinstance(pre, dict) or not isinstance(post, dict)
            or uint(pre.get("trade_time_ns"), "pre-window trade time") >= start_ns
            or uint(post.get("trade_time_ns"), "post-window trade time") < end_ns):
        raise DatasetRejected("pre/post window trade coverage evidence is invalid")
    for item in (pre, post):
        uint(item.get("aggregate_trade_id"), "boundary aggregate trade ID")
        uint(item.get("received_at_ns"), "boundary receive time")
        require_hash(item.get("raw_frame_sha256"), "boundary raw-frame hash")
    require_hash(manifest.get("converter_code_sha256"), "converter code hash")
    if source_captures:
        actual = {sha256_file(path.resolve(strict=True)) for path in source_captures}
        if actual != set(hashes):
            raise DatasetRejected("provided source captures do not match manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-capture", action="append", type=Path)
    args = parser.parse_args()
    try:
        manifest = verify_dataset(args.dataset, args.source_capture)
    except (DatasetRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M aggregate-trades Dataset rejected: {error}") from error
    print(f"USD-M aggregate-trades Dataset valid: events={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
