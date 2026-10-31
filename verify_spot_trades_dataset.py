#!/usr/bin/env python3
"""Re-verify hashes, provenance shape, timestamps, and trade sequence in a Spot Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from export_spot_depth_dataset import canonical
from export_spot_trades_dataset import MAX_DATASET_BYTES, MAX_EVENTS, SpotTradesRejected, decimal


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def verify(dataset: Path) -> dict[str, Any]:
    if not dataset.is_absolute() or dataset.is_symlink() or not dataset.is_dir():
        raise SpotTradesRejected("Dataset must be an absolute regular directory")
    if {entry.name for entry in dataset.iterdir()} != {"manifest.json", "events.jsonl"}:
        raise SpotTradesRejected("Dataset contains missing or unexpected files")
    manifest_path = dataset / "manifest.json"
    events_path = dataset / "events.jsonl"
    if manifest_path.is_symlink() or events_path.is_symlink() or not manifest_path.is_file() or not events_path.is_file():
        raise SpotTradesRejected("Dataset files must be regular files")
    if events_path.stat().st_size > MAX_DATASET_BYTES or manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise SpotTradesRejected("Dataset file exceeds size policy")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SpotTradesRejected("Dataset manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "td_market_data_spot_trades_dataset" or manifest.get("version") != 1:
        raise SpotTradesRejected("unsupported Spot trades Dataset schema")
    if manifest.get("replay_usable") is not True or manifest.get("events_file") != "events.jsonl":
        raise SpotTradesRejected("Dataset is not replay-usable or names an unexpected event file")
    for key in ("source_capture_sha256", "collector_code_sha256", "converter_code_sha256",
                "events_sha256", "logical_fingerprint_sha256"):
        if not is_sha256(manifest.get(key)):
            raise SpotTradesRejected(f"manifest {key} is missing or malformed")
    builder = manifest.get("builder_source")
    if not isinstance(builder, dict) or not is_sha256(builder.get("tree_sha256")):
        raise SpotTradesRejected("builder source-tree fingerprint is missing")
    if not isinstance(manifest.get("collector_source_revision"), str) or not manifest["collector_source_revision"]:
        raise SpotTradesRejected("collector source revision is missing")
    max_receive_gap_ms = manifest.get("max_receive_gap_ms")
    if not isinstance(max_receive_gap_ms, int) or isinstance(max_receive_gap_ms, bool) or max_receive_gap_ms <= 0:
        raise SpotTradesRejected("manifest receive-gap policy is invalid")
    symbol, session = manifest.get("symbol"), manifest.get("session_id")
    window = manifest.get("window")
    if (manifest.get("exchange") != "BINANCE" or manifest.get("market") != "SPOT"
            or not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum()
            or not isinstance(session, str) or not session
            or not isinstance(window, dict)):
        raise SpotTradesRejected("Dataset identity or window is invalid")
    start_ns, end_ns = window.get("start_ns"), window.get("end_ns")
    if (not isinstance(start_ns, int) or isinstance(start_ns, bool)
            or not isinstance(end_ns, int) or isinstance(end_ns, bool)
            or start_ns >= end_ns):
        raise SpotTradesRejected("Dataset window is invalid")
    domain = hashlib.sha256(b"td-market-data-spot-trades-dataset-v1\0" + canonical(window))
    events_hash = hashlib.sha256()
    count = 0
    previous_id = None
    previous_event_ns = None
    previous_trade_ns = None
    previous_receive_ns = None
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    with events_path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 4 * 1024 * 1024:
                raise SpotTradesRejected(f"event line {line_number} is truncated or oversized")
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SpotTradesRejected(f"event line {line_number} is invalid JSON") from error
            if not isinstance(event, dict) or raw != canonical(event) + b"\n":
                raise SpotTradesRejected(f"event line {line_number} is not a canonical object")
            if (event.get("schema") != "td_market_data_spot_trade_event" or event.get("version") != 1
                    or event.get("exchange") != "BINANCE" or event.get("market") != "SPOT"
                    or event.get("symbol") != symbol or event.get("session_id") != session):
                raise SpotTradesRejected(f"event line {line_number} identity mismatch")
            trade_id = event.get("trade_id")
            event_ns, trade_ns, receive_ns, available_ns = (event.get(key) for key in
                                                             ("event_time_ns", "trade_time_ns", "receive_time_ns", "available_at_ns"))
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in (trade_id, event_ns, trade_ns, receive_ns, available_ns)):
                raise SpotTradesRejected(f"event line {line_number} has invalid ID/time")
            if available_ns != max(event_ns, trade_ns, receive_ns) or not (start_ns <= available_ns < end_ns):
                raise SpotTradesRejected(f"event line {line_number} is outside its window or has invalid availability")
            if event_ns < trade_ns or event_ns > receive_ns:
                raise SpotTradesRejected(f"event line {line_number} has invalid event/trade/receive ordering")
            if previous_id is not None and trade_id != previous_id + 1:
                raise SpotTradesRejected(f"event line {line_number} has a trade ID gap")
            if ((previous_event_ns is not None and event_ns < previous_event_ns)
                    or (previous_trade_ns is not None and trade_ns < previous_trade_ns)
                    or (previous_receive_ns is not None and receive_ns <= previous_receive_ns)):
                raise SpotTradesRejected(f"event line {line_number} timestamps regress")
            if previous_receive_ns is not None and receive_ns - previous_receive_ns > max_receive_gap_ms * 1_000_000:
                raise SpotTradesRejected(f"event line {line_number} receive gap exceeds manifest policy")
            if not isinstance(event.get("buyer_is_maker"), bool):
                raise SpotTradesRejected(f"event line {line_number} buyer-maker flag is invalid")
            expected_aggressor = "SELL" if event["buyer_is_maker"] else "BUY"
            if event.get("aggressor_side") != expected_aggressor:
                raise SpotTradesRejected(f"event line {line_number} aggressor side mismatch")
            price, quantity = decimal(event.get("price"), "trade price"), decimal(event.get("quantity"), "trade quantity")
            quote_quantity = decimal(event.get("quote_quantity"), "trade quote quantity")
            if price * quantity != quote_quantity:
                raise SpotTradesRejected(f"event line {line_number} quote quantity mismatch")
            if not is_sha256(event.get("payload_sha256")):
                raise SpotTradesRejected(f"event line {line_number} payload hash is malformed")
            events_hash.update(raw)
            domain.update(raw)
            count += 1
            if count > MAX_EVENTS:
                raise SpotTradesRejected("Dataset event count exceeds limit")
            if first is None:
                first = event
            last = event
            previous_id, previous_event_ns, previous_trade_ns, previous_receive_ns = trade_id, event_ns, trade_ns, receive_ns
    if not count or first is None or last is None:
        raise SpotTradesRejected("Dataset has no events")
    if count != manifest.get("event_count"):
        raise SpotTradesRejected("Dataset event count differs from manifest")
    if events_hash.hexdigest() != manifest.get("events_sha256"):
        raise SpotTradesRejected("Dataset event file hash mismatch")
    if domain.hexdigest() != manifest.get("logical_fingerprint_sha256"):
        raise SpotTradesRejected("Dataset logical fingerprint mismatch")
    summary_fields = {"first_trade_id": first["trade_id"], "last_trade_id": last["trade_id"],
                      "first_event_time_ns": first["event_time_ns"], "last_event_time_ns": last["event_time_ns"],
                      "first_receive_time_ns": first["receive_time_ns"], "last_receive_time_ns": last["receive_time_ns"]}
    if any(manifest.get(key) != value for key, value in summary_fields.items()):
        raise SpotTradesRejected("Dataset summary fields differ from events")
    return {"schema": "td_market_data_spot_trades_verification", "version": 1,
            "status": "valid", "replay_usable": True, "symbol": symbol,
            "window": window, "event_count": count,
            "events_sha256": events_hash.hexdigest(),
            "logical_fingerprint_sha256": domain.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.dataset), ensure_ascii=False, indent=2))
    except (SpotTradesRejected, OSError, ValueError) as error:
        raise SystemExit(f"Spot trades Dataset verification failed: {error}") from error


if __name__ == "__main__":
    main()
