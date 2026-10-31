#!/usr/bin/env python3
"""Independently verify an immutable USD-M one-minute Kline Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


BAR_MS = 60_000


class DatasetRejected(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def uint(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DatasetRejected(f"{field} must be a nonnegative integer")
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise DatasetRejected(f"{field} is not a lowercase SHA-256")
    return value


def verify_dataset(dataset: Path, source_captures: list[Path] | None = None) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    if (dataset.is_symlink() or not dataset.is_dir()
            or {path.name for path in dataset.iterdir()} != {"manifest.json", "events.jsonl"}):
        raise DatasetRejected("Dataset directory has unexpected entries or type")
    manifest_path, events_path = dataset / "manifest.json", dataset / "events.jsonl"
    if manifest_path.is_symlink() or events_path.is_symlink():
        raise DatasetRejected("Dataset files cannot be symlinks")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetRejected("manifest.json is invalid") from error
    if (not isinstance(manifest, dict) or manifest.get("version") != 1 or manifest.get("exchange") != "BINANCE"
            or manifest.get("market") not in {"SPOT", "USD_M"} or manifest.get("interval") != "1m"
            or manifest.get("events_file") != "events.jsonl"):
        raise DatasetRejected("unsupported Spot/USD-M Kline Dataset manifest")
    market = manifest["market"]
    if manifest.get("schema") != f"td_market_data_{market.lower()}_klines_dataset":
        raise DatasetRejected("Kline market/schema mismatch")
    symbol = manifest.get("symbol")
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum():
        raise DatasetRejected("manifest symbol invalid")
    window = manifest.get("window")
    if not isinstance(window, dict):
        raise DatasetRejected("manifest window missing")
    start_ns = uint(window.get("start_ns"), "window start")
    end_ns = uint(window.get("end_ns"), "window end")
    if start_ns >= end_ns or start_ns % (BAR_MS * 1_000_000) or end_ns % (BAR_MS * 1_000_000):
        raise DatasetRejected("manifest window invalid")
    expected_count = (end_ns - start_ns) // (BAR_MS * 1_000_000)
    logical = hashlib.sha256(f"td-market-data-{market.lower()}-klines-1m-v1\0".encode() + canonical({
        "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns}))
    event_hash = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    with events_path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 2 * 1024 * 1024:
                raise DatasetRejected(f"events.jsonl line {number} truncated/oversized")
            try:
                event = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise DatasetRejected(f"events.jsonl line {number} invalid JSON") from error
            if not isinstance(event, dict) or event.get("record_type") != "finalized_kline" or event.get("symbol") != symbol or event.get("interval") != "1m":
                raise DatasetRejected(f"events.jsonl line {number} identity mismatch")
            open_ms = uint(event.get("open_time_ms"), "open time")
            close_ms = uint(event.get("close_time_ms"), "close time")
            event_ms = uint(event.get("exchange_event_time_ms"), "exchange event time")
            available_ns = uint(event.get("available_at_ns"), "availability time")
            if (close_ms != open_ms + BAR_MS - 1 or start_ns // 1_000_000 <= open_ms < end_ns // 1_000_000
                    and available_ns < (close_ms + 1) * 1_000_000
                    or event_ms * 1_000_000 > available_ns + 2_000_000_000):
                raise DatasetRejected(f"events.jsonl line {number} invalid time fields")
            prices: dict[str, Decimal] = {}
            for key in ("open", "high", "low", "close"):
                value = event.get(key)
                if not isinstance(value, str):
                    raise DatasetRejected(f"events.jsonl line {number} {key} must be a decimal string")
                try:
                    prices[key] = Decimal(value)
                except InvalidOperation as error:
                    raise DatasetRejected(f"events.jsonl line {number} invalid {key}") from error
                if not prices[key].is_finite() or prices[key] <= 0:
                    raise DatasetRejected(f"events.jsonl line {number} invalid {key}")
            if (prices["high"] < max(prices["open"], prices["close"], prices["low"])
                    or prices["low"] > min(prices["open"], prices["close"], prices["high"])):
                raise DatasetRejected(f"events.jsonl line {number} invalid OHLC envelope")
            volumes = {}
            for key in ("base_volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"):
                value = event.get(key)
                if not isinstance(value, str):
                    raise DatasetRejected(f"events.jsonl line {number} {key} must be a decimal string")
                try:
                    volumes[key] = Decimal(value)
                except InvalidOperation as error:
                    raise DatasetRejected(f"events.jsonl line {number} invalid {key}") from error
                if not volumes[key].is_finite() or volumes[key] < 0:
                    raise DatasetRejected(f"events.jsonl line {number} invalid {key}")
            if (volumes["taker_buy_base_volume"] > volumes["base_volume"]
                    or volumes["taker_buy_quote_volume"] > volumes["quote_volume"]):
                raise DatasetRejected(f"events.jsonl line {number} taker-buy volume exceeds total")
            uint(event.get("trade_count"), "trade count")
            require_hash(event.get("raw_frame_sha256"), "raw frame hash")
            rows.append(event)
            event_hash.update(raw)
            logical.update(raw)
    times = [row["open_time_ms"] for row in rows]
    expected_first = start_ns // 1_000_000
    if (len(rows) != expected_count or times != list(range(expected_first, expected_first + expected_count * BAR_MS, BAR_MS))):
        raise DatasetRejected("events do not exactly cover every finalized minute in the window")
    if manifest.get("event_count") != len(rows):
        raise DatasetRejected("manifest event count mismatch")
    if (manifest.get("events_sha256") != event_hash.hexdigest()
            or manifest.get("logical_fingerprint_sha256") != logical.hexdigest()):
        raise DatasetRejected("events file or logical fingerprint mismatch")
    if (manifest.get("first_open_time_ms") != rows[0]["open_time_ms"]
            or manifest.get("last_open_time_ms") != rows[-1]["open_time_ms"]
            or manifest.get("first_available_at_ns") != rows[0]["available_at_ns"]
            or manifest.get("last_available_at_ns") != rows[-1]["available_at_ns"]):
        raise DatasetRejected("manifest event boundaries mismatch")
    if len(rows) > 1:
        max_gap = max((right["available_at_ns"] - left["available_at_ns"]) // 1_000_000
                      for left, right in zip(rows, rows[1:]))
    else:
        max_gap = 0
    if manifest.get("max_receive_gap_ms") != max_gap:
        raise DatasetRejected("manifest receive-gap summary mismatch")
    boundary = manifest.get("boundary_coverage")
    if not isinstance(boundary, dict):
        raise DatasetRejected("pre/post boundary evidence missing")
    pre, post = boundary.get("pre_window_candle"), boundary.get("post_window_candle")
    if (not isinstance(pre, dict) or not isinstance(post, dict)
            or uint(pre.get("open_time_ms"), "pre-window open") >= expected_first
            or uint(post.get("open_time_ms"), "post-window open") < expected_first + expected_count * BAR_MS):
        raise DatasetRejected("pre/post boundary evidence invalid")
    for item in (pre, post):
        uint(item.get("available_at_ns"), "boundary candle availability")
        require_hash(item.get("raw_frame_sha256"), "boundary raw frame hash")
    sources = manifest.get("capture_sources")
    if not isinstance(sources, list) or not sources:
        raise DatasetRejected("capture provenance missing")
    hashes, code_hashes = [], set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("collector_source_revision"), str):
            raise DatasetRejected("capture provenance row invalid")
        hashes.append(require_hash(source.get("sha256"), "source capture hash"))
        code_hashes.add(require_hash(source.get("collector_code_sha256"), "collector code hash"))
    if manifest.get("source_capture_set_sha256") != hashlib.sha256(canonical(hashes)).hexdigest():
        raise DatasetRejected("source capture-set fingerprint mismatch")
    if manifest.get("collector_code_sha256") != sorted(code_hashes):
        raise DatasetRejected("collector code fingerprint inventory mismatch")
    require_hash(manifest.get("converter_code_sha256"), "converter code hash")
    if source_captures:
        actual = {digest(path.resolve(strict=True)) for path in source_captures}
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
        raise SystemExit(f"USD-M Kline Dataset rejected: {error}") from error
    print(f"{manifest['market']} finalized Kline Dataset valid: bars={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
