#!/usr/bin/env python3
"""Stream-audit Binance Vision USD-M bulk archives without extracting them.

The report is an integrity/coverage inventory only. It never treats bulk archive
timestamps as live receive-time evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


MONTH_RE = re.compile(r"(20\d{2})-(0[1-9]|1[0-2])")
MINUTE_MS = 60_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def months_between(first: tuple[int, int], last: tuple[int, int]) -> Iterator[tuple[int, int]]:
    year, month = first
    while (year, month) <= last:
        yield year, month
        year, month = (year + (month == 12), 1 if month == 12 else month + 1)


def month_key(path: Path) -> tuple[int, int]:
    match = MONTH_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot infer YYYY-MM from {path.name}")
    return int(match.group(1)), int(match.group(2))


def archive_inventory(root: Path, pattern: str) -> list[dict[str, Any]]:
    paths = sorted(root.glob(pattern), key=month_key)
    if not paths:
        raise ValueError(f"no {pattern} files in {root}")
    keys = [month_key(path) for path in paths]
    expected = list(months_between(keys[0], keys[-1]))
    if keys != expected:
        missing = sorted(set(expected) - set(keys))
        raise ValueError(f"monthly archive set has missing files in {root}: {missing}")
    result = []
    for path in paths:
        sidecar = path.with_name(path.name + ".CHECKSUM")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ValueError(f"missing/unsafe checksum sidecar: {sidecar}")
        parts = sidecar.read_text().strip().split()
        actual = sha256_file(path)
        if len(parts) < 1 or parts[0].lower() != actual:
            raise ValueError(f"archive SHA-256 mismatch: {path}")
        result.append({"path": str(path), "sha256": actual, "size_bytes": path.stat().st_size,
                       "month": path.stem[-7:]})
    return result


def csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"expected one CSV member in {path}, found {len(members)}")
        info = members[0]
        if not info.filename.endswith(".csv"):
            raise ValueError(f"unexpected ZIP member: {info.filename}")
        with archive.open(info) as binary:
            text = io.TextIOWrapper(binary, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            if not reader.fieldnames:
                raise ValueError(f"CSV header missing: {path}")
            yield from reader


def coalesce_minutes(minutes: list[int], pad: int = 0) -> list[dict[str, int]]:
    if not minutes:
        return []
    minutes = sorted(set(minutes))
    spans: list[list[int]] = []
    start = previous = minutes[0]
    for value in minutes[1:]:
        if value > previous + 1:
            spans.append([start, previous + 1])
            start = value
        previous = value
    spans.append([start, previous + 1])
    if pad:
        spans = [[max(0, start - pad), end + pad] for start, end in spans]
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        spans = merged
    return [{"start_ms": start * MINUTE_MS, "end_ms": end * MINUTE_MS} for start, end in spans]


def merge_intervals(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda row: (row["start_ms"], row["end_ms"]))
    merged: list[dict[str, Any]] = []
    for row in ordered:
        if row["start_ms"] >= row["end_ms"]:
            continue
        if merged and row["start_ms"] <= merged[-1]["end_ms"]:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], row["end_ms"])
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + row["reasons"]))
        else:
            merged.append({"start_ms": row["start_ms"], "end_ms": row["end_ms"],
                           "reasons": list(row["reasons"])})
    return merged


def derive_windows(mark: dict[str, Any], agg: dict[str, Any]) -> dict[str, Any]:
    start_ms = mark["first_open_ms"]
    end_ms = mark["last_open_ms"] + MINUTE_MS
    excluded = [{"start_ms": start_ms, "end_ms": start_ms + 61 * MINUTE_MS,
                 "reasons": ["60-minute feature warmup at source start"]}]
    for window in mark["missing_minute_windows"]:
        excluded.append({"start_ms": window["start_ms"],
                         "end_ms": min(end_ms, window["end_ms"] + 61 * MINUTE_MS),
                         "reasons": ["missing Mark Price 1-minute candles and 60-minute feature recovery"]})
    for window in agg["empty_minute_windows"]:
        excluded.append({"start_ms": window["start_ms"],
                         "end_ms": min(end_ms, window["end_ms"] + 5 * MINUTE_MS),
                         "reasons": ["missing aggTrades minutes and 5-minute feature recovery"]})
    excluded = merge_intervals(excluded)
    reliable = []
    cursor = start_ms
    for window in excluded:
        if cursor < window["start_ms"]:
            reliable.append({"start_ms": cursor, "end_ms": window["start_ms"]})
        cursor = max(cursor, window["end_ms"])
    if cursor < end_ms:
        reliable.append({"start_ms": cursor, "end_ms": end_ms})

    role_bounds = {
        "train": (1735689600000, 1767225600000),
        "validation": (1767225600000, 1775001600000),
        "test": (1775001600000, 1782864000000),
        "oos": (1782864000000, 1790812800000),
    }
    roles = {}
    for role, (role_start, role_end) in role_bounds.items():
        roles[role] = [{"start_ms": max(span["start_ms"], role_start),
                        "end_ms": min(span["end_ms"], role_end)}
                       for span in reliable
                       if max(span["start_ms"], role_start) < min(span["end_ms"], role_end)]
    return {"source_window_ms": {"start_ms": start_ms, "end_ms": end_ms},
            "exclusion_windows": excluded, "reliable_windows": reliable,
            "chronological_roles": roles,
            "window_semantics": "UTC half-open milliseconds; exclude missing source intervals plus the feature lookback recovery period"}


def audit_mark(archives: list[dict[str, Any]]) -> dict[str, Any]:
    first = last = None
    count = 0
    invalid: list[dict[str, Any]] = []
    duplicate_times: list[int] = []
    previous = None
    previous_close = None
    for item in archives:
        path = Path(item["path"])
        month_first = month_last = None
        for line, row in enumerate(csv_rows(path), 2):
            try:
                t = int(row["open_time"])
                close_t = int(row["close_time"])
                prices = [float(row[key]) for key in ("open", "high", "low", "close")]
                if (t % MINUTE_MS or close_t != t + MINUTE_MS - 1
                        or any(not math.isfinite(value) or value <= 0 for value in prices)
                        or prices[2] > min(prices[0], prices[3])
                        or prices[1] < max(prices[0], prices[3])):
                    raise ValueError("invalid minute timestamp/OHLC")
            except (KeyError, TypeError, ValueError) as error:
                invalid.append({"archive": path.name, "line": line, "error": str(error)})
                continue
            if previous is not None:
                if t == previous:
                    duplicate_times.append(t)
                elif t != previous + MINUTE_MS:
                    pass
            previous = t
            previous_close = close_t
            first = t if first is None else first
            last = t
            month_first = t if month_first is None else month_first
            month_last = t
            count += 1
    gaps: list[int] = []
    prior = None
    for item in archives:
        for row in csv_rows(Path(item["path"])):
            try:
                stamp = int(row["open_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if prior is not None and stamp > prior + MINUTE_MS:
                # The next observed candle is the exclusive end of the gap.
                gaps.extend(range((prior + MINUTE_MS) // MINUTE_MS,
                                  stamp // MINUTE_MS))
            prior = stamp
    return {"archives": archives, "row_count": count, "first_open_ms": first,
            "last_open_ms": last, "missing_minute_windows": coalesce_minutes(gaps),
            "duplicate_open_times_ms": duplicate_times[:1000],
            "invalid_rows": invalid[:1000], "invalid_row_count": len(invalid)}


def audit_aggtrades(archives: list[dict[str, Any]]) -> dict[str, Any]:
    rows = 0
    first_id = last_id = first_time = last_time = None
    id_gaps = 0
    missing_ids = 0
    gap_windows: list[dict[str, Any]] = []
    duplicate_ids = 0
    invalid_rows = 0
    invalid_samples: list[dict[str, Any]] = []
    prior_id = prior_time = None
    minute_counts: Counter[int] = Counter()
    for item in archives:
        path = Path(item["path"])
        for line, row in enumerate(csv_rows(path), 2):
            try:
                trade_id = int(row["agg_trade_id"])
                event_ms = int(row["transact_time"])
                price = float(row["price"])
                quantity = float(row["quantity"])
                if (trade_id < 0 or event_ms < 0 or not math.isfinite(price) or price <= 0
                        or not math.isfinite(quantity) or quantity <= 0
                        or row["is_buyer_maker"] not in ("true", "false")):
                    raise ValueError("invalid id/time/price/quantity/maker flag")
            except (KeyError, TypeError, ValueError) as error:
                invalid_rows += 1
                if len(invalid_samples) < 1000:
                    invalid_samples.append({"archive": path.name, "line": line, "error": str(error)})
                continue
            if prior_id is not None:
                if trade_id == prior_id:
                    duplicate_ids += 1
                elif trade_id > prior_id + 1:
                    id_gaps += 1
                    missing_ids += trade_id - prior_id - 1
                    gap_windows.append({"previous_agg_trade_id": prior_id, "next_agg_trade_id": trade_id,
                                        "previous_time_ms": prior_time, "next_time_ms": event_ms,
                                        "missing_id_count": trade_id - prior_id - 1})
            if prior_time is not None and event_ms < prior_time:
                invalid_rows += 1
                if len(invalid_samples) < 1000:
                    invalid_samples.append({"archive": path.name, "line": line,
                                            "error": "event timestamp regressed"})
            first_id = trade_id if first_id is None else first_id
            last_id, prior_id = trade_id, trade_id
            first_time = event_ms if first_time is None else first_time
            last_time = prior_time = event_ms
            minute_counts[event_ms // MINUTE_MS] += 1
            rows += 1
    empty_minutes = []
    if minute_counts:
        first_minute, last_minute = min(minute_counts), max(minute_counts)
        # Store only ranges; do not expand long spans into per-minute lists.
        absent_start = None
        absent_ranges = []
        for minute in range(first_minute, last_minute + 1):
            if minute not in minute_counts and absent_start is None:
                absent_start = minute
            elif minute in minute_counts and absent_start is not None:
                absent_ranges.append({"start_ms": absent_start * MINUTE_MS,
                                      "end_ms": minute * MINUTE_MS})
                absent_start = None
        if absent_start is not None:
            absent_ranges.append({"start_ms": absent_start * MINUTE_MS,
                                  "end_ms": (last_minute + 1) * MINUTE_MS})
        empty_minutes = absent_ranges
    return {"archives": archives, "row_count": rows, "first_agg_trade_id": first_id,
            "last_agg_trade_id": last_id, "first_time_ms": first_time, "last_time_ms": last_time,
            "id_gap_count": id_gaps, "missing_agg_trade_ids": missing_ids,
            "duplicate_agg_trade_ids": duplicate_ids, "invalid_row_count": invalid_rows,
            "invalid_rows": invalid_samples, "id_gap_windows": gap_windows,
            "empty_minute_windows": empty_minutes}


def audit_funding(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("*.json"), key=month_key)
    rows: list[dict[str, Any]] = []
    for path in files:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"funding archive must be a list: {path}")
        for row in data:
            try:
                stamp = int(row["fundingTime"])
                rate = float(row["fundingRate"])
                price = float(row["markPrice"])
                if row.get("symbol") != "BTCUSDT" or not math.isfinite(rate) or not math.isfinite(price) or price <= 0:
                    raise ValueError("invalid symbol/rate/price")
                rows.append({"funding_time_ms": stamp, "funding_rate": rate, "mark_price": price,
                             "rate_type": row.get("rateType"), "source_file": path.name})
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid funding row in {path}: {error}") from error
    rows.sort(key=lambda item: item["funding_time_ms"])
    timestamps = [row["funding_time_ms"] for row in rows]
    return {"files": [{"path": str(path), "sha256": sha256_file(path),
                       "size_bytes": path.stat().st_size} for path in files], "row_count": len(rows),
            "first_funding_time_ms": timestamps[0] if timestamps else None,
            "last_funding_time_ms": timestamps[-1] if timestamps else None,
            "duplicate_funding_times_ms": [t for i, t in enumerate(timestamps) if i and t == timestamps[i-1]][:1000],
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/mnt/data/td-gateway-datasets/rl-preproduction-v1/raw"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("output must be a new path")
    if any(root == args.output.resolve() or root in args.output.resolve().parents
           for root in (Path("/tmp"), Path("/var/tmp"))):
        raise SystemExit("output must be on the data volume, not system temporary storage")
    raw = args.raw_root.resolve(strict=True)
    mark_archives = archive_inventory(raw / "markPriceKlines", "*.zip")
    agg_archives = archive_inventory(raw / "aggTrades", "*.zip")
    print(f"verified archive checksums: Mark={len(mark_archives)} aggTrades={len(agg_archives)}", flush=True)
    mark = audit_mark(mark_archives)
    print(f"Mark parsed: {mark['row_count']} rows", flush=True)
    agg = audit_aggtrades(agg_archives)
    print(f"aggTrades parsed: {agg['row_count']} rows; ID gaps={agg['id_gap_count']} missing IDs={agg['missing_agg_trade_ids']}", flush=True)
    funding = audit_funding(raw / "fundingRate")
    windows = derive_windows(mark, agg)
    report = {"schema": "td_binance_vision_bulk_audit_v1", "symbol": "BTCUSDT",
              "market": "USD_M", "time_basis": "exchange event time from historical monthly archives; no live receive-time evidence",
              "mark_price_klines_1m": mark, "aggregate_trades": agg,
              "funding_rate": funding, "derived_reliability_windows": windows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(f"audit report: {args.output}", flush=True)
    print(f"Mark missing windows: {len(mark['missing_minute_windows'])}; aggTrade empty-minute windows: {len(agg['empty_minute_windows'])}", flush=True)


if __name__ == "__main__":
    main()
