#!/usr/bin/env python3
"""Create causal minute features and separated execution/funding ledger inputs."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import deque
from pathlib import Path
from typing import Any, Iterator

from build_usdm_rl_dataset import (canonical, publish_directory, reject_system_temp,
                                   seal_directory, sha256_file)
from verify_usdm_rl_dataset import verify_dataset as verify_input


SCHEMA = "td_market_data_usdm_rl_features"
MINUTE_NS = 60_000_000_000
MAX_MINUTES = 525_600
MAX_MARK_STALENESS_NS = 5_000_000_000
FEATURE_NAMES = (
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
)


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if len(raw) > 2 * 1024 * 1024 or not raw.endswith(b"\n"):
                raise ValueError(f"{path}:{number}: truncated or oversized JSONL")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: record must be an object")
            yield row


def mark_rows(dataset: Path, manifest: dict[str, Any]) -> Iterator[tuple[int, int, float]]:
    previous_available = 0
    previous_event = 0
    evidence = manifest["datasets"]["usdm-funding-mark"]["evidence"]
    for item in evidence:
        for row in rows(dataset / item["path"]):
            if row.get("record_type") != "mark_price_funding_observation":
                continue
            event_ns = row.get("exchange_event_time_ms")
            receive_ns = row.get("received_at_unix_ns")
            available_ns = row.get("available_at_unix_ns")
            price = row.get("mark_price")
            if (not isinstance(event_ns, int) or isinstance(event_ns, bool)
                    or not isinstance(receive_ns, int) or isinstance(receive_ns, bool)
                    or not isinstance(available_ns, int) or isinstance(available_ns, bool)
                    or not isinstance(price, (int, float)) or isinstance(price, bool)
                    or not math.isfinite(price) or price <= 0
                    or event_ns * 1_000_000 > receive_ns + 2_000_000_000
                    or available_ns < max(receive_ns, event_ns * 1_000_000)
                    or event_ns * 1_000_000 < previous_event
                    or available_ns < previous_available):
                raise ValueError("Mark events are invalid or not chronological in availability time")
            previous_event = event_ns * 1_000_000
            previous_available = available_ns
            yield previous_event, available_ns, float(price)


def aggtrade_rows(dataset: Path) -> Iterator[dict[str, Any]]:
    path = dataset / manifest_child(dataset, "usdm-aggtrades")
    previous_event = previous_available = 0
    for row in rows(path / "events.jsonl"):
        event_ns, available_ns = row.get("trade_time_ns"), row.get("available_at_ns")
        if (not isinstance(event_ns, int) or isinstance(event_ns, bool)
                or not isinstance(available_ns, int) or isinstance(available_ns, bool)
                or available_ns < event_ns or event_ns < previous_event
                or available_ns < previous_available):
            raise ValueError("aggregate-trade events are invalid or not chronological in availability time")
        previous_event, previous_available = event_ns, available_ns
        yield row


def manifest_child(dataset: Path, name: str) -> str:
    root = json.loads((dataset / "manifest.json").read_bytes())
    return root["datasets"][name]["path"]


def mark_price_at(minute_marks: dict[int, tuple[int, int, float]], endpoint_ns: int,
                  decision_ns: int, minute_keys: list[int] | None = None) -> float | None:
    key = (endpoint_ns - 1) // MINUTE_NS
    keys = minute_keys if minute_keys is not None else sorted(minute_marks)
    index = bisect.bisect_right(keys, key) - 1
    while index >= 0:
        item = minute_marks[keys[index]]
        event_ns, available_ns, price = item
        if event_ns < endpoint_ns and available_ns <= decision_ns:
            if endpoint_ns - event_ns <= MAX_MARK_STALENESS_NS:
                return price
            return None
        index -= 1
    return None


def aggregate_ledger_inputs(dataset: Path, manifest: dict[str, Any], feature_path: Path,
                            output: Path) -> tuple[str, int]:
    trades_path = dataset / manifest["datasets"]["usdm-aggtrades"]["path"] / "events.jsonl"
    trades = iter(rows(trades_path))
    next_trade = next(trades, None)
    funding_path = dataset / manifest["datasets"]["usdm-finalized-funding"]["path"] / "events.jsonl"
    funding_by_minute: dict[int, list[dict[str, Any]]] = {}
    for funding in rows(funding_path):
        minute = funding["funding_time_ns"] // MINUTE_NS
        funding_by_minute.setdefault(minute, []).append({
            "funding_time_ns": funding["funding_time_ns"],
            "available_at_ns": funding["available_at_ns"],
            "funding_rate": funding["funding_rate"],
            "mark_price": funding["mark_price"],
            "capture_id": funding["capture_id"],
            "api_response_sha256": funding["api_response_sha256"],
        })
    digest = hashlib.sha256()
    count = 0
    with feature_path.open("rb") as features, output.open("xb", buffering=0) as target:
        for raw in features:
            feature = json.loads(raw)
            minute = feature["decision_at_ns"] // MINUTE_NS
            end_ns = feature["decision_at_ns"] + MINUTE_NS
            base = quote = buy_base = sell_base = buy_quote = sell_quote = 0.0
            trade_count = 0
            max_available = 0
            while next_trade is not None and next_trade["trade_time_ns"] < end_ns:
                if next_trade["trade_time_ns"] < feature["decision_at_ns"]:
                    raise ValueError("trade cursor fell behind the execution minute")
                quantity, price = float(next_trade["quantity"]), float(next_trade["price"])
                notional = quantity * price
                base += quantity
                quote += notional
                if next_trade["aggressor_side"] == "BUY":
                    buy_base += quantity
                    buy_quote += notional
                else:
                    sell_base += quantity
                    sell_quote += notional
                max_available = max(max_available, next_trade["available_at_ns"])
                trade_count += 1
                next_trade = next(trades, None)
            record = {"minute_open_ns": feature["decision_at_ns"],
                      "mark_price_at_decision": feature["mark_price"],
                      "execution_window_end_ns": end_ns,
                      "execution_trade_count": trade_count,
                      "execution_base_volume": base,
                      "execution_quote_volume_usdt": quote,
                      "execution_vwap": quote / base if base > 0 else None,
                      "execution_buy_base_volume": buy_base,
                      "execution_sell_base_volume": sell_base,
                      "execution_available_at_ns": max_available or None,
                      "funding_settlements": funding_by_minute.pop(minute, [])}
            encoded = canonical(record) + b"\n"
            target.write(encoded)
            digest.update(encoded)
            count += 1
        target.flush()
        os.fsync(target.fileno())
    if next_trade is not None or funding_by_minute:
        raise ValueError("execution/funding events fall outside the fixed minute window")
    return digest.hexdigest(), count


def build(dataset: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    output = output.absolute()
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ValueError("feature output must be a new create-only directory")
    reject_system_temp(output, "feature output")
    repo = Path(__file__).resolve().parent
    if output.resolve() == repo or repo in output.resolve().parents:
        raise ValueError("feature output must be outside the source repository")
    protocol_path = protocol_path.resolve(strict=True)
    protocol = json.loads(protocol_path.read_bytes())
    if protocol.get("schema") != "td_market_data_usdm_rl_feature_ledger_protocol" or protocol.get("version") != 1:
        raise ValueError("unsupported feature/ledger protocol")
    declared_features = [item.get("name") for item in protocol.get("features", [])]
    if declared_features != list(FEATURE_NAMES):
        raise ValueError("protocol feature order does not match the builder")
    manifest = verify_input(dataset)
    start_ns, end_ns = manifest["window"]["start_ns"], manifest["window"]["end_ns"]
    if start_ns % MINUTE_NS or end_ns % MINUTE_NS:
        raise ValueError("fixed feature Dataset windows must align to UTC minute boundaries")
    minute_count = (end_ns - start_ns) // MINUTE_NS
    if minute_count <= 0 or minute_count > MAX_MINUTES:
        raise ValueError("feature window must contain at most 525600 minutes")
    # The streaming time merge below assumes both streams are chronological in
    # event and effective availability time. Check the full source inventory first.
    for _ in mark_rows(dataset, manifest):
        pass
    for _ in aggtrade_rows(dataset):
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        (stage / "protocol.json").write_bytes(canonical(protocol) + b"\n")
        marks = iter(mark_rows(dataset, manifest))
        next_mark = next(marks, None)
        aggtrades = iter(aggtrade_rows(dataset))
        next_agg = next(aggtrades, None)
        minute_marks: dict[int, tuple[int, int, float]] = {}
        minute_trade_volumes: dict[int, tuple[float, float, int, int]] = {}
        mark_cursor_avail = trade_cursor_avail = 0
        ready_count = 0
        feature_hash = hashlib.sha256()
        with (stage / "features.jsonl").open("xb", buffering=0) as target:
            for index in range(minute_count):
                decision_ns = start_ns + index * MINUTE_NS
                while next_mark is not None and next_mark[1] <= decision_ns and next_mark[0] < decision_ns:
                    if next_mark[1] < mark_cursor_avail:
                        raise ValueError("Mark availability time regressed")
                    mark_cursor_avail = next_mark[1]
                    minute_marks[next_mark[0] // MINUTE_NS] = next_mark
                    next_mark = next(marks, None)
                while next_agg is not None and next_agg["available_at_ns"] <= decision_ns \
                        and next_agg["trade_time_ns"] < decision_ns:
                    if next_agg["available_at_ns"] < trade_cursor_avail:
                        raise ValueError("aggregate-trade availability time regressed")
                    trade_cursor_avail = next_agg["available_at_ns"]
                    key = next_agg["trade_time_ns"] // MINUTE_NS
                    buy, sell, available, count = minute_trade_volumes.get(key, (0.0, 0.0, 0, 0))
                    quantity = float(next_agg["quantity"])
                    if next_agg["aggressor_side"] == "BUY":
                        buy += quantity
                    else:
                        sell += quantity
                    minute_trade_volumes[key] = (buy, sell, max(available, next_agg["available_at_ns"]), count + 1)
                    next_agg = next(aggtrades, None)
                min_mark_minute = decision_ns // MINUTE_NS - 62
                for minute in [key for key in minute_marks if key < min_mark_minute]:
                    del minute_marks[minute]
                min_trade_minute = decision_ns // MINUTE_NS - 6
                for minute in [key for key in minute_trade_volumes if key < min_trade_minute]:
                    del minute_trade_volumes[minute]

                prices: dict[int, float | None] = {}
                available_mark_minutes = sorted(minute_marks)
                for lag in range(0, 61):
                    endpoint = decision_ns - lag * MINUTE_NS
                    prices[lag] = (mark_price_at(minute_marks, endpoint, decision_ns,
                                                 available_mark_minutes)
                                   if endpoint > 0 else None)
                ready = all(prices[lag] is not None for lag in range(61))
                # This field is also needed by execution accounting when the market vector is unready.
                current_price = prices[0]
                returns = [math.log(prices[i] / prices[i + 1])
                           for i in range(60) if prices[i] is not None and prices[i + 1] is not None]
                windows = ((decision_ns // MINUTE_NS - 1, 1),
                           (decision_ns // MINUTE_NS - 5, 5))
                imbalance_values = []
                for first_minute, length in windows:
                    buy_total = sell_total = 0.0
                    complete = True
                    for minute in range(first_minute, first_minute + length):
                        item = minute_trade_volumes.get(minute)
                        if item is None or item[2] > decision_ns or item[3] == 0:
                            complete = False
                            break
                        buy_total += item[0]
                        sell_total += item[1]
                    total = buy_total + sell_total
                    imbalance_values.append((buy_total - sell_total) / total if complete and total > 0 else None)
                if ready and len(returns) == 60:
                    feature_values = {
                        "mark_log_return_1m": math.log(prices[0] / prices[1]),
                        "mark_log_return_5m": math.log(prices[0] / prices[5]),
                        "mark_log_return_15m": math.log(prices[0] / prices[15]),
                        "mark_realized_volatility_15m": math.sqrt(sum(value * value for value in returns[:15])),
                        "mark_realized_volatility_60m": math.sqrt(sum(value * value for value in returns)),
                        "aggtrade_signed_volume_imbalance_1m": imbalance_values[0],
                        "aggtrade_signed_volume_imbalance_5m": imbalance_values[1],
                    }
                    ready = all(value is not None and math.isfinite(value) for value in feature_values.values())
                else:
                    feature_values = None
                if ready:
                    ready_count += 1
                record = {"decision_at_ns": decision_ns,
                          "decision_due": decision_ns % (15 * MINUTE_NS) == 0,
                          "feature_ready": ready,
                          "mark_price": current_price,
                          "features": feature_values if ready else None}
                encoded = canonical(record) + b"\n"
                target.write(encoded)
                feature_hash.update(encoded)
            target.flush()
            os.fsync(target.fileno())
        ledger_hash, ledger_count = aggregate_ledger_inputs(
            dataset, manifest, stage / "features.jsonl", stage / "ledger_inputs.jsonl")
        if ledger_count != minute_count:
            raise ValueError("ledger input rows do not cover the requested minute window")
        feature_manifest = {
            "schema": SCHEMA, "version": 1, "input_dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
            "input_source_capture_set_sha256": manifest["source_capture_set_sha256"],
            "protocol_sha256": hashlib.sha256((stage / "protocol.json").read_bytes()).hexdigest(),
            "features_file": "features.jsonl", "features_sha256": feature_hash.hexdigest(),
            "ledger_inputs_file": "ledger_inputs.jsonl", "ledger_inputs_sha256": ledger_hash,
            "window": manifest["window"], "minute_count": minute_count,
            "feature_ready_count": ready_count, "feature_unready_count": minute_count - ready_count,
            "feature_names": list(FEATURE_NAMES),
            "policy_state": "append current_position_btc / 0.001 at runtime; never fit position into market normalizer",
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        }
        with (stage / "manifest.json").open("xb") as target:
            target.write(json.dumps(feature_manifest, sort_keys=True, indent=2).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        seal_directory(stage)
        publish_directory(stage, output.resolve())
        return feature_manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("offline-usdm-feature-ledger-v1.json"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = build(args.dataset, args.protocol, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"USD-M RL features rejected: {error}") from error
    print(f"USD-M RL features published: minutes={manifest['minute_count']} ready={manifest['feature_ready_count']}")
    print(f"features: {args.output.resolve()}")


if __name__ == "__main__":
    main()
