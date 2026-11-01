#!/usr/bin/env python3
"""Independently verify the hashes and timeline of a derived USD-M feature set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from verify_usdm_rl_dataset import verify_dataset as verify_input


SCHEMA = "td_market_data_usdm_rl_features"
MINUTE_NS = 60_000_000_000


class FeaturesRejected(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(features: Path, dataset: Path) -> dict[str, Any]:
    features = features.resolve(strict=True)
    if features.is_symlink() or not features.is_dir():
        raise FeaturesRejected("feature artifact must be a regular directory")
    manifest_path, protocol_path = features / "manifest.json", features / "protocol.json"
    feature_path, ledger_path = features / "features.jsonl", features / "ledger_inputs.jsonl"
    for path in (manifest_path, protocol_path, feature_path, ledger_path):
        if path.is_symlink() or not path.is_file():
            raise FeaturesRejected(f"artifact file missing or unsafe: {path.name}")
    if {path.name for path in features.iterdir()} != {
            "manifest.json", "protocol.json", "features.jsonl", "ledger_inputs.jsonl"}:
        raise FeaturesRejected("feature artifact has unmanifested files")
    try:
        manifest = json.loads(manifest_path.read_bytes())
        protocol = json.loads(protocol_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeaturesRejected("feature manifest/protocol JSON invalid") from error
    if manifest.get("schema") != SCHEMA or manifest.get("version") != 1:
        raise FeaturesRejected("unsupported feature manifest")
    source = verify_input(dataset)
    if manifest.get("input_dataset_manifest_sha256") != digest(dataset.resolve(strict=True) / "manifest.json"):
        raise FeaturesRejected("input Dataset manifest hash mismatch")
    if manifest.get("input_source_capture_set_sha256") != source.get("source_capture_set_sha256"):
        raise FeaturesRejected("input capture-set identity mismatch")
    if manifest.get("protocol_sha256") != digest(protocol_path):
        raise FeaturesRejected("feature protocol hash mismatch")
    if protocol.get("schema") != "td_market_data_usdm_rl_feature_ledger_protocol" or protocol.get("version") != 1:
        raise FeaturesRejected("unsupported feature protocol")
    protocol_features = [item.get("name") for item in protocol.get("features", [])]
    if manifest.get("feature_names") != protocol_features or len(protocol_features) != 7:
        raise FeaturesRejected("feature order differs from frozen protocol")
    window = manifest.get("window")
    if window != source.get("window") or not isinstance(window, dict):
        raise FeaturesRejected("feature and source windows differ")
    count = (window["end_ns"] - window["start_ns"]) // MINUTE_NS
    if (window["start_ns"] % MINUTE_NS or window["end_ns"] % MINUTE_NS
            or count != manifest.get("minute_count") or count <= 0):
        raise FeaturesRejected("feature minute window/count invalid")
    feature_hash = hashlib.sha256()
    ready_count = 0
    row_count = 0
    with feature_path.open("rb") as stream:
        for index, raw in enumerate(stream):
            if not raw.endswith(b"\n") or len(raw) > 1024 * 1024:
                raise FeaturesRejected(f"features.jsonl row {index + 1} truncated/oversized")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FeaturesRejected(f"features.jsonl row {index + 1} invalid JSON") from error
            expected_time = window["start_ns"] + index * MINUTE_NS
            if (not isinstance(row, dict) or row.get("decision_at_ns") != expected_time
                    or row.get("decision_due") is not (expected_time % (15 * MINUTE_NS) == 0)
                    or not isinstance(row.get("feature_ready"), bool)):
                raise FeaturesRejected(f"features.jsonl row {index + 1} has an invalid time/decision marker")
            price = row.get("mark_price")
            if price is not None and (not isinstance(price, (int, float)) or isinstance(price, bool)
                                      or not math.isfinite(price) or price <= 0):
                raise FeaturesRejected(f"features.jsonl row {index + 1} Mark price invalid")
            vector = row.get("features")
            if row["feature_ready"]:
                if not isinstance(vector, dict) or set(vector) != set(manifest["feature_names"]):
                    raise FeaturesRejected(f"features.jsonl row {index + 1} feature vector incomplete")
                for name, value in vector.items():
                    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                        raise FeaturesRejected(f"features.jsonl row {index + 1} invalid feature {name}")
                if (vector["mark_realized_volatility_15m"] < 0
                        or vector["mark_realized_volatility_60m"] < 0
                        or any(abs(vector[name]) > 1 for name in (
                            "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m"))):
                    raise FeaturesRejected(f"features.jsonl row {index + 1} feature outside protocol range")
                ready_count += 1
            elif vector is not None:
                raise FeaturesRejected(f"features.jsonl row {index + 1} unready vector must remain null")
            feature_hash.update(raw)
            row_count = index + 1
    if count != row_count or feature_hash.hexdigest() != manifest.get("features_sha256"):
        raise FeaturesRejected("feature row count/hash mismatch")
    if ready_count != manifest.get("feature_ready_count") or count - ready_count != manifest.get("feature_unready_count"):
        raise FeaturesRejected("feature ready/unready counts mismatch")

    ledger_hash = hashlib.sha256()
    ledger_count = 0
    funding_dataset_path = dataset.resolve(strict=True) / source["datasets"]["usdm-finalized-funding"]["path"] / "events.jsonl"
    funding_stream = funding_dataset_path.open("rb")
    aggtrade_path = dataset.resolve(strict=True) / source["datasets"]["usdm-aggtrades"]["path"] / "events.jsonl"
    aggtrade_stream = aggtrade_path.open("rb")
    next_funding = None
    next_trade = None

    def read_funding() -> dict[str, Any] | None:
        raw_funding = funding_stream.readline()
        if not raw_funding:
            return None
        try:
            return json.loads(raw_funding)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeaturesRejected("source Funding Dataset contains invalid JSON") from error

    next_funding = read_funding()

    def read_trade() -> dict[str, Any] | None:
        raw_trade = aggtrade_stream.readline()
        if not raw_trade:
            return None
        try:
            return json.loads(raw_trade)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeaturesRejected("source aggregate-trades Dataset contains invalid JSON") from error

    next_trade = read_trade()
    with feature_path.open("rb") as features_stream, ledger_path.open("rb") as stream:
        for ledger_count, pair in enumerate(zip(features_stream, stream, strict=False), 1):
            feature_raw, raw = pair
            if not raw.endswith(b"\n") or len(raw) > 1024 * 1024:
                raise FeaturesRejected(f"ledger input row {ledger_count} truncated/oversized")
            row = json.loads(raw)
            feature = json.loads(feature_raw)
            expected_time = window["start_ns"] + (ledger_count - 1) * MINUTE_NS
            if row.get("minute_open_ns") != expected_time or row.get("execution_window_end_ns") != expected_time + MINUTE_NS:
                raise FeaturesRejected(f"ledger input row {ledger_count} timeline mismatch")
            if row.get("mark_price_at_decision") != feature.get("mark_price"):
                raise FeaturesRejected(f"ledger input row {ledger_count} Mark price differs from features table")
            execution_base = execution_quote = execution_buy = execution_sell = 0.0
            execution_count = 0
            execution_available = 0
            while next_trade is not None and next_trade["trade_time_ns"] < expected_time + MINUTE_NS:
                if next_trade["trade_time_ns"] < expected_time:
                    raise FeaturesRejected("source trade falls behind its execution minute")
                quantity, price = float(next_trade["quantity"]), float(next_trade["price"])
                notional = quantity * price
                execution_base += quantity
                execution_quote += notional
                if next_trade["aggressor_side"] == "BUY":
                    execution_buy += quantity
                else:
                    execution_sell += quantity
                execution_available = max(execution_available, next_trade["available_at_ns"])
                execution_count += 1
                next_trade = read_trade()
            for field, actual, expected in (
                    ("execution_trade_count", row.get("execution_trade_count"), execution_count),
                    ("execution_base_volume", row.get("execution_base_volume"), execution_base),
                    ("execution_quote_volume_usdt", row.get("execution_quote_volume_usdt"), execution_quote),
                    ("execution_buy_base_volume", row.get("execution_buy_base_volume"), execution_buy),
                    ("execution_sell_base_volume", row.get("execution_sell_base_volume"), execution_sell)):
                if field == "execution_trade_count":
                    if actual != expected:
                        raise FeaturesRejected(f"ledger input row {ledger_count} {field} differs from source events")
                elif not isinstance(actual, (int, float)) or abs(actual - expected) > max(1e-12, abs(expected) * 1e-10):
                    raise FeaturesRejected(f"ledger input row {ledger_count} {field} differs from source events")
            if row.get("execution_available_at_ns") != (execution_available or None):
                raise FeaturesRejected(f"ledger input row {ledger_count} execution availability differs from source events")
            trades = row.get("execution_trade_count")
            base, quote, vwap = row.get("execution_base_volume"), row.get("execution_quote_volume_usdt"), row.get("execution_vwap")
            if not isinstance(trades, int) or isinstance(trades, bool) or trades < 0:
                raise FeaturesRejected(f"ledger input row {ledger_count} trade count invalid")
            if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                       and math.isfinite(value) and value >= 0 for value in (base, quote)):
                raise FeaturesRejected(f"ledger input row {ledger_count} execution volume invalid")
            if (base == 0) != (vwap is None) or (base > 0 and (
                    not isinstance(vwap, (int, float)) or isinstance(vwap, bool)
                    or not math.isfinite(vwap) or vwap <= 0 or abs(quote / base - vwap) > 1e-8 * vwap)):
                raise FeaturesRejected(f"ledger input row {ledger_count} execution VWAP inconsistent")
            settlements = row.get("funding_settlements")
            if not isinstance(settlements, list):
                raise FeaturesRejected(f"ledger input row {ledger_count} Funding list invalid")
            expected_settlements = []
            while next_funding is not None and next_funding["funding_time_ns"] < expected_time + MINUTE_NS:
                if next_funding["funding_time_ns"] < expected_time:
                    raise FeaturesRejected("source Funding settlement is outside its UTC minute")
                expected_settlements.append({
                    "funding_time_ns": next_funding["funding_time_ns"],
                    "available_at_ns": next_funding["available_at_ns"],
                    "funding_rate": next_funding["funding_rate"],
                    "mark_price": next_funding["mark_price"],
                    "capture_id": next_funding["capture_id"],
                    "api_response_sha256": next_funding["api_response_sha256"],
                })
                next_funding = read_funding()
            if settlements != expected_settlements:
                raise FeaturesRejected(f"ledger input row {ledger_count} Funding time misbucketed")
            ledger_hash.update(raw)
        if features_stream.read(1) or stream.read(1):
            raise FeaturesRejected("feature/ledger minute counts differ")
    funding_stream.close()
    aggtrade_stream.close()
    if next_funding is not None:
        raise FeaturesRejected("source Funding settlement falls outside the fixed feature window")
    if next_trade is not None:
        raise FeaturesRejected("source aggregate trade falls outside the fixed feature window")
    if ledger_count != count or ledger_hash.hexdigest() != manifest.get("ledger_inputs_sha256"):
        raise FeaturesRejected("ledger input row count/hash mismatch")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = verify(args.features, args.dataset)
    except (FeaturesRejected, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"USD-M RL features rejected: {error}") from error
    print(f"USD-M RL features valid: minutes={manifest['minute_count']} ready={manifest['feature_ready_count']}")


if __name__ == "__main__":
    main()
