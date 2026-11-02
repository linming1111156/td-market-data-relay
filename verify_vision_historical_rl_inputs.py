#!/usr/bin/env python3
"""Independently verify a published Vision historical RL input set."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import zstandard


SCHEMA = "td_market_data_usdm_rl_historical_input_set"
MINUTE_NS = 60_000_000_000
FEATURE_NAMES = {
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
}
FEATURE_NAME_ORDER = [
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def rows(path: Path):
    import io

    with path.open("rb") as stream:
        reader = zstandard.ZstdDecompressor().stream_reader(stream)
        buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
        for line_number, line in enumerate(buffered, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"truncated JSONL at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row is not an object at {path}:{line_number}")
            yield value


def verify(root: Path, adapter_path: Path | None) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root must be a regular directory")
    manifest_path = root / "manifest.json"
    manifest = load_object(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("version") != 1:
        raise ValueError("unsupported historical input schema/version")
    if manifest.get("symbol") != "BTCUSDT" or manifest.get("market") != "USD_M":
        raise ValueError("unexpected instrument")
    if manifest.get("window_time_basis") != (
            "historical Binance Vision exchange event time; no local receive/availability time exists"):
        raise ValueError("historical time semantics missing or changed")
    if manifest.get("funding_time_basis") != "fundingTime event timestamp; actual local publication time is unknown":
        raise ValueError("Funding time semantics missing or changed")
    if manifest.get("gap_policy") != "only audited reliable half-open windows are emitted; no filling or interpolation":
        raise ValueError("gap policy missing or changed")
    sources = manifest.get("source_manifests", {})
    for field in ("bulk_audit_sha256", "rust_minute_manifest_sha256",
                  "rust_feature_manifest_sha256", "rust_feature_source_manifest_sha256"):
        value = sources.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid provenance fingerprint: {field}")
    adapter = manifest.get("adapter", {})
    if (adapter.get("version") != 1 or not isinstance(adapter.get("sha256"), str)
            or len(adapter["sha256"]) != 64):
        raise ValueError("adapter provenance is missing")
    if adapter_path is not None:
        actual = sha256_file(adapter_path)
        if actual != adapter["sha256"]:
            raise ValueError("artifact was built with a different adapter source")
    config_hash = manifest.get("adapter_configuration_sha256")
    if not isinstance(config_hash, str) or len(config_hash) != 64:
        raise ValueError("adapter configuration fingerprint is missing")

    windows = manifest.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("window inventory is empty or malformed")
    expected_files = {"manifest.json"}
    total_rows = 0
    role_totals: dict[str, dict[str, int]] = {}
    previous_end = None
    funding_total = 0
    config_windows = []
    for window in windows:
        identifier, role = window.get("id"), window.get("role")
        start_ms, end_ms, count = (window.get(k) for k in ("start_ms", "end_ms", "row_count"))
        if (not isinstance(identifier, str) or role not in {"train", "validation", "test", "oos"}
                or not all(isinstance(x, int) and not isinstance(x, bool) for x in (start_ms, end_ms, count))
                or start_ms >= end_ms or start_ms % 60_000 or end_ms % 60_000
                or count != (end_ms - start_ms) // 60_000):
            raise ValueError("invalid role window declaration")
        if previous_end is not None and start_ms < previous_end:
            raise ValueError("role windows overlap or are out of chronological order")
        previous_end = end_ms
        config_windows.append({key: window[key] for key in (
            "id", "role", "start_ms", "end_ms", "start", "end", "minute_count")})
        directory = root / "windows" / identifier
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"missing or unsafe window directory: {identifier}")
        feature_path = directory / window["features"]["path"]
        ledger_path = directory / window["ledger_inputs"]["path"]
        entries = ((feature_path, window["features"]), (ledger_path, window["ledger_inputs"]))
        for path, entry in entries:
            if path.parent != directory or path.is_symlink() or not path.is_file():
                raise ValueError(f"missing or unsafe shard: {path}")
            if path.stat().st_size != entry.get("size_bytes") or sha256_file(path) != entry.get("sha256"):
                raise ValueError(f"shard size/hash mismatch: {path}")
            expected_files.add(path.relative_to(root).as_posix())

        expected_ns = start_ms * 1_000_000
        window_funding = 0
        observed = 0
        for feature, ledger in itertools.zip_longest(rows(feature_path), rows(ledger_path)):
            if feature is None or ledger is None:
                raise ValueError(f"feature/ledger row count mismatch: {identifier}")
            timestamp = feature.get("decision_at_ns")
            if (timestamp != expected_ns or ledger.get("minute_open_ns") != expected_ns
                    or feature.get("decision_due") is not (timestamp % (15 * MINUTE_NS) == 0)
                    or feature.get("feature_ready") is not True
                    or feature.get("time_basis") != "historical_exchange_event_time_no_receive_timestamp"
                    or set(feature.get("features", {})) != FEATURE_NAMES):
                raise ValueError(f"feature/ledger time or schema mismatch: {identifier} row {observed}")
            if any(feature.get(flag) is not True for flag in (
                    "mark_candle_present", "aggtrade_minute_complete", "aggtrade_history_complete")):
                raise ValueError(f"reliable window contains incomplete feature provenance: {identifier}")
            for name, value in feature["features"].items():
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                    raise ValueError(f"non-finite factor {name}: {identifier} row {observed}")
            settlements = ledger.get("funding_settlements")
            if not isinstance(settlements, list):
                raise ValueError(f"Funding settlement list missing: {identifier}")
            for settlement in settlements:
                funding_time = settlement.get("funding_time_ns")
                if (not isinstance(funding_time, int) or funding_time // MINUTE_NS != expected_ns // MINUTE_NS
                        or settlement.get("available_at_ns") is not None
                        or settlement.get("availability_basis") != "historical_settlement_timestamp_only"):
                    raise ValueError(f"invalid Funding event time semantics: {identifier}")
            window_funding += len(settlements)
            expected_ns += MINUTE_NS
            observed += 1
        if observed != count or expected_ns != end_ms * 1_000_000:
            raise ValueError(f"window does not exactly cover its declared range: {identifier}")
        total_rows += observed
        funding_total += window_funding
        role_total = role_totals.setdefault(role, {"window_count": 0, "minute_count": 0})
        role_total["window_count"] += 1
        role_total["minute_count"] += observed

    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual_files != expected_files:
        raise ValueError("artifact has unmanifested or missing files")
    if total_rows != manifest.get("totals", {}).get("reliable_rows"):
        raise ValueError("reliable row total mismatch")
    if funding_total != manifest.get("totals", {}).get("funding_rows_assigned_to_reliable_windows"):
        raise ValueError("Funding assigned-row total mismatch")
    totals = manifest.get("totals", {})
    if (totals.get("funding_source_rows") != funding_total + totals.get("funding_rows_excluded_by_reliability_windows", -1)
            or manifest.get("feature_names") != FEATURE_NAME_ORDER):
        raise ValueError("source Funding totals or feature order mismatch")
    config = {"schema": SCHEMA, "version": 1, "feature_names": FEATURE_NAME_ORDER,
              "reliable_windows": config_windows,
              "funding_mapping": "fundingTime_ms_floor_to_utc_minute",
              "availability_basis": "historical_exchange_event_time_no_receive_timestamp"}
    if hashlib.sha256(canonical(config)).hexdigest() != config_hash:
        raise ValueError("adapter configuration fingerprint does not match role windows")
    for role, actual in role_totals.items():
        if manifest.get("roles", {}).get(role) != actual:
            raise ValueError(f"role totals mismatch: {role}")
    return {"rows": total_rows, "funding_rows": funding_total, "roles": role_totals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, default=Path(__file__).with_name("prepare_vision_historical_rl_inputs.py"))
    args = parser.parse_args()
    result = verify(args.artifact.resolve(strict=True), args.adapter.resolve(strict=True))
    print(f"historical RL input valid: rows={result['rows']} funding={result['funding_rows']}")
    print(json.dumps(result["roles"], sort_keys=True))


if __name__ == "__main__":
    main()
