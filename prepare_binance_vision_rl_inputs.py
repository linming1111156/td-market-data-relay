#!/usr/bin/env python3
"""Repackage frozen historical minute caches as current Relay RL feature/ledger inputs.

This reads the old cache as data only, checks its source-manifest binding against
the audited Binance Vision ZIP hashes, and emits only feature-ready rows inside
the reliable windows. Historical data has exchange event time but no live receive
time; outputs preserve that limitation in every window manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

MINUTE_MS = 60_000
FEATURE_RENAME = {
    "mark_log_return_1m": "mark_log_return_1m",
    "mark_log_return_5m": "mark_log_return_5m",
    "mark_log_return_15m": "mark_log_return_15m",
    "mark_realized_volatility_15m": "mark_realized_volatility_15m",
    "mark_realized_volatility_60m": "mark_realized_volatility_60m",
    "aggtrade_imbalance_1m": "aggtrade_signed_volume_imbalance_1m",
    "aggtrade_imbalance_5m": "aggtrade_signed_volume_imbalance_5m",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def iso_ms(value: int) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def in_windows(stamp: int, windows: list[dict[str, int]]) -> bool:
    return any(row["start_ms"] <= stamp < row["end_ms"] for row in windows)


def source_hashes(audit: dict[str, Any]) -> dict[tuple[str, str], str]:
    found = {}
    for family, key in (("aggTrades", "aggregate_trades"), ("markPriceKlines", "mark_price_klines_1m")):
        for row in audit[key]["archives"]:
            found[(family, row["month"])] = row["sha256"]
    return found


def verify_source_binding(cache_root: Path, audit: dict[str, Any]) -> tuple[dict[str, Any], str]:
    source_path = cache_root.parent / "source-manifest.json"
    source = json.loads(source_path.read_bytes())
    source_hash = sha256_file(source_path)
    if source_hash != "ef2c199d9b2511c8b4ef4b37e420a51b2b6d192a536433b023ad70372f18cb8b":
        raise ValueError("historical cache source manifest differs from the audited baseline")
    audited = source_hashes(audit)
    matched = set()
    for item in source.get("archive_files", []):
        if item.get("family") not in ("aggTrades", "markPriceKlines"):
            continue
        key = (item["family"], item["month"])
        if audited.get(key) != item.get("sha256") or item.get("checksum_match") is not True:
            raise ValueError(f"cache source archive does not match current audit: {key}")
        matched.add(key)
    if len(matched) != 42:
        raise ValueError(f"expected 42 Mark/aggTrades archives bound to audit, found {len(matched)}")
    return source, source_hash


def process_role(role: str, cache_root: Path, windows: list[dict[str, int]],
                 output_root: Path, warmup_mark_close: float | None
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None]:
    cache = cache_root / role
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("role") != role or manifest.get("schema") != "td_gateway_rl_minute_research_cache_manifest":
        raise ValueError(f"unexpected historical minute cache schema/role: {role}")
    source_path = cache / manifest["gzip_jsonl_path"]
    if source_path.is_symlink() or not source_path.is_file() or sha256_file(source_path) != manifest["gzip_jsonl_sha256"]:
        raise ValueError(f"historical {role} gzip JSONL hash mismatch")

    role_windows: list[dict[str, Any]] = []
    skipped_unready: list[int] = []
    active = None
    feature_target = ledger_target = compressor_feature = compressor_ledger = None
    previous_mark_close = warmup_mark_close
    previous_time = None

    def close_window() -> None:
        nonlocal active, feature_target, ledger_target, compressor_feature, compressor_ledger
        if active is None:
            return
        compressor_feature.flush()
        compressor_ledger.flush()
        feature_target.close()
        ledger_target.close()
        active["end_ms"] = previous_time + MINUTE_MS
        active["start"] = iso_ms(active["start_ms"])
        active["end"] = iso_ms(active["end_ms"])
        active["features_sha256"] = sha256_file(active["path"] / "features.jsonl")
        active["ledger_sha256"] = sha256_file(active["path"] / "ledger_inputs.jsonl")
        role_windows.append(active)
        active = None
        feature_target = ledger_target = compressor_feature = compressor_ledger = None

    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                stamp = row["decision_time_ms"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"{role} cache row {line_number} is malformed") from error
            if not isinstance(stamp, int) or stamp % MINUTE_MS:
                raise ValueError(f"{role} cache row {line_number} has a non-minute timestamp")
            execution = row.get("post_decision_execution_minute")
            audit_row = row.get("audit")
            if not isinstance(execution, dict) or not isinstance(audit_row, dict):
                raise ValueError(f"{role} cache row {line_number} lacks execution/source audit fields")

            keep = in_windows(stamp, windows) and row.get("feature_ready") is True
            if in_windows(stamp, windows) and row.get("feature_ready") is not True:
                skipped_unready.append(stamp)
            if not keep:
                close_window()
                previous_mark_close = audit_row.get("mark_kline_close")
                previous_time = stamp
                continue
            if (previous_time is not None and previous_time != stamp - MINUTE_MS) or previous_mark_close is None:
                close_window()
                raise ValueError(f"{role} ready row {iso_ms(stamp)} has no immediately prior Mark close")
            if active is None:
                index = len(role_windows) + 1
                window_dir = output_root / f"{role}-{index:02d}"
                window_dir.mkdir(parents=True, exist_ok=False)
                feature_target = (window_dir / "features.jsonl").open("xb")
                ledger_target = (window_dir / "ledger_inputs.jsonl").open("xb")
                compressor_feature, compressor_ledger = feature_target, ledger_target
                active = {"id": f"{role}-{index:02d}", "role": role, "path": window_dir,
                          "start_ms": stamp, "minute_count": 0}

            raw_features = row["policy_features"]
            if set(raw_features) != set(FEATURE_RENAME):
                raise ValueError(f"{role} feature names do not match expected frozen cache schema")
            features = {FEATURE_RENAME[key]: value for key, value in raw_features.items()}
            feature_record = {"decision_at_ns": stamp * 1_000_000,
                              "decision_due": stamp % (15 * MINUTE_MS) == 0,
                              "feature_ready": True, "mark_price": previous_mark_close,
                              "features": features}
            settlements = []
            for settlement in execution.get("settlements", []):
                settlements.append({"funding_time_ns": int(settlement["event_time_ms"]) * 1_000_000,
                                    "available_at_ns": None,
                                    "funding_rate": float(settlement["rate"]),
                                    "mark_price": float(settlement["mark_price"]),
                                    "availability_basis": "historical_settlement_timestamp_only"})
            ledger_record = {"minute_open_ns": stamp * 1_000_000,
                             "mark_price_at_decision": previous_mark_close,
                             "execution_window_end_ns": int(execution["minute_end_ms_exclusive"]) * 1_000_000,
                             "execution_trade_count": int(execution["aggtrade_count"]),
                             "execution_base_volume": float(execution["base_quantity"]),
                             "execution_quote_volume_usdt": float(execution["quote_quantity"]),
                             "execution_vwap": float(execution["vwap"]) if execution["vwap"] is not None else None,
                             "execution_buy_base_volume": float(audit_row["aggtrade_buy_quantity"]),
                             "execution_sell_base_volume": float(audit_row["aggtrade_sell_quantity"]),
                             "execution_available_at_ns": None,
                             "funding_settlements": settlements,
                             "availability_basis": "historical_exchange_event_time_no_receive_timestamp"}
            compressor_feature.write(canonical(feature_record) + b"\n")
            compressor_ledger.write(canonical(ledger_record) + b"\n")
            active["minute_count"] += 1
            previous_mark_close = audit_row.get("mark_kline_close")
            previous_time = stamp
    close_window()

    for window in role_windows:
        manifest = {"schema": "td_market_data_usdm_rl_historical_window_v1",
                    "role": role, "window_id": window["id"],
                    "window": {key: window[key] for key in ("start_ms", "end_ms", "start", "end")},
                    "minute_count": window["minute_count"],
                    "files": {"features": {"path": "features.jsonl", "sha256": window["features_sha256"]},
                              "ledger_inputs": {"path": "ledger_inputs.jsonl", "sha256": window["ledger_sha256"]}},
                    "source": "Binance Vision historical archives; no live receive-time evidence",
                    "source_cache_manifest_sha256": sha256_file(manifest_path)}
        (window["path"] / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        del window["path"]
    excluded_unready = []
    for stamp in sorted(set(skipped_unready)):
        if excluded_unready and stamp == excluded_unready[-1]["end_ms"]:
            excluded_unready[-1]["end_ms"] += MINUTE_MS
        else:
            excluded_unready.append({"start_ms": stamp, "end_ms": stamp + MINUTE_MS,
                                     "reason": "historical cache feature_ready=false"})
    return role_windows, excluded_unready, previous_mark_close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path,
                        default=Path("/mnt/data/td-gateway-experiments/rl-preproduction-v1/minute-cache"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or output == Path(__file__).resolve().parent:
        raise SystemExit("output must be a new external directory")
    if any(root == output.resolve() or root in output.resolve().parents
           for root in (Path("/tmp"), Path("/var/tmp"))):
        raise SystemExit("output must be on the selected data volume, not system temporary storage")
    audit = json.loads(args.audit.resolve(strict=True).read_bytes())
    if audit.get("schema") != "td_binance_vision_bulk_audit_v1":
        raise SystemExit("unsupported source audit")
    cache_root = args.cache_root.resolve(strict=True)
    source, source_hash = verify_source_binding(cache_root, audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        roles = {}
        unready_exclusions = []
        warmup_mark_close = None
        for role in ("train", "validation", "test", "oos"):
            raw_windows = audit["derived_reliability_windows"]["chronological_roles"][role]
            role_windows, skipped, warmup_mark_close = process_role(
                role, cache_root, raw_windows, stage, warmup_mark_close)
            roles[role] = role_windows
            unready_exclusions.extend({"role": role, **row} for row in skipped)
            print(f"{role}: {len(role_windows)} reliable window(s), "
                  f"{sum(item['minute_count'] for item in role_windows)} usable minutes", flush=True)
        manifest = {"schema": "td_market_data_usdm_rl_historical_input_set_v1",
                    "symbol": "BTCUSDT", "market": "USD_M",
                    "source_manifest_sha256": source_hash,
                    "archive_audit_sha256": sha256_file(args.audit.resolve(strict=True)),
                    "raw_archives": {"aggTrades": len(audit["aggregate_trades"]["archives"]),
                                     "markPriceKlines": len(audit["mark_price_klines_1m"]["archives"]),
                                     "funding_rate_files": len(audit["funding_rate"]["files"])},
                    "time_basis": "historical exchange timestamps; received/available time was not recorded",
                    "unready_cache_exclusions": unready_exclusions,
                    "feature_names": list(FEATURE_RENAME.values()), "roles": roles,
                    "gap_policy": "missing Mark/aggTrades windows and feature lookback recovery are excluded; no fill/interpolation"}
        (stage / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        for path in stage.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"historical RL inputs: {output}", flush=True)


if __name__ == "__main__":
    main()
