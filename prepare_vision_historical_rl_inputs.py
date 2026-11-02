#!/usr/bin/env python3
"""Build immutable historical RL inputs from Rust Vision feature shards.

The output has the Python runner's feature/ledger row contract, but an explicit
historical-data manifest. It is not represented as a live Relay capture.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import zstandard


SCHEMA = "td_market_data_usdm_rl_historical_input_set"
FEATURE_SCHEMA = "td_market_data_usdm_rl_historical_feature_set"
ADAPTER_VERSION = 1
MINUTE_MS = 60_000
MINUTE_NS = 60_000_000_000
FEATURE_NAMES = [
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
]
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/data").resolve(strict=True)
DEFAULT_RAW_ROOT = Path("/mnt/data/td-gateway-datasets/rl-preproduction-v1/raw")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def adapter_code_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def iso_ms(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_external_output(output: Path, inputs: list[Path]) -> Path:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new path: {output}")
    resolved = output.resolve()
    if resolved == DATA_ROOT or DATA_ROOT not in resolved.parents:
        raise ValueError("output must be written under the /mnt/data data volume")
    for root in (Path("/tmp"), Path("/var/tmp")):
        if resolved == root or root in resolved.parents:
            raise ValueError("output must be on the data volume, not system temporary storage")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("output must be outside the source repository")
    for item in inputs:
        item = item.resolve(strict=True)
        if resolved == item or item in resolved.parents:
            raise ValueError(f"output must not be inside an input: {item}")
    return output


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def index_archives(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"audit {label} inventory is malformed")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("month"), str):
            raise ValueError(f"audit {label} entry is malformed")
        month = row["month"]
        if month in result:
            raise ValueError(f"duplicate {label} archive for {month}")
        result[month] = row
    return result


def verify_archive_binding(audit: dict[str, Any], minute: dict[str, Any],
                           minute_root: Path, raw_root: Path) -> dict[str, dict[str, Any]]:
    if (audit.get("schema") != "td_binance_vision_bulk_audit_v1"
            or audit.get("symbol") != "BTCUSDT" or audit.get("market") != "USD_M"
            or minute.get("schema") != "td_market_data_vision_minute_dataset"
            or minute.get("version") != 1 or minute.get("symbol") != "BTCUSDT"
            or minute.get("market") != "USD_M"):
        raise ValueError("unsupported bulk audit or Rust minute dataset")
    marks = index_archives(audit.get("mark_price_klines_1m", {}).get("archives"), "Mark")
    trades = index_archives(audit.get("aggregate_trades", {}).get("archives"), "aggTrades")
    if len(marks) != 21 or len(trades) != 21 or len(minute.get("months", [])) != 21:
        raise ValueError("expected 21 audited Mark and aggTrades months")
    archive_dirs = {"mark": (raw_root / "markPriceKlines").resolve(strict=True),
                    "aggTrades": (raw_root / "aggTrades").resolve(strict=True)}
    for label, inventory, directory in (("Mark", marks, archive_dirs["mark"]),
                                        ("aggTrades", trades, archive_dirs["aggTrades"])):
        for month, item in inventory.items():
            path = Path(item.get("path", ""))
            if (not path.is_absolute() or path.parent.resolve(strict=True) != directory
                    or path.is_symlink() or not path.is_file()):
                raise ValueError(f"unsafe or missing {label} archive for {month}: {path}")
            if (path.stat().st_size != item.get("size_bytes")
                    or sha256_file(path) != item.get("sha256")):
                raise ValueError(f"{label} archive hash/size differs from bulk audit for {month}")
    seen = set()
    for month in minute["months"]:
        key = month.get("month")
        if key in seen or key not in marks or key not in trades:
            raise ValueError(f"unexpected or duplicate Rust minute month: {key}")
        seen.add(key)
        if (month.get("mark_archive_sha256") != marks[key].get("sha256")
                or month.get("aggtrades_archive_sha256") != trades[key].get("sha256")):
            raise ValueError(f"Rust minute archive hashes differ from audit for {key}")
        minute_path = minute_root / month.get("output_path", "")
        if (minute_path.parent != minute_root or minute_path.is_symlink() or not minute_path.is_file()
                or minute_path.stat().st_size != month.get("output_size_bytes")
                or sha256_file(minute_path) != month.get("output_sha256")):
            raise ValueError(f"Rust minute shard hash/size mismatch for {key}")
    if seen != set(marks) or seen != set(trades):
        raise ValueError("Rust minute dataset and audited archive month sets differ")
    return {month["month"]: month for month in minute["months"]}


def load_funding(audit: dict[str, Any], raw_root: Path) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    inventory = audit.get("funding_rate", {}).get("files")
    if not isinstance(inventory, list) or len(inventory) != 21:
        raise ValueError("expected 21 audited monthly Funding files")
    raw_dir = (raw_root / "fundingRate").resolve(strict=True)
    expected_rows: dict[int, dict[str, Any]] = {}
    normalized_by_minute: dict[int, list[dict[str, Any]]] = {}
    file_reports = []
    for item in inventory:
        if not isinstance(item, dict):
            raise ValueError("Funding file audit entry is malformed")
        path = Path(item.get("path", ""))
        if (not path.is_absolute() or path.parent.resolve(strict=True) != raw_dir
                or not path.name.startswith("BTCUSDT-fundingRate-")
                or path.suffix != ".json"):
            raise ValueError(f"Funding source escaped raw root: {path}")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Funding source is missing or unsafe: {path}")
        digest = sha256_file(path)
        if digest != item.get("sha256") or path.stat().st_size != item.get("size_bytes"):
            raise ValueError(f"Funding source hash/size mismatch: {path.name}")
        records = load_json_array(path)
        file_count = 0
        for record in records:
            if not isinstance(record, dict) or record.get("symbol") != "BTCUSDT":
                raise ValueError(f"invalid Funding event in {path.name}")
            stamp = strict_int(record.get("fundingTime"), "fundingTime")
            rate = finite_float(record.get("fundingRate"), "fundingRate")
            mark_raw = record.get("markPrice")
            mark = None if mark_raw in (None, "") else finite_float(mark_raw, "markPrice")
            if mark is not None and mark <= 0:
                raise ValueError(f"Funding Mark price is not positive in {path.name}")
            if stamp in expected_rows:
                raise ValueError(f"duplicate Funding timestamp: {stamp}")
            row = {"funding_time_ms": stamp, "funding_rate": rate, "mark_price": mark,
                   "rate_type": "Regular", "source_file": path.name}
            expected_rows[stamp] = row
            minute_ns = (stamp // MINUTE_MS) * MINUTE_NS
            normalized_by_minute.setdefault(minute_ns, []).append({
                "funding_time_ns": stamp * 1_000_000,
                "available_at_ns": None,
                "funding_rate": rate,
                "mark_price": mark,
                "availability_basis": "historical_settlement_timestamp_only",
            })
            file_count += 1
        file_reports.append({"path": path.name, "sha256": digest,
                             "size_bytes": path.stat().st_size, "row_count": file_count})
    if list(expected_rows) != sorted(expected_rows):
        raise ValueError("Funding settlements are not strictly chronological by source month")
    audit_rows = audit.get("funding_rate", {}).get("rows")
    if not isinstance(audit_rows, list) or len(audit_rows) != len(expected_rows):
        raise ValueError("Funding rows do not reconcile with audit row count")
    for row in audit_rows:
        stamp = row.get("funding_time_ms")
        actual = expected_rows.get(stamp)
        if actual is None or any(actual[key] != row.get(key) for key in (
                "funding_time_ms", "funding_rate", "mark_price", "rate_type", "source_file")):
            raise ValueError(f"raw Funding row differs from bulk audit at {stamp}")
    if len(expected_rows) != audit["funding_rate"].get("row_count"):
        raise ValueError("Funding source totals differ from audit")
    return normalized_by_minute, file_reports


def load_json_array(path: Path) -> list[Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Funding JSON: {path.name}") from error
    if not isinstance(value, list):
        raise ValueError(f"Funding JSON must contain an array: {path.name}")
    return value


def strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if str(parsed) != str(value).strip():
        raise ValueError(f"{label} must be an exact integer")
    return parsed


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def role_windows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    roles = audit.get("derived_reliability_windows", {}).get("chronological_roles")
    if not isinstance(roles, dict):
        raise ValueError("bulk audit has no chronological role windows")
    result = []
    for role in ("train", "validation", "test", "oos"):
        windows = roles.get(role)
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"bulk audit is missing {role} windows")
        for index, window in enumerate(windows, 1):
            start, end = window.get("start_ms"), window.get("end_ms")
            if (not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(end, int) or isinstance(end, bool)
                    or start >= end or start % MINUTE_MS or end % MINUTE_MS):
                raise ValueError(f"invalid {role} reliable window")
            result.append({"id": f"{role}-{index:02d}", "role": role,
                           "start_ms": start, "end_ms": end,
                           "start": iso_ms(start), "end": iso_ms(end),
                           "minute_count": (end - start) // MINUTE_MS})
    ordered = sorted(result, key=lambda row: row["start_ms"])
    if any(left["end_ms"] > right["start_ms"] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("reliable role windows overlap")
    train_minutes = sum(row["minute_count"] for row in ordered if row["role"] == "train")
    if train_minutes > 525_600:
        raise ValueError("Train window exceeds the one-year limit")
    return ordered


@dataclass
class WindowSink:
    window: dict[str, Any]
    root: Path
    feature_file: BinaryIO
    ledger_file: BinaryIO
    feature_stream: Any
    ledger_stream: Any
    rows: int = 0

    @classmethod
    def open(cls, root: Path, window: dict[str, Any]) -> "WindowSink":
        folder = root / window["id"]
        folder.mkdir(parents=True, exist_ok=False)
        feature_file = (folder / "features.jsonl.zst").open("xb", buffering=0)
        ledger_file = (folder / "ledger_inputs.jsonl.zst").open("xb", buffering=0)
        return cls(window, folder, feature_file, ledger_file,
                   zstandard.ZstdCompressor(level=3).stream_writer(feature_file, closefd=False),
                   zstandard.ZstdCompressor(level=3).stream_writer(ledger_file, closefd=False))

    def write(self, feature: dict[str, Any], ledger: dict[str, Any]) -> None:
        self.feature_stream.write(canonical(feature) + b"\n")
        self.ledger_stream.write(canonical(ledger) + b"\n")
        self.rows += 1

    def close(self) -> dict[str, Any]:
        self.feature_stream.close()
        self.ledger_stream.close()
        for stream in (self.feature_file, self.ledger_file):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        if self.rows != self.window["minute_count"]:
            raise ValueError(f"window {self.window['id']} row count mismatch: {self.rows}")
        feature_path = self.root / "features.jsonl.zst"
        ledger_path = self.root / "ledger_inputs.jsonl.zst"
        return {**self.window, "row_count": self.rows,
                "features": {"path": feature_path.name, "sha256": sha256_file(feature_path),
                             "size_bytes": feature_path.stat().st_size},
                "ledger_inputs": {"path": ledger_path.name, "sha256": sha256_file(ledger_path),
                                  "size_bytes": ledger_path.stat().st_size}}


def verify_inputs(audit_path: Path, minute_root: Path, feature_root: Path,
                  raw_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = load_json(audit_path, "bulk audit")
    minute_path = minute_root / "manifest.json"
    feature_path = feature_root / "manifest.json"
    minute = load_json(minute_path, "Rust minute manifest")
    feature = load_json(feature_path, "Rust feature manifest")
    if feature.get("schema") != FEATURE_SCHEMA or feature.get("version") != 1:
        raise ValueError("unsupported Rust historical feature manifest")
    if feature.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Rust feature order does not match Relay v1 contract")
    if feature.get("source_manifest_sha256") != sha256_file(minute_path):
        raise ValueError("Rust feature manifest is not bound to supplied minute manifest")
    minute_months = verify_archive_binding(audit, minute, minute_root, raw_root)
    if len(feature.get("months", [])) != len(minute_months):
        raise ValueError("Rust feature month inventory differs from minute dataset")
    for item in feature["months"]:
        month = item.get("month")
        minute_item = minute_months.get(month)
        if minute_item is None or item.get("input_sha256") != minute_item.get("output_sha256"):
            raise ValueError(f"Rust feature shard does not bind to minute source for {month}")
        path = feature_root / item.get("output_path", "")
        if path.parent != feature_root or path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe Rust feature shard for {month}")
        if (path.stat().st_size != item.get("output_size_bytes")
                or sha256_file(path) != item.get("output_sha256")):
            raise ValueError(f"Rust feature shard hash mismatch for {month}")
    return audit, minute, feature


def process(audit_path: Path, minute_root: Path, feature_root: Path,
            raw_root: Path, output: Path) -> dict[str, Any]:
    output = safe_external_output(output, [audit_path, minute_root, feature_root, raw_root])
    audit, minute, feature_manifest = verify_inputs(audit_path, minute_root, feature_root, raw_root)
    funding_by_minute, funding_files = load_funding(audit, raw_root)
    windows = role_windows(audit)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.historical-stage-", dir=output.parent))
    (stage / "windows").mkdir()
    artifacts = []
    window_index = 0
    sink: WindowSink | None = None
    previous_time = None
    total_rows = 0
    ready_rows = 0
    assigned_funding_rows = 0
    try:
        for month in feature_manifest["months"]:
            shard = feature_root / month["output_path"]
            with shard.open("rb") as source:
                reader = zstandard.ZstdDecompressor().stream_reader(source)
                buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
                for encoded in buffered:
                    if not encoded.endswith(b"\n"):
                        raise ValueError(f"truncated Rust feature row in {shard.name}")
                    row = json.loads(encoded)
                    stamp = row.get("decision_time_ms")
                    if not isinstance(stamp, int) or stamp % MINUTE_MS:
                        raise ValueError(f"invalid Rust feature timestamp in {shard.name}")
                    if previous_time is not None and stamp != previous_time + MINUTE_MS:
                        raise ValueError(f"feature timeline discontinuity at {stamp}")
                    previous_time = stamp
                    total_rows += 1

                    while window_index < len(windows) and stamp >= windows[window_index]["end_ms"]:
                        if sink is not None:
                            artifacts.append(sink.close())
                            sink = None
                        window_index += 1
                    if window_index >= len(windows):
                        continue
                    window = windows[window_index]
                    if stamp < window["start_ms"]:
                        continue
                    if stamp >= window["end_ms"]:
                        continue
                    if sink is None:
                        sink = WindowSink.open(stage / "windows", window)

                    required_quality = (row.get("feature_ready") is True
                                        and row.get("mark_candle_present") is True
                                        and row.get("aggtrade_minute_complete") is True
                                        and row.get("aggtrade_history_complete") is True)
                    if not required_quality:
                        raise ValueError(f"audit reliable window contains an unready row at {stamp}")
                    features = row.get("features")
                    if not isinstance(features, dict) or set(features) != set(FEATURE_NAMES):
                        raise ValueError(f"feature vector schema mismatch at {stamp}")
                    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
                           or not math.isfinite(value) for value in features.values()):
                        raise ValueError(f"non-finite feature value at {stamp}")
                    execution = row.get("execution")
                    if not isinstance(execution, dict) or execution.get("minute_open_ns") != stamp * 1_000_000:
                        raise ValueError(f"execution ledger timestamp mismatch at {stamp}")
                    ledger = dict(execution)
                    minute_funding = funding_by_minute.pop(stamp * 1_000_000, [])
                    assigned_funding_rows += len(minute_funding)
                    ledger["funding_settlements"] = minute_funding
                    ledger["availability_basis"] = "historical_exchange_event_time_no_receive_timestamp"
                    feature_row = {
                        "decision_at_ns": row["decision_at_ns"],
                        "decision_due": row["decision_due"],
                        "feature_ready": True,
                        "mark_price": row["mark_price"],
                        "features": features,
                        "mark_candle_present": row["mark_candle_present"],
                        "aggtrade_minute_complete": row["aggtrade_minute_complete"],
                        "aggtrade_history_complete": row["aggtrade_history_complete"],
                        "time_basis": "historical_exchange_event_time_no_receive_timestamp",
                    }
                    sink.write(feature_row, ledger)
                    ready_rows += 1
        if sink is not None:
            artifacts.append(sink.close())
            sink = None
        if (window_index < len(windows) and previous_time is not None
                and previous_time + MINUTE_MS >= windows[window_index]["end_ms"]):
            window_index += 1
        if window_index != len(windows):
            raise ValueError("one or more audit role windows were not found in Rust feature shards")
        unassigned_inside_roles = [
            timestamp for timestamp in funding_by_minute
            if any(window["start_ms"] * 1_000_000 <= timestamp < window["end_ms"] * 1_000_000
                   for window in windows)
        ]
        if unassigned_inside_roles:
            raise ValueError(f"Funding events inside reliable windows were not assigned: {unassigned_inside_roles[:3]}")

        audit_sha = sha256_file(audit_path)
        minute_manifest_sha = sha256_file(minute_root / "manifest.json")
        feature_manifest_sha = sha256_file(feature_root / "manifest.json")
        per_role = {}
        for role in ("train", "validation", "test", "oos"):
            role_artifacts = [item for item in artifacts if item["role"] == role]
            per_role[role] = {"window_count": len(role_artifacts),
                              "minute_count": sum(item["row_count"] for item in role_artifacts)}
        root_manifest = {
            "schema": SCHEMA,
            "version": ADAPTER_VERSION,
            "symbol": "BTCUSDT",
            "market": "USD_M",
            "window_time_basis": "historical Binance Vision exchange event time; no local receive/availability time exists",
            "funding_time_basis": "fundingTime event timestamp; actual local publication time is unknown",
            "decision_rule": "features use only completed market records before decision_at_ns; execution ledger describes the following minute",
            "source_manifests": {
                "bulk_audit_sha256": audit_sha,
                "rust_minute_manifest_sha256": minute_manifest_sha,
                "rust_feature_manifest_sha256": feature_manifest_sha,
                "rust_feature_source_manifest_sha256": feature_manifest.get("source_manifest_sha256"),
            },
            "adapter": {"path": Path(__file__).name,
                        "version": ADAPTER_VERSION,
                        "sha256": adapter_code_sha256()},
            "adapter_configuration_sha256": hashlib.sha256(canonical({
                "schema": SCHEMA,
                "version": ADAPTER_VERSION,
                "feature_names": FEATURE_NAMES,
                "reliable_windows": windows,
                "funding_mapping": "fundingTime_ms_floor_to_utc_minute",
                "availability_basis": "historical_exchange_event_time_no_receive_timestamp",
            })).hexdigest(),
            "funding_sources": funding_files,
            "feature_names": FEATURE_NAMES,
            "gap_policy": "only audited reliable half-open windows are emitted; no filling or interpolation",
            "roles": per_role,
            "windows": artifacts,
            "totals": {"feature_source_minute_rows": total_rows,
                       "reliable_rows": ready_rows,
                       "funding_source_rows": sum(item["row_count"] for item in funding_files),
                       "funding_rows_assigned_to_reliable_windows": assigned_funding_rows,
                       "funding_rows_excluded_by_reliability_windows": sum(
                           len(rows) for rows in funding_by_minute.values())},
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(canonical(root_manifest) + b"\n")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        # Independently re-read the produced tree and its compressed files before publication.
        verify_artifact(stage)
        for path in stage.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted((p for p in stage.rglob("*") if p.is_dir()),
                           key=lambda item: len(item.parts), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return root_manifest


def verify_artifact(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json", "historical input manifest")
    if manifest.get("schema") != SCHEMA or manifest.get("version") != 1:
        raise ValueError("unsupported historical input artifact")
    expected = {"manifest.json"}
    total = 0
    for window in manifest.get("windows", []):
        if ((window.get("end_ms", 0) - window.get("start_ms", 0)) // MINUTE_MS
                != window.get("row_count")):
            raise ValueError(f"historical window duration/count mismatch: {window.get('id')}")
        folder = root / "windows" / window["id"]
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError(f"missing historical role window {window['id']}")
        feature_file = folder / window["features"]["path"]
        ledger_file = folder / window["ledger_inputs"]["path"]
        for path, entry in ((feature_file, window["features"]), (ledger_file, window["ledger_inputs"])):
            if path.is_symlink() or not path.is_file() or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"historical role file hash mismatch: {path}")
            if path.stat().st_size != entry["size_bytes"]:
                raise ValueError(f"historical role file size mismatch: {path}")
            expected.add(str(path.relative_to(root)))
        expected_time = window["start_ms"] * 1_000_000
        count = 0
        for feature_row, ledger_row in itertools.zip_longest(
                iter_jsonl_zst(feature_file), iter_jsonl_zst(ledger_file)):
            if feature_row is None or ledger_row is None:
                raise ValueError(f"historical feature/ledger row count mismatch: {window['id']}")
            timestamp = feature_row.get("decision_at_ns")
            if (timestamp != expected_time or ledger_row.get("minute_open_ns") != timestamp
                    or feature_row.get("feature_ready") is not True
                    or feature_row.get("decision_due") is not (timestamp % (15 * MINUTE_NS) == 0)
                    or feature_row.get("time_basis") != "historical_exchange_event_time_no_receive_timestamp"
                    or set(feature_row.get("features", {})) != set(FEATURE_NAMES)):
                raise ValueError(f"invalid/misaligned historical feature and ledger row in {window['id']}")
            settlements = ledger_row.get("funding_settlements")
            if not isinstance(settlements, list):
                raise ValueError(f"Funding settlements missing in {window['id']}")
            for settlement in settlements:
                if (settlement.get("available_at_ns") is not None
                        or settlement.get("availability_basis") != "historical_settlement_timestamp_only"):
                    raise ValueError(f"historical Funding availability overstated in {window['id']}")
            expected_time += MINUTE_NS
            count += 1
        if count != window["row_count"] or expected_time != window["end_ms"] * 1_000_000:
            raise ValueError(f"historical rows do not cover exact range: {window['id']}")
        total += count
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("historical input artifact contains missing or unmanifested files")
    if total != manifest.get("totals", {}).get("reliable_rows"):
        raise ValueError("historical input root row total mismatch")
    return manifest


def iter_jsonl_zst(path: Path):
    with path.open("rb") as stream:
        reader = zstandard.ZstdDecompressor().stream_reader(stream)
        buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)
        for line in buffered:
            if not line.endswith(b"\n"):
                raise ValueError(f"truncated historical row: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"historical row must be a JSON object: {path}")
            yield value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minute-dataset", type=Path,
                        default=Path("/mnt/data/td-market-data-local/research-inputs/vision-minute-engine-v1"))
    parser.add_argument("--features", type=Path,
                        default=Path("/mnt/data/td-market-data-local/research-inputs/vision-rl-features-rust-v4"))
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit.resolve(strict=True)
    minute_root = args.minute_dataset.resolve(strict=True)
    feature_root = args.features.resolve(strict=True)
    raw_root = args.raw_root.resolve(strict=True)
    result = process(audit_path, minute_root, feature_root, raw_root, args.output)
    print(f"historical RL input set published: {args.output.absolute()}")
    print(f"reliable rows: {result['totals']['reliable_rows']}")
    print(f"Funding settlements assigned: {result['totals']['funding_rows_assigned_to_reliable_windows']}")


if __name__ == "__main__":
    main()
