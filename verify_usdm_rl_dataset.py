#!/usr/bin/env python3
"""Independently verify a fixed USD-M Relay research Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from verify_usdm_aggtrades_dataset import DatasetRejected as AggRejected, verify_dataset as verify_agg
from verify_usdm_funding_dataset import DatasetRejected as FundingRejected, verify_dataset as verify_funding
from verify_usdm_klines_dataset import DatasetRejected as KlinesRejected, verify_dataset as verify_klines


SCHEMA = "td_market_data_usdm_rl_dataset"
REQUIRED = {"usdm-funding-mark", "usdm-aggtrades", "usdm-finalized-funding"}
OPTIONAL = {"usdm-klines-1m"}


class DatasetRejected(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DatasetRejected(f"{field} must be a lowercase SHA-256")
    return value


def verify_dataset(dataset: Path, source_captures: list[tuple[str, Path]] | None = None) -> dict[str, Any]:
    dataset = dataset.resolve(strict=True)
    if dataset.is_symlink() or not dataset.is_dir():
        raise DatasetRejected("Dataset must be a regular directory")
    manifest_path = dataset / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DatasetRejected("Dataset manifest is missing or unsafe")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetRejected("Dataset manifest is invalid JSON") from error
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or manifest.get("version") != 1 or manifest.get("exchange") != "BINANCE"
            or manifest.get("market") != "USD_M"):
        raise DatasetRejected("unsupported USD-M RL Dataset manifest")
    symbol = manifest.get("symbol")
    if not isinstance(symbol, str) or not symbol.isascii() or not symbol.isalnum():
        raise DatasetRejected("Dataset symbol is invalid")
    window = manifest.get("window")
    if (not isinstance(window, dict) or not isinstance(window.get("start_ns"), int)
            or isinstance(window.get("start_ns"), bool) or not isinstance(window.get("end_ns"), int)
            or isinstance(window.get("end_ns"), bool) or window["start_ns"] >= window["end_ns"]
            or window.get("closed") != "[start,end)"):
        raise DatasetRejected("Dataset window is invalid")
    required_streams = manifest.get("required_streams")
    if (not isinstance(required_streams, list) or set(required_streams) != REQUIRED
            or len(required_streams) != len(REQUIRED)):
        raise DatasetRejected("required stream contract mismatch")
    optional = manifest.get("optional_streams")
    if not isinstance(optional, dict) or set(optional) != OPTIONAL or not isinstance(optional.get("usdm-klines-1m"), bool):
        raise DatasetRejected("optional stream inventory is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DatasetRejected("source inventory is missing")
    seen: set[tuple[str, str]] = set()
    hashes: list[str] = []
    by_stream: dict[str, int] = {}
    expected_files = {"manifest.json"}
    for source in sources:
        if not isinstance(source, dict):
            raise DatasetRejected("source entry is invalid")
        stream, capture_id = source.get("stream"), source.get("capture_id")
        if stream not in REQUIRED | OPTIONAL or not isinstance(capture_id, str) or re.fullmatch(r"[0-9a-f]{32}", capture_id) is None:
            raise DatasetRejected("source stream/capture identity is invalid")
        if (stream, capture_id) in seen:
            raise DatasetRejected("duplicate capture identity in source inventory")
        seen.add((stream, capture_id))
        by_stream[stream] = by_stream.get(stream, 0) + 1
        hashes.append(require_hash(source.get("compressed_sha256"), "compressed archive hash"))
        require_hash(source.get("raw_sha256"), "raw capture hash")
        if source.get("origin") == "vps_recovered":
            recovered = source.get("recovered_at_unix_ns")
            if not isinstance(recovered, int) or isinstance(recovered, bool) or recovered <= 0:
                raise DatasetRejected("recovered source lacks its local recovery time")
        elif source.get("origin") != "local" or source.get("recovered_at_unix_ns") is not None:
            raise DatasetRejected("source origin/recovery time is invalid")
        started, stopped = source.get("capture_started_at_ns"), source.get("capture_stopped_at_ns")
        if (not isinstance(started, int) or isinstance(started, bool)
                or not isinstance(stopped, int) or isinstance(stopped, bool) or stopped <= started):
            raise DatasetRejected("source lifecycle boundaries are invalid")
    if not REQUIRED.issubset(by_stream) or bool(by_stream.get("usdm-klines-1m")) != optional["usdm-klines-1m"]:
        raise DatasetRejected("source stream inventory does not match required/optional declaration")
    expected_set = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("source_capture_set_sha256") != expected_set:
        raise DatasetRejected("source capture-set hash mismatch")

    datasets = manifest.get("datasets")
    expected_dataset_names = REQUIRED | ({"usdm-klines-1m"} if optional["usdm-klines-1m"] else set())
    if not isinstance(datasets, dict) or set(datasets) != expected_dataset_names:
        raise DatasetRejected("child Dataset inventory mismatch")
    for name, entry in datasets.items():
        if name == "usdm-funding-mark":
            evidence = entry.get("evidence") if isinstance(entry, dict) else None
            if not isinstance(evidence, list) or not evidence:
                raise DatasetRejected("Mark evidence inventory is missing")
            for item in evidence:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise DatasetRejected("Mark evidence entry is invalid")
                path = dataset / item["path"]
                if path.parent != dataset or path.is_symlink() or not path.is_file():
                    raise DatasetRejected("Mark evidence file is missing or unsafe")
                if sha256_file(path) != require_hash(item.get("sha256"), "Mark evidence hash"):
                    raise DatasetRejected("Mark evidence hash mismatch")
                if item.get("size_bytes") != path.stat().st_size:
                    raise DatasetRejected("Mark evidence size mismatch")
                with path.open("rb") as stream:
                    try:
                        header = json.loads(stream.readline())
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise DatasetRejected("Mark evidence header is invalid") from error
                if (not isinstance(header, dict) or header.get("record_type") != "capture_started"
                        or header.get("source_capture_sha256") is None):
                    raise DatasetRejected("Mark evidence source identity is missing")
                matching = [source for source in sources
                            if source["stream"] == "usdm-funding-mark"
                            and source["capture_id"] == header.get("capture_id")]
                if (len(matching) != 1 or matching[0]["raw_sha256"] != header["source_capture_sha256"]
                        or matching[0]["origin"] != header.get("origin")
                        or matching[0]["recovered_at_unix_ns"] != header.get("recovered_at_unix_ns")):
                    raise DatasetRejected("Mark evidence source provenance mismatch")
                expected_files.add(path.name)
            continue
        child = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(child, str) or Path(child).name != child:
            raise DatasetRejected(f"child Dataset path invalid: {name}")
        child_path = dataset / child
        child_manifest = child_path / "manifest.json"
        if child_path.is_symlink() or not child_path.is_dir() or child_manifest.is_symlink() or not child_manifest.is_file():
            raise DatasetRejected(f"child Dataset missing or unsafe: {name}")
        if sha256_file(child_manifest) != require_hash(entry.get("manifest_sha256"), f"{name} manifest hash"):
            raise DatasetRejected(f"child Dataset manifest hash mismatch: {name}")
        try:
            child_data = verify_agg(child_path) if name == "usdm-aggtrades" else (
                verify_funding(child_path) if name == "usdm-finalized-funding" else verify_klines(child_path))
        except (AggRejected, FundingRejected, KlinesRejected, OSError, ValueError) as error:
            raise DatasetRejected(f"child Dataset failed independent verification ({name}): {error}") from error
        if child_data.get("symbol") != symbol or child_data.get("window") != {
                "start_ns": window["start_ns"], "end_ns": window["end_ns"]}:
            raise DatasetRejected(f"child Dataset window/symbol differs from root: {name}")
        if name == "usdm-aggtrades":
            child_sources = child_data.get("capture_sources", [])
        elif name == "usdm-finalized-funding":
            child_sources = child_data.get("mark_capture_sources", []) + child_data.get("funding_capture_sources", [])
        else:
            child_sources = child_data.get("capture_sources", [])
        child_set = {(item.get("capture_id"), item.get("sha256")) for item in child_sources}
        root_set = {(item["capture_id"], item["raw_sha256"])
                    for item in sources if item["stream"] in (
                        {"usdm-aggtrades"} if name == "usdm-aggtrades" else
                        {"usdm-funding-mark", "usdm-finalized-funding"} if name == "usdm-finalized-funding" else
                        {name})}
        if child_set != root_set:
            raise DatasetRejected(f"child Dataset source set differs from root: {name}")
        expected_files.add(child)
    if {path.name for path in dataset.iterdir()} != expected_files:
        raise DatasetRejected("Dataset contains missing or unmanifested files")
    if source_captures is not None:
        expected = {(row["stream"], row["compressed_sha256"]) for row in sources}
        actual = set()
        for stream, path in source_captures:
            if path.is_symlink() or not path.is_file():
                raise DatasetRejected(f"provided source capture is unsafe: {path}")
            actual.add((stream, sha256_file(path)))
        if actual != expected:
            raise DatasetRejected("provided source captures do not match source inventory")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--source-capture", action="append", default=[], metavar="STREAM=PATH")
    args = parser.parse_args()
    supplied = []
    for value in args.source_capture:
        if "=" not in value:
            raise SystemExit("--source-capture must use STREAM=PATH")
        stream, path = value.split("=", 1)
        supplied.append((stream, Path(path)))
    try:
        manifest = verify_dataset(args.dataset, supplied if supplied else None)
    except (DatasetRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M RL Dataset rejected: {error}") from error
    print(f"USD-M RL Dataset valid: sources={len(manifest['sources'])} window="
          f"[{manifest['window']['start']},{manifest['window']['end']})")


if __name__ == "__main__":
    main()
