#!/usr/bin/env python3
"""Run and independently re-run the existing offline USD-M account replay.

This adapter does not implement accounting or connect to an exchange. It binds
all immutable inputs and the replay executable to the deterministic report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


MAX_INPUT_JSON_BYTES = 8 * 1024 * 1024
MAX_REPORT_BYTES = 128 * 1024 * 1024
MAX_INVENTORY_FILES = 1_000_000


class ReplayBundleRejected(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_absolute_regular(path: Path, name: str, max_bytes: int | None = None) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ReplayBundleRejected(f"{name} must be an absolute regular non-symlink file")
    if max_bytes is not None and path.stat().st_size > max_bytes:
        raise ReplayBundleRejected(f"{name} exceeds the {max_bytes}-byte limit")
    return path.resolve(strict=True)


def dataset_inventory(path: Path, name: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ReplayBundleRejected(f"{name} must be an absolute non-symlink Dataset directory")
    root = path.resolve(strict=True)
    files: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ReplayBundleRejected(f"{name} contains a symlink: {candidate.relative_to(root)}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ReplayBundleRejected(f"{name} contains a non-regular entry: {candidate.relative_to(root)}")
        relative = candidate.relative_to(root).as_posix()
        size = candidate.stat().st_size
        total_bytes += size
        files.append({"path": relative, "size_bytes": size, "sha256": sha256_file(candidate)})
        if len(files) > MAX_INVENTORY_FILES:
            raise ReplayBundleRejected(f"{name} exceeds the file inventory limit")
    if not files or not any(row["path"] == "manifest.json" for row in files):
        raise ReplayBundleRejected(f"{name} has no manifest.json")
    inventory_bytes = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"path": str(root), "file_count": len(files), "total_bytes": total_bytes,
            "files": files, "inventory_sha256": hashlib.sha256(
                b"td-usdm-replay-input-inventory-v1\0" + inventory_bytes).hexdigest()}


def input_file(path: Path, name: str) -> dict[str, Any]:
    resolved = require_absolute_regular(path, name, MAX_INPUT_JSON_BYTES)
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayBundleRejected(f"{name} is not valid JSON") from error
    return {"path": str(resolved), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "json_type": type(value).__name__}


def read_manifest(path: Path, name: str) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReplayBundleRejected(f"{name} manifest must be a regular file")
    if manifest_path.stat().st_size > MAX_INPUT_JSON_BYTES:
        raise ReplayBundleRejected(f"{name} manifest exceeds the JSON input limit")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayBundleRejected(f"{name} manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise ReplayBundleRejected(f"{name} manifest must be an object")
    return manifest


def verify_input_compatibility(depth_path: Path, mark_path: Path,
                               funding_path: Path) -> dict[str, Any]:
    depth = read_manifest(depth_path, "depth Dataset")
    mark = read_manifest(mark_path, "Mark Dataset")
    funding = read_manifest(funding_path, "Funding Dataset")
    if (depth.get("schema") != "td_gateway_usdm_depth_dataset" or depth.get("market") != "USD_M"
            or depth.get("contract") != "PERPETUAL" or depth.get("replay_usable") is not True):
        raise ReplayBundleRejected("depth Dataset is not a replay-usable USD-M perpetual Dataset")
    for name, manifest, expected_kind in (("Mark", mark, "mark_price_observation"),
                                           ("Funding", funding, "funding_rate")):
        if (manifest.get("schema") != "td_gateway_futures_supplemental_dataset"
                or manifest.get("market") != "USD_M" or manifest.get("contract") != "PERPETUAL"
                or manifest.get("kind") != expected_kind):
            raise ReplayBundleRejected(f"{name} Dataset has the wrong market/schema/kind")
    if mark.get("replay_usable") is not True or mark.get("coverage_known") is not True:
        raise ReplayBundleRejected("Mark Dataset lacks complete point-in-time replay coverage")
    if funding.get("settlement_usable") is not True:
        raise ReplayBundleRejected("Funding Dataset lacks verified finalized settlement facts")
    symbol = depth.get("symbol")
    if not isinstance(symbol, str) or mark.get("symbol") != symbol or funding.get("symbol") != symbol:
        raise ReplayBundleRejected("depth, Mark, and Funding symbols differ")
    if symbol != "BTCUSDT":
        raise ReplayBundleRejected("the current USD-M account replay supports BTCUSDT only")
    window = depth.get("window")
    if not isinstance(window, dict) or mark.get("window") != window or funding.get("window") != window:
        raise ReplayBundleRejected("depth, Mark, and Funding windows differ")
    start, end = window.get("start_ns"), window.get("end_ns")
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
            or isinstance(end, bool) or start >= end):
        raise ReplayBundleRejected("Dataset window is invalid")
    return {"symbol": symbol, "window": window}


def create_only_json(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise ReplayBundleRejected("evidence output must be a new absolute path in an existing directory")
    resolved_parent = path.parent.resolve(strict=True)
    destination = resolved_parent / path.name
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=resolved_parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        temporary.unlink()
        parent_fd = os.open(resolved_parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def cli_command(executable: Path, depth: Path, mark: Path, funding: Path, config: Path,
                orders: Path, output: Path, selection: Path | None, action: str) -> list[str]:
    command = [str(executable), "--depth", str(depth), "--mark", str(mark), "--funding", str(funding),
               "--config", str(config), "--orders", str(orders)]
    if selection is not None:
        command += ["--selection-manifest", str(selection)]
    command += ["--output", str(output)] if action == "write" else ["--verify-report", str(output)]
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    executable = require_absolute_regular(args.replay_executable, "replay executable")
    if not os.access(executable, os.X_OK):
        raise ReplayBundleRejected("replay executable is not executable")
    depth_path = args.depth.resolve(strict=True)
    mark_path = args.mark.resolve(strict=True)
    funding_path = args.funding.resolve(strict=True)
    depth_before = dataset_inventory(depth_path, "depth Dataset")
    mark_before = dataset_inventory(mark_path, "Mark Dataset")
    funding_before = dataset_inventory(funding_path, "Funding Dataset")
    replay_identity = verify_input_compatibility(depth_path, mark_path, funding_path)
    config_before = input_file(args.config, "config")
    orders_before = input_file(args.orders, "orders")
    selection_path = None
    selection_before = None
    if args.selection_manifest is not None:
        selection_path = require_absolute_regular(args.selection_manifest, "selection manifest",
                                                  MAX_INPUT_JSON_BYTES)
        selection_before = input_file(selection_path, "selection manifest")
    executable_hash = sha256_file(executable)
    report_path = args.report
    evidence_path = args.evidence
    if not report_path.is_absolute() or not report_path.parent.is_dir() or report_path.exists():
        raise ReplayBundleRejected("report must be a new absolute path in an existing directory")
    if not evidence_path.is_absolute() or not evidence_path.parent.is_dir() or evidence_path.exists():
        raise ReplayBundleRejected("evidence must be a new absolute path in an existing directory")
    if report_path.resolve() == evidence_path.resolve():
        raise ReplayBundleRejected("report and evidence output paths must differ")
    for output_path in (report_path.resolve(), evidence_path.resolve()):
        if any(root in output_path.parents for root in (depth_path, mark_path, funding_path)):
            raise ReplayBundleRejected("report/evidence outputs must be outside immutable input Datasets")

    command = cli_command(executable, depth_path, mark_path, funding_path, args.config.resolve(strict=True),
                          args.orders.resolve(strict=True), report_path, selection_path, "write")
    first = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if first.returncode != 0:
        raise ReplayBundleRejected(f"USD-M replay failed ({first.returncode}): {first.stderr[-4000:]}")
    require_absolute_regular(report_path, "replay report", MAX_REPORT_BYTES)
    try:
        report = json.loads(report_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayBundleRejected("replay report is unreadable or invalid JSON") from error
    if not isinstance(report, dict) or report.get("schema") != "td_gateway_futures_depth_comparative_replay_v2":
        raise ReplayBundleRejected("replay output has an unexpected comparison report schema")

    verify_command = cli_command(executable, depth_path, mark_path, funding_path,
                                 args.config.resolve(strict=True), args.orders.resolve(strict=True),
                                 report_path, selection_path, "verify")
    verification = subprocess.run(verify_command, check=False, text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if verification.returncode != 0:
        raise ReplayBundleRejected(f"deterministic replay verification failed ({verification.returncode}): "
                                   f"{verification.stderr[-4000:]}")
    report_hash_after_verification = sha256_file(report_path)

    depth_after = dataset_inventory(depth_path, "depth Dataset")
    mark_after = dataset_inventory(mark_path, "Mark Dataset")
    funding_after = dataset_inventory(funding_path, "Funding Dataset")
    config_after = input_file(args.config, "config")
    orders_after = input_file(args.orders, "orders")
    if (depth_before != depth_after or mark_before != mark_after or funding_before != funding_after
            or config_before != config_after or orders_before != orders_after
            or (selection_path is not None and input_file(selection_path, "selection manifest") != selection_before)
            or sha256_file(executable) != executable_hash):
        raise ReplayBundleRejected("an input or replay executable changed during the replay")

    report_hash = sha256_file(report_path)
    if report_hash != report_hash_after_verification:
        raise ReplayBundleRejected("comparison report changed after deterministic verification")
    candidate = report.get("candidate")
    candidate_execution = candidate.get("execution") if isinstance(candidate, dict) else None
    if not isinstance(candidate_execution, dict) or not isinstance(candidate_execution.get("symbol"), str):
        raise ReplayBundleRejected("replay report lacks the candidate execution identity")
    evidence = {"schema": "td_market_data_usdm_replay_evidence", "version": 1,
                "symbol": candidate_execution.get("symbol"), "window": replay_identity["window"],
                "replay_executable": {"path": str(executable), "sha256": executable_hash},
                "inputs": {"depth": depth_after, "mark": mark_after, "funding": funding_after,
                           "config": config_after, "orders": orders_after,
                           "selection_manifest": selection_before},
                "report": {"path": str(report_path.resolve()), "size_bytes": report_path.stat().st_size,
                           "sha256": report_hash,
                           "comparison_fingerprint_sha256": report.get("comparison_fingerprint_sha256")},
                "deterministic_rerun": {"passed": True, "stdout": verification.stdout.strip()}}
    create_only_json(evidence_path, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-executable", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--mark", required=True, type=Path)
    parser.add_argument("--funding", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--orders", required=True, type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    try:
        evidence = run(args)
    except (ReplayBundleRejected, OSError, ValueError) as error:
        raise SystemExit(f"USD-M verified replay rejected: {error}") from error
    print("USD-M account replay and deterministic rerun passed")
    print(f"report sha256: {evidence['report']['sha256']}")
    print(f"comparison fingerprint: {evidence['report']['comparison_fingerprint_sha256']}")
    print(f"evidence: {args.evidence}")


if __name__ == "__main__":
    main()
