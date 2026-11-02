#!/usr/bin/env python3
"""Derive causal historical RL features with the Rust streaming engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_SCHEMA = "td_market_data_usdm_rl_historical_feature_set"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_locations(input_root: Path, output: Path) -> None:
    resolved = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new path: {output}")
    if any(root == resolved or root in resolved.parents for root in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("output must be on the data volume, not system temporary storage")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("output must be outside the source repository")
    resolved_input = input_root.resolve(strict=True)
    if resolved == resolved_input or resolved_input in resolved.parents:
        raise ValueError("output must not be inside the immutable input dataset")
    manifest = json.loads((input_root / "manifest.json").read_bytes())
    if (manifest.get("schema") != "td_market_data_vision_minute_dataset"
            or manifest.get("version") != 1 or manifest.get("market") != "USD_M"):
        raise ValueError("unsupported Rust minute dataset")


def verify_output(payload: Path, input_manifest: dict[str, Any],
                  input_manifest_sha256: str) -> dict[str, Any]:
    manifest = json.loads((payload / "manifest.json").read_bytes())
    if (manifest.get("schema") != OUTPUT_SCHEMA or manifest.get("version") != 1
            or manifest.get("symbol") != input_manifest.get("symbol")
            or manifest.get("market") != input_manifest.get("market")
            or manifest.get("source_manifest_sha256") != input_manifest_sha256
            or manifest.get("feature_schema") != "td_futures_research_features"
            or manifest.get("feature_schema_version") != 1):
        raise ValueError("Rust feature output manifest does not match its input")
    source_months = input_manifest.get("months")
    months = manifest.get("months")
    if not isinstance(source_months, list) or not isinstance(months, list) or len(months) != len(source_months):
        raise ValueError("Rust feature engine omitted requested month shards")
    expected_names = [
        "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
        "mark_realized_volatility_15m", "mark_realized_volatility_60m",
        "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
    ]
    if manifest.get("feature_names") != expected_names:
        raise ValueError("Rust feature schema order differs from Relay's frozen feature contract")
    for source, month in zip(source_months, months, strict=True):
        if (month.get("month") != source.get("month")
                or month.get("input_path") != source.get("output_path")
                or month.get("input_sha256") != source.get("output_sha256")):
            raise ValueError(f"feature shard source binding mismatch for {month.get('month')}")
        if month.get("not_ready_rows") != month.get("minute_rows") - month.get("feature_ready_rows"):
            raise ValueError(f"invalid readiness counts for {month.get('month')}")
        expected_rows = source.get("mark_rows", 0) + sum(
            max(0, (window["end_ms"] - window["start_ms"]) // 60_000)
            for window in source.get("missing_mark_windows", [])
        )
        if month.get("minute_rows") != expected_rows:
            raise ValueError(f"input and feature minute counts differ for {month.get('month')}")
        output_file = payload / month["output_path"]
        if output_file.parent != payload or not output_file.is_file():
            raise ValueError(f"missing or unsafe feature output: {output_file}")
        if (output_file.stat().st_size != month.get("output_size_bytes")
                or sha256_file(output_file) != month.get("output_sha256")):
            raise ValueError(f"feature output hash/size mismatch: {output_file.name}")
    totals = manifest.get("totals", {})
    if totals.get("minute_rows") != sum(row["minute_rows"] for row in months):
        raise ValueError("feature manifest minute totals do not reconcile")
    if totals.get("feature_ready_rows") != sum(row["feature_ready_rows"] for row in months):
        raise ValueError("feature manifest readiness totals do not reconcile")
    if totals.get("not_ready_rows") != sum(row["not_ready_rows"] for row in months):
        raise ValueError("feature manifest unready totals do not reconcile")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Rust Vision minute dataset directory")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    input_root = args.input.resolve(strict=True)
    output = args.output.absolute()
    validate_locations(input_root, output)
    input_manifest_path = input_root / "manifest.json"
    input_manifest_bytes = input_manifest_path.read_bytes()
    input_manifest = json.loads(input_manifest_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.feature-stage-", dir=output.parent))
    payload = stage / "payload"
    try:
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(REPO_ROOT / "bulk_engine" / "Cargo.toml")],
                       cwd=REPO_ROOT, check=True)
        binary = REPO_ROOT / "bulk_engine" / "target" / "release" / "td-market-data-bulk-engine"
        completed = subprocess.run(
            [str(binary), "derive-features", "--input", str(input_root), "--output", str(payload)],
            cwd=REPO_ROOT, text=True, capture_output=True,
        )
        if completed.stderr:
            print(completed.stderr, end="")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"Rust feature engine exited {completed.returncode}")
        manifest = json.loads((payload / "manifest.json").read_bytes())
        source_manifest_sha256 = hashlib.sha256(input_manifest_bytes).hexdigest()
        if manifest.get("source_manifest_sha256") != source_manifest_sha256:
            raise ValueError("Rust feature output was derived from another input manifest")
        verified = verify_output(payload, input_manifest, source_manifest_sha256)
        os.rename(payload, output)
        shutil.rmtree(stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"feature shards published: {output}")
    print(f"months: {len(verified['months'])}")
    print(f"minutes: {verified['totals']['minute_rows']}")
    print(f"feature-ready: {verified['totals']['feature_ready_rows']}")


if __name__ == "__main__":
    main()
