#!/usr/bin/env python3
"""Python orchestration wrapper for the Rust Binance Vision minute engine."""

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
MANIFEST_SCHEMA = "td_market_data_vision_minute_dataset"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reject_unsafe_output(path: Path) -> None:
    resolved = path.resolve()
    if path.exists() or path.is_symlink():
        raise ValueError(f"output must be a new path: {path}")
    if any(root == resolved or root in resolved.parents for root in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("output must be on the data volume, not system temporary storage")
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("output must be outside the source repository")


def expected_month_count(start: str, end: str) -> int:
    def parse(value: str) -> tuple[int, int]:
        try:
            year_text, month_text = value.split("-")
            year, month = int(year_text), int(month_text)
        except (ValueError, TypeError) as error:
            raise ValueError("month boundaries must use YYYY-MM") from error
        if len(year_text) != 4 or len(month_text) != 2 or not 1 <= month <= 12:
            raise ValueError("month boundaries must use YYYY-MM")
        return year, month

    first, last = parse(start), parse(end)
    if first > last:
        raise ValueError("start month is after end month")
    return (last[0] - first[0]) * 12 + last[1] - first[1] + 1


def verify_payload(payload: Path, expected_months: int) -> dict[str, Any]:
    manifest_path = payload / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("version") != 1
            or manifest.get("symbol") != "BTCUSDT" or manifest.get("market") != "USD_M"):
        raise ValueError("Rust engine returned an unsupported manifest")
    months = manifest.get("months")
    if not isinstance(months, list) or len(months) != expected_months:
        raise ValueError("Rust engine did not produce all requested months")
    for month in months:
        if month.get("crc_verified_to_eof") is not True:
            raise ValueError(f"ZIP CRC was not verified for {month.get('month')}")
        if month.get("duplicate_agg_trade_ids") or month.get("missing_agg_trade_ids"):
            raise ValueError(f"aggregate trade ID continuity failed in {month.get('month')}")
        output_file = payload / month["output_path"]
        if output_file.parent != payload or not output_file.is_file():
            raise ValueError(f"Rust output is missing or unsafe: {output_file}")
        if (output_file.stat().st_size != month.get("output_size_bytes")
                or sha256_file(output_file) != month.get("output_sha256")):
            raise ValueError(f"Rust output hash/size mismatch: {output_file.name}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=Path("/mnt/data/td-gateway-datasets/rl-preproduction-v1/raw"))
    parser.add_argument("--start", required=True, help="first YYYY-MM archive month")
    parser.add_argument("--end", required=True, help="last YYYY-MM archive month")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve(strict=True)
    output = args.output.absolute()
    reject_unsafe_output(output)
    month_count = expected_month_count(args.start, args.end)
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.rust-stage-", dir=output.parent))
    payload = stage / "payload"
    try:
        subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path",
                        str(REPO_ROOT / "bulk_engine" / "Cargo.toml")],
                       cwd=REPO_ROOT, check=True)
        binary = REPO_ROOT / "bulk_engine" / "target" / "release" / "td-market-data-bulk-engine"
        command = [str(binary), "process", "--raw-root", str(raw_root), "--output", str(payload),
                   "--start", args.start, "--end", args.end]
        completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        if completed.stderr:
            print(completed.stderr, end="")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or f"Rust engine exited {completed.returncode}")
        manifest = verify_payload(payload, month_count)
        os.rename(payload, output)
        shutil.rmtree(stage)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"minute dataset published: {output}")
    print(f"months: {len(manifest['months'])}")
    print(f"mark rows: {manifest['totals']['mark_rows']}")
    print(f"aggTrade rows: {manifest['totals']['aggtrade_rows']}")


if __name__ == "__main__":
    main()
