#!/usr/bin/env python3
"""Fetch and provenance-bind the exact public USD-M Funding settlement row."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SOURCE = "https://fapi.binance.com/fapi/v1/fundingRate"


class FundingEvidenceRejected(ValueError):
    pass


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FundingEvidenceRejected(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def read_schedule(path: Path) -> tuple[int, int]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise FundingEvidenceRejected("schedule must be an absolute regular non-symlink file")
    try:
        schedule = json.loads(path.read_bytes(), object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FundingEvidenceRejected("schedule is unreadable or invalid JSON") from error
    if (not isinstance(schedule, dict)
            or schedule.get("schema") != "td_gateway_usdm_funding_settlement_schedule"
            or schedule.get("version") != 1 or schedule.get("symbol") != "BTCUSDT"):
        raise FundingEvidenceRejected("unsupported schedule schema, version, or symbol")
    expected = schedule.get("expected_settlement_times_ns")
    if (not isinstance(expected, list) or len(expected) != 1
            or not isinstance(expected[0], int) or isinstance(expected[0], bool)
            or expected[0] <= 0 or expected[0] % 1_000_000):
        raise FundingEvidenceRejected("this fetcher requires one millisecond-aligned expected settlement")
    stopped_ns = schedule.get("capture_stopped_at_unix_ns")
    if (not isinstance(stopped_ns, int) or isinstance(stopped_ns, bool)
            or stopped_ns < 0 or expected[0] > stopped_ns):
        raise FundingEvidenceRejected("Mark capture must stop at or after the scheduled settlement")
    return expected[0] // 1_000_000, stopped_ns


def fetch_response(funding_time_ms: int) -> tuple[bytes, int, str]:
    query = urllib.parse.urlencode({"symbol": "BTCUSDT", "startTime": funding_time_ms,
                                   "endTime": funding_time_ms, "limit": 1000})
    url = f"{SOURCE}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        status = response.status
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if status != 200:
        raise FundingEvidenceRejected(f"Funding endpoint returned HTTP {status}")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise FundingEvidenceRejected("Funding response exceeds 4 MiB")
    try:
        records = json.loads(raw, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FundingEvidenceRejected("Funding response is invalid JSON") from error
    if (not isinstance(records, list) or len(records) != 1
            or not isinstance(records[0], dict)
            or records[0].get("symbol") != "BTCUSDT"
            or records[0].get("fundingTime") != funding_time_ms):
        raise FundingEvidenceRejected("response does not contain exactly the scheduled BTCUSDT settlement")
    for name in ("fundingRate", "markPrice"):
        try:
            number = float(records[0][name])
        except (KeyError, TypeError, ValueError) as error:
            raise FundingEvidenceRejected(f"response has invalid {name}") from error
        if not math.isfinite(number) or (name == "markPrice" and number <= 0):
            raise FundingEvidenceRejected(f"response has invalid {name}")
    retrieved_ns = time.time_ns()
    seconds, nanos = divmod(retrieved_ns, 1_000_000_000)
    retrieved_dt = dt.datetime.fromtimestamp(seconds, dt.timezone.utc).replace(
        microsecond=nanos // 1_000)
    retrieved = retrieved_dt.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return raw, status, retrieved, retrieved_ns


def write_evidence(schedule_path: Path, output: Path) -> dict[str, Any]:
    funding_time_ms, stopped_ns = read_schedule(schedule_path)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise FundingEvidenceRejected("output must be a new absolute directory path")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    repository = Path(__file__).resolve().parent
    if output == repository or repository in output.parents:
        raise FundingEvidenceRejected("evidence output must be outside the source repository")
    raw, status, retrieved, retrieved_ns = fetch_response(funding_time_ms)
    if retrieved_ns < max(stopped_ns, funding_time_ms * 1_000_000):
        raise FundingEvidenceRejected("Funding response was retrieved before Mark capture completion or settlement")
    records = json.loads(raw, object_pairs_hook=unique_object)
    request_url = (f"{SOURCE}?" + urllib.parse.urlencode(
        {"symbol": "BTCUSDT", "startTime": funding_time_ms,
         "endTime": funding_time_ms, "limit": 1000}))
    manifest = {"request_url": request_url, "retrieved_at_utc": retrieved,
                "http_status": status, "response_sha256": hashlib.sha256(raw).hexdigest(),
                "response_bytes": len(raw), "record_count": len(records)}
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        for name, data in (("funding-response.json", raw),
                           ("funding-source-manifest.json", json.dumps(
                               manifest, sort_keys=True, indent=2).encode() + b"\n")):
            with (stage / name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        stage_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
        if renameat2 is None or renameat2(-100, os.fsencode(stage), -100, os.fsencode(output), 1) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), str(output))
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = write_evidence(args.schedule, args.output_directory)
    except (FundingEvidenceRejected, OSError, ValueError) as error:
        raise SystemExit(f"finalized Funding evidence rejected: {error}") from error
    print(f"finalized Funding evidence captured: records={manifest['record_count']} "
          f"sha256={manifest['response_sha256']} output={args.output_directory}")


if __name__ == "__main__":
    main()
