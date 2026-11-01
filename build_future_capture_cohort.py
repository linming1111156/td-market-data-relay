#!/usr/bin/env python3
"""Build an immutable multi-stream Dataset cohort from closed local archives.

The input files are the relay's compressed raw captures. They are hash-checked,
decoded into a stable sibling source directory, passed to the existing strict
stream exporters, and bound together by a cohort manifest. The final output is
create-only so a published cohort cannot be silently replaced.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

STREAMS = (
    "spot-depth",
    "spot-klines-1m",
    "spot-trades",
    "usdm-aggtrades",
    "usdm-depth",
    "usdm-finalized-funding",
    "usdm-funding-mark",
    "usdm-klines-1m",
)
SCHEMA = "td_market_data_future_capture_cohort_v1"
MAX_LINE_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_time(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if parsed.tzinfo != dt.timezone.utc:
        raise argparse.ArgumentTypeError("timestamp must use UTC")
    return parsed


def ns(value: dt.datetime) -> int:
    return int(value.timestamp()) * 1_000_000_000 + value.microsecond * 1_000


def fmt_ns(value: int) -> str:
    return dt.datetime.fromtimestamp(value / 1_000_000_000, dt.timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def write_decoded_capture(compressed: Path, sidecar: Path, destination: Path,
                          start_ns: int, end_ns: int) -> dict[str, Any]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("Python package zstandard is required") from error

    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("schema") != "td_market_data_archive_metadata_v1":
        raise RuntimeError(f"unsupported archive metadata: {sidecar}")
    if sha256_file(compressed) != metadata.get("compressed_sha256"):
        raise RuntimeError(f"compressed archive hash mismatch: {compressed}")

    raw_hash = hashlib.sha256()
    record_types: dict[str, int] = {}
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    line_number = 0
    pending = b""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with compressed.open("rb") as source, destination.open("xb") as output:
        reader = zstandard.ZstdDecompressor().stream_reader(source)
        try:
            while block := reader.read(1024 * 1024):
                raw_hash.update(block)
                output.write(block)
                pending += block
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    line_number += 1
                    if len(raw) > MAX_LINE_BYTES:
                        raise RuntimeError(f"oversized capture record {line_number}: {compressed}")
                    row = json.loads(raw)
                    if not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record":
                        raise RuntimeError(f"invalid capture record {line_number}: {compressed}")
                    if first is None:
                        first = row
                    last = row
                    kind = row.get("record_type")
                    record_types[kind] = record_types.get(kind, 0) + 1
        finally:
            reader.close()
        output.flush()
        os.fsync(output.fileno())

    if pending:
        raise RuntimeError(f"capture does not end with a newline: {compressed}")
    if raw_hash.hexdigest() != metadata.get("raw_sha256"):
        raise RuntimeError(f"decoded raw hash mismatch: {compressed}")
    if destination.stat().st_size != metadata.get("raw_size_bytes"):
        raise RuntimeError(f"decoded raw size mismatch: {compressed}")
    if (not first or first.get("record_type") != "capture_started"
            or not last or last.get("record_type") != "capture_stopped"):
        raise RuntimeError(f"capture is not cleanly closed: {compressed}")
    started_ns = first.get("recorded_at_unix_ns")
    stopped_ns = last.get("recorded_at_unix_ns")
    if not isinstance(started_ns, int) or not isinstance(stopped_ns, int):
        raise RuntimeError(f"capture lifecycle timestamps are invalid: {compressed}")
    if started_ns > start_ns or stopped_ns < start_ns:
        raise RuntimeError(f"capture does not cover the requested window start: {compressed}")
    fatal = {kind: record_types.get(kind, 0) for kind in
             ("gap_detected", "clock_discontinuity", "funding_api_error")
             if record_types.get(kind, 0)}
    if fatal:
        raise RuntimeError(f"capture contains integrity errors {fatal}: {compressed}")
    return {
        "capture_id": metadata.get("capture_id"),
        "compressed_path": str(compressed.resolve()),
        "compressed_sha256": metadata["compressed_sha256"],
        "raw_path": str(destination.resolve()),
        "raw_sha256": raw_hash.hexdigest(),
        "raw_size_bytes": destination.stat().st_size,
        "record_count": line_number,
        "record_types": record_types,
        "started_at": fmt_ns(started_ns),
        "stopped_at": fmt_ns(stopped_ns),
    }


def capture_arguments(paths: dict[str, Path], *streams: str) -> list[str]:
    arguments: list[str] = []
    for stream in streams:
        arguments.extend(("--capture", str(paths[stream])))
    return arguments


def run_export(script: Path, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(script), *arguments], check=True)


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def publish_directory(staging: Path, output: Path) -> None:
    libc = __import__("ctypes").CDLL(None, use_errno=True)
    rename_noreplace = getattr(libc, "renameat2", None)
    if rename_noreplace is None:
        raise RuntimeError("atomic create-only publication requires renameat2")
    result = rename_noreplace(-100, os.fsencode(staging), -100, os.fsencode(output), 1)
    if result != 0:
        error_number = __import__("ctypes").get_errno()
        raise OSError(error_number, os.strerror(error_number), str(output))
    fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True,
                        metavar="STREAM=PATH", help="one compressed archive for each supported stream")
    parser.add_argument("--start", required=True, type=parse_time, help="RFC3339 UTC dataset start")
    parser.add_argument("--end", required=True, type=parse_time, help="RFC3339 UTC dataset end")
    parser.add_argument("--output", required=True, type=Path, help="create-only Dataset cohort directory")
    parser.add_argument("--source-output", type=Path,
                        help="stable decoded raw sources (default: OUTPUT-sources)")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    if args.start >= args.end or args.output.exists():
        raise SystemExit("window must be increasing and output must not already exist")
    sources: dict[str, Path] = {}
    for value in args.capture:
        if "=" not in value:
            raise SystemExit(f"invalid --capture value: {value}")
        stream, raw_path = value.split("=", 1)
        if stream not in STREAMS or stream in sources:
            raise SystemExit(f"unsupported or duplicate capture stream: {stream}")
        sources[stream] = Path(raw_path)
    if set(sources) != set(STREAMS):
        raise SystemExit("exactly one compressed capture is required for each of the eight streams")

    source_output = (args.source_output or args.output.with_name(args.output.name + "-sources")).resolve()
    if source_output.is_symlink() or (source_output.exists() and not source_output.is_dir()):
        raise SystemExit(f"source output must be a regular directory: {source_output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_output.parent.mkdir(parents=True, exist_ok=True)
    source_stage = Path(tempfile.mkdtemp(prefix=f".{source_output.name}.staging-",
                                         dir=source_output.parent))
    output_stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.staging-",
                                         dir=args.output.parent))
    # Exporters publish their own child directories create-only.
    start_ns, end_ns = ns(args.start), ns(args.end)
    script_dir = Path(__file__).resolve().parent
    try:
        raw_paths: dict[str, Path] = {}
        capture_records: dict[str, Any] = {}
        for stream in STREAMS:
            if sources[stream].is_symlink():
                raise RuntimeError(f"capture archive cannot be a symlink: {sources[stream]}")
            compressed = sources[stream].resolve(strict=True)
            if not compressed.is_file() or not compressed.name.endswith(".jsonl.zst"):
                raise RuntimeError(f"capture must be a regular .jsonl.zst archive: {compressed}")
            sidecar = compressed.with_suffix(compressed.suffix + ".meta.json")
            if sidecar.is_symlink() or not sidecar.is_file():
                raise RuntimeError(f"capture metadata must be a regular file: {sidecar}")
            raw_name = compressed.name.removesuffix(".zst")
            raw_destination = source_stage / stream / raw_name
            capture_records[stream] = write_decoded_capture(
                compressed, sidecar, raw_destination, start_ns, end_ns)
            raw_paths[stream] = raw_destination

        # Source paths are embedded in exporter manifests. Publish them at their
        # stable final path before running exporters so those references remain valid.
        if source_output.exists():
            for stream in STREAMS:
                prior = source_output / stream / raw_paths[stream].name
                if not prior.is_file() or prior.is_symlink() or sha256_file(prior) != capture_records[stream]["raw_sha256"]:
                    raise RuntimeError(f"existing decoded source differs from archive: {prior}")
            shutil.rmtree(source_stage)
        else:
            publish_directory(source_stage, source_output)
        for stream, record in capture_records.items():
            record["raw_path"] = str((source_output / stream / Path(record["raw_path"]).name).resolve())
        raw_paths = {stream: source_output / stream / path.name for stream, path in raw_paths.items()}

        symbol = args.symbol.upper()
        bounds = ["--symbol", symbol, "--start", args.start.isoformat().replace("+00:00", "Z"),
                  "--end", args.end.isoformat().replace("+00:00", "Z")]
        outputs = {
            "spot-depth": output_stage / "spot-depth",
            "spot-klines-1m": output_stage / "spot-klines-1m",
            "spot-trades": output_stage / "spot-trades",
            "usdm-aggtrades": output_stage / "usdm-aggtrades",
            "usdm-depth": output_stage / "usdm-depth",
            "usdm-finalized-funding": output_stage / "usdm-funding",
            "usdm-funding-mark": output_stage / "usdm-funding-mark-evidence.jsonl",
            "usdm-klines-1m": output_stage / "usdm-klines-1m",
        }
        run_export(script_dir / "export_spot_depth_dataset.py",
                   ["--capture", str(raw_paths["spot-depth"]), "--output", str(outputs["spot-depth"]), *bounds])
        run_export(script_dir / "export_spot_klines_dataset.py",
                   capture_arguments(raw_paths, "spot-klines-1m") + ["--market", "SPOT", "--output",
                                                                      str(outputs["spot-klines-1m"]), *bounds])
        run_export(script_dir / "export_spot_trades_dataset.py",
                   ["--capture", str(raw_paths["spot-trades"]), "--output", str(outputs["spot-trades"]), *bounds])
        run_export(script_dir / "export_usdm_aggtrades_dataset.py",
                   capture_arguments(raw_paths, "usdm-aggtrades") + ["--output", str(outputs["usdm-aggtrades"]), *bounds])
        run_export(script_dir / "export_usdm_depth_dataset.py",
                   ["--capture", str(raw_paths["usdm-depth"]), "--output", str(outputs["usdm-depth"]), *bounds])
        run_export(script_dir / "export_usdm_klines_dataset.py",
                   capture_arguments(raw_paths, "usdm-klines-1m") + ["--output", str(outputs["usdm-klines-1m"]), *bounds])
        run_export(script_dir / "export_funding_capture.py",
                   ["--capture", str(raw_paths["usdm-funding-mark"]), "--output",
                    str(outputs["usdm-funding-mark"])])
        run_export(script_dir / "export_usdm_funding_dataset.py",
                   ["--mark-capture", str(raw_paths["usdm-funding-mark"]), "--funding-capture",
                    str(raw_paths["usdm-finalized-funding"]), "--output",
                    str(outputs["usdm-finalized-funding"]), *bounds])

        dataset_manifests = {}
        for name, path in outputs.items():
            manifest_path = path / "manifest.json" if path.is_dir() else None
            if manifest_path is not None and manifest_path.is_file():
                dataset_manifests[name] = {
                    "manifest_sha256": sha256_file(manifest_path),
                    "manifest": read_manifest(manifest_path),
                }
            else:
                dataset_manifests[name] = {
                    "evidence_sha256": sha256_file(path),
                    "evidence_bytes": path.stat().st_size,
                }
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=script_dir,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        cohort = {
            "schema": SCHEMA,
            "version": 1,
            "exchange": "BINANCE",
            "symbol": symbol,
            "window": {"start": args.start.isoformat().replace("+00:00", "Z"),
                       "end": args.end.isoformat().replace("+00:00", "Z"),
                       "start_unix_ns": start_ns, "end_unix_ns": end_ns},
            "source_revision": revision,
            "sources": capture_records,
            "datasets": dataset_manifests,
            "funding_settlements": dataset_manifests["usdm-finalized-funding"].get("manifest", {}).get("event_count", 0),
            "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        }
        manifest_path = output_stage / "cohort-manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(json.dumps(cohort, sort_keys=True, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_directory(output_stage, args.output.resolve())
        print(f"future capture cohort published: streams={len(capture_records)} funding_settlements={cohort['funding_settlements']}")
        print(f"cohort: {args.output.resolve()}")
        print(f"source captures: {source_output}")
    except BaseException:
        shutil.rmtree(output_stage, ignore_errors=True)
        shutil.rmtree(source_stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
