#!/usr/bin/env python3
"""Build a create-only USD-M research Dataset from multiple Relay archive segments.

This deliberately has a smaller stream contract than build_future_capture_cohort.py.
Large staging files are created beside the selected output, never under system /tmp.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from capture_recovery import SCHEMA as RECOVERY_SCHEMA


SCHEMA = "td_market_data_usdm_rl_dataset"
REQUIRED = ("usdm-funding-mark", "usdm-aggtrades", "usdm-finalized-funding")
OPTIONAL = ("usdm-klines-1m",)
STREAMS = REQUIRED + OPTIONAL
MAX_DECODED_BYTES = 256 * 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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
        timespec="milliseconds").replace("+00:00", "Z")


def reject_system_temp(path: Path, label: str) -> None:
    resolved = path.resolve()
    if any(resolved == root or root in resolved.parents for root in (Path("/tmp"), Path("/var/tmp"))):
        raise SystemExit(f"{label} must be on the selected data volume, not system temporary storage")


def publish_directory(staging: Path, output: Path) -> None:
    libc = __import__("ctypes").CDLL(None, use_errno=True)
    rename_noreplace = getattr(libc, "renameat2", None)
    if rename_noreplace is None:
        raise RuntimeError("atomic create-only publication requires renameat2")
    if rename_noreplace(-100, os.fsencode(staging), -100, os.fsencode(output), 1) != 0:
        error_number = __import__("ctypes").get_errno()
        raise OSError(error_number, os.strerror(error_number), str(output))
    fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def seal_directory(root: Path) -> None:
    """Make the published Dataset tree read-only before atomic publication."""
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def decode_archive(archive: Path, destination: Path) -> dict[str, Any]:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("zstandard package is required (install the Relay uv environment)") from error
    if archive.is_symlink():
        raise RuntimeError(f"archive cannot be a symlink: {archive}")
    archive = archive.resolve(strict=True)
    if not archive.is_file() or not archive.name.endswith(".jsonl.zst"):
        raise RuntimeError(f"input must be a regular .jsonl.zst archive: {archive}")
    sidecar = archive.with_name(archive.name + ".meta.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise RuntimeError(f"archive metadata must be a regular file: {sidecar}")
    try:
        metadata = json.loads(sidecar.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"archive metadata is invalid: {sidecar}") from error
    archive_hash = sha256_file(archive)
    if (not isinstance(metadata, dict) or metadata.get("schema") != "td_market_data_archive_metadata_v1"
            or metadata.get("compressed_sha256") != archive_hash
            or metadata.get("compressed_size_bytes") != archive.stat().st_size):
        raise RuntimeError(f"archive metadata/hash mismatch: {archive}")
    capture_id = metadata.get("capture_id")
    if not isinstance(capture_id, str) or len(capture_id) != 32:
        raise RuntimeError(f"archive capture ID is invalid: {sidecar}")
    filename_id = archive.name.rsplit("-", 1)[-1].removesuffix(".jsonl.zst")
    if filename_id != capture_id:
        raise RuntimeError(f"archive filename/capture ID mismatch: {archive}")
    recovery = metadata.get("recovery_provenance")
    if recovery is not None and (not isinstance(recovery, dict)
                                 or recovery.get("schema") != RECOVERY_SCHEMA
                                 or recovery.get("version") != 1
                                 or recovery.get("capture_id") != capture_id
                                 or recovery.get("origin") != "vps_recovered"
                                 or not isinstance(recovery.get("recovered_at_unix_ns"), int)
                                 or isinstance(recovery.get("recovered_at_unix_ns"), bool)
                                 or recovery["recovered_at_unix_ns"] <= 0
                                 or not isinstance(recovery.get("relay_host"), str)):
        raise RuntimeError(f"archive recovery provenance is invalid: {sidecar}")

    digest = hashlib.sha256()
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    record_types: dict[str, int] = {}
    line_count = total = 0
    pending = b""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as compressed, destination.open("xb") as output:
        with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
            while block := reader.read(1024 * 1024):
                total += len(block)
                if total > MAX_DECODED_BYTES:
                    raise RuntimeError(f"decoded capture exceeds 256 MiB: {archive}")
                digest.update(block)
                output.write(block)
                pending += block
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    line_count += 1
                    if len(raw) > MAX_LINE_BYTES:
                        raise RuntimeError(f"oversized capture record {line_count}: {archive}")
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise RuntimeError(f"invalid capture JSON line {line_count}: {archive}") from error
                    if (not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record"
                            or row.get("version") != 1):
                        raise RuntimeError(f"unsupported capture record {line_count}: {archive}")
                    first = first or row
                    last = row
                    kind = row.get("record_type")
                    record_types[kind] = record_types.get(kind, 0) + 1
        output.flush()
        os.fsync(output.fileno())
    if pending:
        raise RuntimeError(f"capture is not newline terminated: {archive}")
    raw_hash = digest.hexdigest()
    if raw_hash != metadata.get("raw_sha256") or total != metadata.get("raw_size_bytes"):
        raise RuntimeError(f"decoded capture hash/size mismatch: {archive}")
    if (first is None or first.get("record_type") != "capture_started"
            or first.get("capture_id") not in (None, capture_id)
            or last is None or last.get("record_type") != "capture_stopped"):
        raise RuntimeError(f"capture is not cleanly closed or ID mismatches: {archive}")
    started_ns, stopped_ns = first.get("recorded_at_unix_ns"), last.get("recorded_at_unix_ns")
    if (not isinstance(started_ns, int) or isinstance(started_ns, bool)
            or not isinstance(stopped_ns, int) or isinstance(stopped_ns, bool)
            or started_ns <= 0 or stopped_ns <= started_ns):
        raise RuntimeError(f"capture lifecycle times invalid: {archive}")
    if any(record_types.get(kind, 0) for kind in ("gap_detected", "clock_discontinuity", "funding_api_error")):
        raise RuntimeError(f"capture contains integrity-error records: {archive}")
    if recovery is not None:
        if recovery.get("raw_sha256") != raw_hash:
            raise RuntimeError(f"recovery provenance raw hash mismatch: {archive}")
        recovery_sidecar = destination.with_name(destination.name + ".recovery.json")
        recovery_sidecar.write_bytes(canonical(recovery) + b"\n")
    return {"stream": first.get("stream"), "symbol": first.get("symbol"),
            "capture_id": capture_id, "archive_path": str(archive),
            "compressed_sha256": archive_hash, "raw_path": str(destination),
            "raw_sha256": raw_hash, "raw_size_bytes": total,
            "source_revision": first.get("source_revision"),
            "source_host_id": first.get("source_host_id"),
            "origin": "vps_recovered" if recovery else "local",
            "recovered_at_unix_ns": recovery.get("recovered_at_unix_ns") if recovery else None,
            "recovery_provenance": recovery,
            "capture_started_at_ns": started_ns, "capture_stopped_at_ns": stopped_ns,
            "record_count": line_count, "record_types": record_types}


def run_export(script: Path, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(script), *arguments], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, metavar="STREAM=PATH")
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-output", type=Path,
                        help="stable decoded sources; default is OUTPUT-sources beside output")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()
    start_ns, end_ns = ns(args.start), ns(args.end)
    if start_ns >= end_ns or not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        raise SystemExit("supply an absolute, increasing window and a not-yet-existing output")
    script_dir = Path(__file__).resolve().parent
    if args.output.resolve().parent == script_dir or script_dir in args.output.resolve().parents:
        raise SystemExit("Dataset output must be outside the source repository")
    captures: dict[str, list[Path]] = {stream: [] for stream in STREAMS}
    for value in args.capture:
        if "=" not in value:
            raise SystemExit(f"invalid --capture value: {value}")
        stream, raw_path = value.split("=", 1)
        if stream not in STREAMS:
            raise SystemExit(f"unsupported USD-M research stream: {stream}")
        captures[stream].append(Path(raw_path))
    if any(not captures[stream] for stream in REQUIRED):
        raise SystemExit("at least one capture is required for each required USD-M stream")
    resolved = [path.resolve(strict=True) for stream in STREAMS for path in captures[stream]]
    if len(resolved) != len(set(resolved)):
        raise SystemExit("duplicate capture paths are not allowed")

    output = args.output.resolve()
    reject_system_temp(output, "Dataset output")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_output = (args.source_output or output.with_name(output.name + "-sources")).resolve()
    reject_system_temp(source_output, "decoded source output")
    if (source_output == output or source_output.is_symlink() or
            (source_output.exists() and not source_output.is_dir())
            or source_output == script_dir or script_dir in source_output.parents):
        raise SystemExit("source output must be a regular directory outside the repository and Dataset output")
    source_output.parent.mkdir(parents=True, exist_ok=True)
    source_stage = Path(tempfile.mkdtemp(prefix=f".{source_output.name}.staging-", dir=source_output.parent))
    output_stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        sources: dict[str, list[dict[str, Any]]] = {stream: [] for stream in STREAMS}
        raw_paths: dict[str, list[Path]] = {stream: [] for stream in STREAMS}
        for stream in STREAMS:
            for archive in captures[stream]:
                destination = source_stage / stream / (archive.name.removesuffix(".zst"))
                record = decode_archive(archive, destination)
                if record["stream"] != stream or record["symbol"] != args.symbol.upper():
                    raise RuntimeError(f"capture stream/symbol mismatch: {archive}")
                sources[stream].append(record)
            sources[stream].sort(key=lambda row: (row["capture_started_at_ns"], row["capture_id"]))
            raw_paths[stream] = [source_stage / stream / (Path(row["raw_path"]).name)
                                 for row in sources[stream]]
        # Child manifests bind absolute paths; reuse only byte-identical decoded sources.
        if source_output.exists():
            for stream in STREAMS:
                for source in sources[stream]:
                    staged = source_stage / stream / Path(source["raw_path"]).name
                    final = source_output / stream / staged.name
                    if final.exists() or final.is_symlink():
                        if final.is_symlink() or not final.is_file() or sha256_file(final) != source["raw_sha256"]:
                            raise RuntimeError(f"existing decoded source differs from archive: {final}")
                    else:
                        final.parent.mkdir(parents=True, exist_ok=True)
                        os.link(staged, final)
                    staged_recovery = staged.with_name(staged.name + ".recovery.json")
                    final_recovery = final.with_name(final.name + ".recovery.json")
                    if staged_recovery.exists():
                        if final_recovery.exists() or final_recovery.is_symlink():
                            if final_recovery.is_symlink() or final_recovery.read_bytes() != staged_recovery.read_bytes():
                                raise RuntimeError(f"existing recovery provenance differs: {final_recovery}")
                        else:
                            os.link(staged_recovery, final_recovery)
                    elif final_recovery.exists() or final_recovery.is_symlink():
                        raise RuntimeError(f"existing decoded source has unexpected recovery provenance: {final_recovery}")
            shutil.rmtree(source_stage)
        else:
            publish_directory(source_stage, source_output)
        for stream in STREAMS:
            for row in sources[stream]:
                row["raw_path"] = str((source_output / stream / Path(row["raw_path"]).name).resolve())
            raw_paths[stream] = [source_output / path.name for path in raw_paths[stream]]

        def window_args() -> list[str]:
            return ["--symbol", args.symbol.upper(), "--start", fmt_ns(start_ns), "--end", fmt_ns(end_ns)]

        datasets: dict[str, dict[str, Any]] = {}
        agg_out = output_stage / "usdm-aggtrades"
        run_export(script_dir / "export_usdm_aggtrades_dataset.py",
                   [item for path in raw_paths["usdm-aggtrades"] for item in ("--capture", str(path))]
                   + ["--output", str(agg_out), *window_args()])
        run_export(script_dir / "verify_usdm_aggtrades_dataset.py", ["--dataset", str(agg_out)])
        datasets["usdm-aggtrades"] = {"path": "usdm-aggtrades", "manifest_sha256": sha256_file(agg_out / "manifest.json")}

        mark_evidence = []
        for index, path in enumerate(raw_paths["usdm-funding-mark"], 1):
            evidence = output_stage / f"usdm-funding-mark-evidence-{index:03}.jsonl"
            run_export(script_dir / "export_funding_capture.py", ["--capture", str(path), "--output", str(evidence)])
            mark_evidence.append(evidence)
        datasets["usdm-funding-mark"] = {"evidence": [
            {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in mark_evidence]}

        funding_out = output_stage / "usdm-finalized-funding"
        funding_args = [item for path in raw_paths["usdm-funding-mark"] for item in ("--mark-capture", str(path))]
        funding_args += [item for path in raw_paths["usdm-finalized-funding"]
                         for item in ("--funding-capture", str(path))]
        run_export(script_dir / "export_usdm_funding_dataset.py",
                   [*funding_args, "--output", str(funding_out), *window_args()])
        run_export(script_dir / "verify_usdm_funding_dataset.py", ["--dataset", str(funding_out)])
        datasets["usdm-finalized-funding"] = {
            "path": "usdm-finalized-funding", "manifest_sha256": sha256_file(funding_out / "manifest.json")}

        if raw_paths["usdm-klines-1m"]:
            kline_out = output_stage / "usdm-klines-1m"
            run_export(script_dir / "export_usdm_klines_dataset.py",
                       [item for path in raw_paths["usdm-klines-1m"] for item in ("--capture", str(path))]
                       + ["--market", "USD_M", "--output", str(kline_out), *window_args()])
            run_export(script_dir / "verify_usdm_klines_dataset.py", ["--dataset", str(kline_out)])
            datasets["usdm-klines-1m"] = {
                "path": "usdm-klines-1m", "manifest_sha256": sha256_file(kline_out / "manifest.json")}

        source_inventory = [{"stream": stream, **row} for stream in STREAMS for row in sources[stream]]
        source_hashes = [row["compressed_sha256"] for row in source_inventory]
        revision = None
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=script_dir,
                                               text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        manifest = {"schema": SCHEMA, "version": 1, "exchange": "BINANCE", "market": "USD_M",
                    "symbol": args.symbol.upper(),
                    "window": {"start_ns": start_ns, "end_ns": end_ns,
                               "start": fmt_ns(start_ns), "end": fmt_ns(end_ns), "closed": "[start,end)"},
                    "required_streams": list(REQUIRED),
                    "optional_streams": {"usdm-klines-1m": bool(raw_paths["usdm-klines-1m"])},
                    "sources": source_inventory,
                    "source_capture_set_sha256": hashlib.sha256(canonical(source_hashes)).hexdigest(),
                    "datasets": datasets,
                    "availability_policy": {"local": "available_at_ns=max(received_at_ns,exchange_event_time_ns)",
                                            "vps_recovered": "available_at_ns=max(received_at_ns,recovered_at_unix_ns,exchange_event_time_ns)"},
                    "gap_policy": "reject; no interpolation and no silent window shortening",
                    "builder_sha256": sha256_file(Path(__file__).resolve()),
                    "git_revision": revision}
        manifest_path = output_stage / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        seal_directory(output_stage)
        publish_directory(output_stage, output)
        print(f"USD-M RL Dataset published: {output}")
        print(f"source segments: {len(source_inventory)}; optional klines: {bool(raw_paths['usdm-klines-1m'])}")
    except BaseException:
        shutil.rmtree(output_stage, ignore_errors=True)
        shutil.rmtree(source_stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
