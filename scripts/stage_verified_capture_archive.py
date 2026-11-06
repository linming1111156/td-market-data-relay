#!/usr/bin/env python3
"""Return a raw capture path, restoring and verifying its local zstd archive if needed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import zstandard
from capture_recovery import SCHEMA as RECOVERY_SCHEMA, recovery_provenance


def digest(path: Path) -> tuple[str, int]:
    value = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
            size += len(block)
    return value.hexdigest(), size


def validate_recovery_sidecar(capture: Path, metadata: dict, raw_sha256: str) -> dict | None:
    value = metadata.get("recovery_provenance")
    if value is None:
        return None
    capture_id = capture.name.rsplit("-", 1)[-1].removesuffix(".jsonl")
    recovered = value.get("recovered_at_unix_ns") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema") != RECOVERY_SCHEMA
            or value.get("version") != 1 or value.get("capture_id") != capture_id
            or value.get("origin") != "vps_recovered" or value.get("raw_sha256") != raw_sha256
            or not isinstance(recovered, int) or isinstance(recovered, bool) or recovered <= 0
            or not isinstance(value.get("relay_host"), str) or not value["relay_host"]
            or len(value["relay_host"]) > 255):
        raise ValueError("archive recovery provenance does not match the staged capture")
    return value


def publish_recovery_sidecar(capture: Path, metadata: dict, raw_sha256: str) -> None:
    value = validate_recovery_sidecar(capture, metadata, raw_sha256)
    if value is None:
        return
    capture_id = capture.name.rsplit("-", 1)[-1].removesuffix(".jsonl")
    sidecar = capture.with_name(capture.name + ".recovery.json")
    if sidecar.exists() or sidecar.is_symlink():
        existing = recovery_provenance(capture, raw_sha256, capture_id)
        if existing != value:
            raise ValueError(f"staged recovery provenance differs from archive metadata: {sidecar}")
        return
    temporary = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError(f"staged recovery sidecar temporary already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, sidecar)
        directory = os.open(sidecar.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    recovery_provenance(capture, raw_sha256, capture_id)


def stage_capture(capture: Path, hot_root: Path, archive_root: Path,
                  staging_root: Path) -> Path:
    hot_root = hot_root.resolve(strict=True)
    capture = capture.absolute()
    try:
        relative = capture.relative_to(hot_root)
    except ValueError as error:
        raise ValueError("capture must be under the hot root") from error
    if len(relative.parts) != 3 or any(part in {".", ".."} for part in relative.parts):
        raise ValueError("capture path must have stream/symbol/filename layout")
    if capture.is_symlink():
        raise ValueError("hot capture cannot be a symlink")
    if capture.is_file():
        return capture

    archive_root = archive_root.resolve(strict=True)
    archive = archive_root / Path(str(relative) + ".zst")
    metadata_path = Path(str(archive) + ".meta.json")
    if archive.is_symlink() or metadata_path.is_symlink() or not archive.is_file() or not metadata_path.is_file():
        raise ValueError(f"neither raw capture nor regular compressed archive exists: {capture}")
    metadata = json.loads(metadata_path.read_bytes())
    capture_id = capture.name.rsplit("-", 1)[-1].removesuffix(".jsonl")
    if (not isinstance(metadata, dict)
            or metadata.get("schema") != "td_market_data_archive_metadata_v1"
            or metadata.get("capture_id") != capture_id):
        raise ValueError("archive metadata identity mismatch")
    archive_sha256, archive_size = digest(archive)
    if (metadata.get("compressed_sha256") != archive_sha256
            or metadata.get("compressed_size_bytes") != archive_size):
        raise ValueError("compressed archive checksum or size mismatch")
    expected_raw_sha256 = metadata.get("raw_sha256")
    expected_raw_size = metadata.get("raw_size_bytes")
    if (not isinstance(expected_raw_sha256, str) or len(expected_raw_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_raw_sha256)
            or not isinstance(expected_raw_size, int) or isinstance(expected_raw_size, bool)
            or expected_raw_size < 0):
        raise ValueError("archive metadata raw-file identity is invalid")
    validate_recovery_sidecar(capture, metadata, expected_raw_sha256)

    staging_root.mkdir(parents=True, exist_ok=True)
    staging_root = staging_root.resolve(strict=True)
    destination = staging_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("staged capture cannot be a symlink")
    if destination.exists():
        raw_sha256, raw_size = digest(destination)
        if raw_sha256 != expected_raw_sha256 or raw_size != expected_raw_size:
            raise ValueError("existing staged capture does not match archive metadata")
        publish_recovery_sidecar(destination, metadata, expected_raw_sha256)
        return destination

    temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
    raw_hash = hashlib.sha256()
    raw_size = 0
    try:
        with archive.open("rb") as compressed, temporary.open("xb") as output:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                while block := reader.read(1024 * 1024):
                    output.write(block)
                    raw_hash.update(block)
                    raw_size += len(block)
            output.flush()
            os.fsync(output.fileno())
        if raw_hash.hexdigest() != expected_raw_sha256 or raw_size != expected_raw_size:
            raise ValueError("decompressed capture checksum or size mismatch")
        os.replace(temporary, destination)
        publish_recovery_sidecar(destination, metadata, expected_raw_sha256)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--hot-root", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--staging-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(stage_capture(args.capture, args.hot_root, args.archive_root, args.staging_root))
    except (OSError, ValueError, json.JSONDecodeError, zstandard.ZstdError) as error:
        print(f"capture staging rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
