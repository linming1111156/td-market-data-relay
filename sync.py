#!/usr/bin/env python3
"""Restore retained VPS captures into local storage with checksum verification."""

from __future__ import annotations

import argparse
import calendar
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from collector import SUPPORTED_STREAMS as COLLECTOR_SUPPORTED_STREAMS

SUPPORTED_STREAMS = COLLECTOR_SUPPORTED_STREAMS
RELAY_RETENTION_NS = 73 * 3600 * 1_000_000_000
LOSS_SCHEMA = "td_market_data_relay_early_discard_v1"
MAX_LOSS_LEDGER_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_LINE_BYTES = 2 * 1024 * 1024
RECOVERY_SCHEMA = "td_market_data_relay_recovery_provenance_v1"


def continuous_scan_start(now_ns: int, since_ns: int) -> int:
    """Reconcile every remotely retained capture on each successful poll."""
    return max(since_ns, now_ns - RELAY_RETENTION_NS, 0)


def request(url: str, api_key: str, timeout: int = 60):
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}",
                 "User-Agent": "td-market-data-sync/1.0"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture_is_closed(path: Path) -> bool:
    """Do not publish a still-growing capture as a permanent local copy."""
    last_record = None
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if len(raw) > MAX_CAPTURE_LINE_BYTES:
                raise RuntimeError(f"capture line {line_number} exceeds size bound: {path}")
            if not raw.endswith(b"\n"):
                return False
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"capture line {line_number} is invalid JSON: {path}") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"capture line {line_number} is not an object: {path}")
            last_record = record
    return isinstance(last_record, dict) and last_record.get("record_type") == "capture_stopped"


def validate_capture_reader(reader, expected_sha256: str) -> int:
    """Hash and validate a capture as a bounded stream without writing decoded bytes."""
    digest_value = hashlib.sha256()
    total = 0
    line_number = 0
    last_record = None
    pending = bytearray()
    while True:
        block = reader.read(64 * 1024)
        if not block:
            break
        digest_value.update(block)
        total += len(block)
        if total > 256 * 1024 * 1024:
            raise RuntimeError("decompressed capture exceeds the 256 MiB cap")
        pending.extend(block)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                if len(pending) > MAX_CAPTURE_LINE_BYTES:
                    raise RuntimeError(f"capture line {line_number + 1} exceeds size bound")
                break
            line_number += 1
            if newline + 1 > MAX_CAPTURE_LINE_BYTES:
                raise RuntimeError(f"capture line {line_number} exceeds size bound")
            raw = bytes(pending[:newline + 1])
            del pending[:newline + 1]
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError(f"capture line {line_number} is invalid JSON") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"capture line {line_number} is not an object")
            last_record = record
    if pending:
        raise RuntimeError(f"capture line {line_number + 1} is truncated")
    if digest_value.hexdigest() != expected_sha256:
        raise RuntimeError("decompressed capture hash mismatch")
    if not isinstance(last_record, dict) or last_record.get("record_type") != "capture_stopped":
        raise RuntimeError("remote capture is not closed")
    return total


def capture_time_from_name(filename: str) -> int:
    match = re.fullmatch(r"[A-Z0-9]{1,32}-(\d{8}T\d{6}Z)-[0-9a-f]{32}\.jsonl(?:\.zst)?", filename)
    if match is None:
        raise RuntimeError("capture filename has no valid UTC timestamp")
    parsed = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return calendar.timegm(parsed.utctimetuple())


def archive_metadata(target: Path, entry: dict, raw_size: int, relay_host: str,
                     recovered_at_ns: int, filename: str | None = None) -> dict:
    metadata = {
        "schema": "td_market_data_archive_metadata_v1",
        "capture_id": str(entry["id"]),
        "raw_sha256": str(entry["raw_sha256"]),
        "raw_size_bytes": raw_size,
        "compressed_sha256": digest(target),
        "compressed_size_bytes": target.stat().st_size,
        "captured_at_unix_seconds": capture_time_from_name(filename or target.name),
        "recovery_provenance": {
            "schema": RECOVERY_SCHEMA,
            "version": 1,
            "capture_id": str(entry["id"]),
            "origin": "vps_recovered",
            "recovered_at_unix_ns": recovered_at_ns,
            "raw_sha256": str(entry["raw_sha256"]),
            "relay_host": relay_host,
        },
    }
    for key in ("capture_started_at_ns", "capture_stopped_at_ns", "gap_count",
                "disconnect_count", "stop_reason", "closed", "trainable"):
        if key in entry:
            metadata[key] = entry[key]
    return metadata


def write_archive_metadata(target: Path, metadata: dict) -> None:
    metadata_path = target.with_name(target.name + ".meta.json")
    temporary = metadata_path.with_name(metadata_path.name + f".tmp-{os.getpid()}")
    remove_abandoned_temp(temporary)
    try:
        with temporary.open("xb") as stream:
            stream.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_without_overwrite(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def compress_capture(source: Path, destination: Path) -> tuple[str, int]:
    """Compress a validated raw capture without materializing another raw copy."""
    import zstandard

    raw_hash = hashlib.sha256()
    raw_size = 0
    with source.open("rb") as src, destination.open("xb") as output:
        with zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(
                output, closefd=False) as writer:
            while block := src.read(1024 * 1024):
                raw_hash.update(block)
                raw_size += len(block)
                writer.write(block)
        output.flush()
        os.fsync(output.fileno())
    return raw_hash.hexdigest(), raw_size


def verify_compressed_capture(path: Path, entry: dict) -> int:
    import zstandard

    try:
        with path.open("rb") as source:
            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                return validate_capture_reader(reader, str(entry["raw_sha256"]))
    except zstandard.ZstdError as error:
        raise RuntimeError(f"invalid zstd archive for capture {entry['id']}") from error


def restore_to_archive(base: str, key: str, entry: dict, target: Path, relay_host: str) -> bool:
    """Restore one verified remote capture directly into the local zstd archive."""
    metadata_path = target.with_name(target.name + ".meta.json")
    if target.exists() or target.is_symlink():
        if metadata_path.exists() or metadata_path.is_symlink():
            if archived_capture_matches(target, entry):
                return False
            raise RuntimeError(f"local archive metadata differs; left untouched: {metadata_path}")
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"local archive is unsafe; left untouched: {target}")
        # Recover safely from a crash after publishing the archive but before metadata.
        raw_size = verify_compressed_capture(target, entry)
        metadata = archive_metadata(target, entry, raw_size, relay_host, time.time_ns())
        write_archive_metadata(target, metadata)
        return False
    if metadata_path.exists() or metadata_path.is_symlink():
        raise RuntimeError(f"orphaned local archive metadata; left untouched: {metadata_path}")

    target.parent.mkdir(parents=True, exist_ok=True)
    download = target.with_name(target.name + f".download-{os.getpid()}")
    compressed = target.with_name(target.name + f".tmp-{os.getpid()}")
    remove_abandoned_temp(download)
    remove_abandoned_temp(compressed)
    route = "/v1/captures/" + urllib.parse.quote(str(entry["id"]), safe="")
    compressed_hash = hashlib.sha256()
    try:
        with request(base + route, key, timeout=180) as response, download.open("xb") as output:
            declared_length = int(response.headers.get("Content-Length", "0"))
            if declared_length != entry["size_bytes"]:
                raise RuntimeError(f"relay size changed for capture {entry['id']}")
            while block := response.read(1024 * 1024):
                compressed_hash.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if compressed_hash.hexdigest() != entry["sha256"]:
            raise RuntimeError(f"download hash mismatch for capture {entry['id']}")

        if entry["compressed"]:
            raw_size = verify_compressed_capture(download, entry)
            os.replace(download, compressed)
        else:
            with download.open("rb") as source:
                raw_size = validate_capture_reader(source, str(entry["raw_sha256"]))
            raw_hash, compressed_raw_size = compress_capture(download, compressed)
            if raw_hash != entry["raw_sha256"] or compressed_raw_size != raw_size:
                raise RuntimeError(f"capture changed while archiving {entry['id']}")

        os.utime(compressed, (capture_time_from_name(target.name),) * 2)
        metadata = archive_metadata(compressed, entry, raw_size, relay_host, time.time_ns(),
                                    filename=target.name)
        publish_without_overwrite(compressed, target)
        write_archive_metadata(target, metadata)
        if not archived_capture_matches(target, entry):
            raise RuntimeError(f"local archive verification failed after publish: {target}")
        return True
    finally:
        download.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)


def archived_capture_matches(path: Path, entry: dict) -> bool:
    """Check whether the local relay archive already holds this raw capture."""
    metadata_path = path.with_name(path.name + ".meta.json")
    if not path.exists() and not path.is_symlink():
        if metadata_path.exists() or metadata_path.is_symlink():
            raise RuntimeError(f"orphaned local archive metadata; left untouched: {metadata_path}")
        return False
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"local archive is not a regular file; left untouched: {path}")
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise RuntimeError(f"local archive metadata is missing or unsafe; left untouched: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"local archive metadata is invalid; left untouched: {metadata_path}") from error
    expected_id = str(entry["id"])
    if (metadata.get("schema") != "td_market_data_archive_metadata_v1"
            or metadata.get("capture_id") != expected_id
            or metadata.get("compressed_size_bytes") != path.stat().st_size
            or re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("compressed_sha256", ""))) is None
            or metadata.get("compressed_sha256") != digest(path)
            or metadata.get("raw_sha256") != entry["raw_sha256"]):
        raise RuntimeError(f"local archive differs from relay copy; left untouched: {path}")
    return True


def remove_abandoned_temp(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    match = re.search(r"\.(?:download|decoded|tmp)-(\d+)$", path.name)
    if match is None or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unexpected sync temporary path; left untouched: {path}")
    owner_pid = int(match.group(1))
    if owner_pid == os.getpid():
        raise RuntimeError(f"sync temporary file is already owned by this process: {path}")
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        path.unlink()
        return
    except PermissionError as error:
        raise RuntimeError(f"cannot verify temporary file owner {owner_pid}; left untouched: {path}") from error
    raise RuntimeError(f"temporary file owner {owner_pid} is still running; left untouched: {path}")


def record_remote_losses(output: Path, losses: list[dict]) -> int:
    validated = []
    for row in losses:
        if not isinstance(row, dict) or row.get("schema") != LOSS_SCHEMA or row.get("version") != 1:
            raise RuntimeError("relay returned an invalid capture-loss record")
        capture_id, stream_name, symbol = row.get("capture_id"), row.get("stream"), row.get("symbol")
        relative_parts = Path(row.get("path", "")).parts
        discarded_ns = row.get("discarded_at_ns")
        if (not isinstance(capture_id, str) or re.fullmatch(r"[0-9a-f]{32}", capture_id) is None
                or stream_name not in SUPPORTED_STREAMS
                or not isinstance(symbol, str) or re.fullmatch(r"[A-Z0-9]{1,32}", symbol) is None
                or len(relative_parts) != 3 or relative_parts[:2] != (stream_name, symbol)
                or ".." in relative_parts
                or relative_parts[2].split(".")[0].rsplit("-", 1)[-1] != capture_id
                or row.get("reason") not in {"archive_quota", "hot_storage_quota"}
                or not isinstance(discarded_ns, int) or isinstance(discarded_ns, bool) or discarded_ns < 0
                or (row.get("raw_sha256") is not None
                    and re.fullmatch(r"[0-9a-f]{64}", str(row.get("raw_sha256"))) is None)):
            raise RuntimeError("relay returned a malformed capture-loss record")
        validated.append(row)
    ledger = output / ".relay-losses.jsonl"
    if ledger.is_symlink():
        raise RuntimeError("local relay-loss evidence path is a symlink")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(ledger, flags, 0o600)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_LOSS_LEDGER_BYTES:
            raise RuntimeError("local relay-loss evidence is not a bounded regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        raw = os.pread(fd, size + 1, 0)
        if len(raw) != size:
            raise RuntimeError("could not read complete local relay-loss evidence")
        seen = set()
        if raw:
            for line_number, line in enumerate(raw.splitlines(), 1):
                try:
                    existing = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"local relay-loss evidence line {line_number} is invalid") from error
                if not isinstance(existing, dict) or existing.get("schema") != LOSS_SCHEMA:
                    raise RuntimeError(f"local relay-loss evidence line {line_number} has invalid schema")
                seen.add((existing.get("capture_id"), existing.get("discarded_at_ns"), existing.get("reason")))
        additions = [row for row in validated
                     if (row["capture_id"], row["discarded_at_ns"], row["reason"]) not in seen]
        encoded = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                           for row in additions)
        if size + len(encoded) > MAX_LOSS_LEDGER_BYTES:
            raise RuntimeError("local relay-loss evidence reached its 16 MiB safety bound")
        os.lseek(fd, 0, os.SEEK_END)
        pending = memoryview(encoded)
        while pending:
            written = os.write(fd, pending)
            if written <= 0:
                raise OSError("short write to local relay-loss evidence")
            pending = pending[written:]
        os.fsync(fd)
        parent_fd = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return len(additions)
    finally:
        os.close(fd)


def publish_without_overwrite(source: Path, target: Path) -> None:
    os.link(source, target)
    source.unlink()
    directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_recovery_provenance(target: Path, entry: dict, relay_host: str) -> None:
    """Bind a recovered raw capture to its local recovery time and source hash."""
    recovery_path = target.with_name(target.name + ".recovery.json")
    expected = {
        "schema": RECOVERY_SCHEMA,
        "version": 1,
        "capture_id": str(entry["id"]),
        "origin": "vps_recovered",
        "recovered_at_unix_ns": time.time_ns(),
        "raw_sha256": str(entry["raw_sha256"]),
        "relay_host": relay_host,
    }
    if recovery_path.exists() or recovery_path.is_symlink():
        if recovery_path.is_symlink() or not recovery_path.is_file():
            raise RuntimeError(f"recovery provenance is not a regular file: {recovery_path}")
        try:
            prior = json.loads(recovery_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"recovery provenance is invalid: {recovery_path}") from error
        if (prior.get("schema") != RECOVERY_SCHEMA or prior.get("version") != 1
                or prior.get("capture_id") != expected["capture_id"]
                or prior.get("origin") != "vps_recovered"
                or prior.get("raw_sha256") != expected["raw_sha256"]
                or not isinstance(prior.get("recovered_at_unix_ns"), int)
                or isinstance(prior.get("recovered_at_unix_ns"), bool)
                or prior["recovered_at_unix_ns"] <= 0
                or not isinstance(prior.get("relay_host"), str)):
            raise RuntimeError(f"recovery provenance does not match remote capture: {recovery_path}")
        return
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    temp = recovery_path.with_name(recovery_path.name + f".tmp-{os.getpid()}")
    remove_abandoned_temp(temp)
    try:
        with temp.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, recovery_path)
        directory = os.open(recovery_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("TD_RELAY_BASE_URL"),
                        help="Cloudflare Tunnel URL for the relay (or TD_RELAY_BASE_URL)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive-root", type=Path,
                        help="local relay archive used to skip existing captures or receive direct archives")
    parser.add_argument("--direct-archive", action="store_true",
                        help="verify remote captures and store them as local zstd archives without hot JSONL files")
    parser.add_argument("--since-ns", type=int, default=0, help="Only captures modified since this Unix nanosecond time")
    parser.add_argument("--stream", default="", help="Optional stream filter")
    parser.add_argument("--symbol", default="", help="Optional symbol filter")
    parser.add_argument("--api-key-env", default="TD_RELAY_API_KEY")
    parser.add_argument("--continuous", action="store_true", help="poll and restore continuously")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--state-file", type=Path, help="atomic cursor file required in continuous mode")
    args = parser.parse_args()
    if not args.base_url:
        raise SystemExit("--base-url or TD_RELAY_BASE_URL is required")
    key = os.environ.get(args.api_key_env, "")
    if len(key) < 32:
        raise SystemExit(f"environment variable {args.api_key_env} must contain the relay API key")
    base = args.base_url.rstrip("/")
    parsed_base = urllib.parse.urlsplit(base)
    local_http = parsed_base.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed_base.scheme != "https" and not (parsed_base.scheme == "http" and local_http):
        raise SystemExit("relay base URL must use HTTPS (HTTP is allowed only for localhost)")
    relay_host = parsed_base.hostname or "unknown"
    if args.continuous and (args.state_file is None or args.poll_seconds < 30):
        raise SystemExit("continuous mode requires --state-file and --poll-seconds >= 30")
    if args.archive_root is not None and (args.archive_root.is_symlink() or not args.archive_root.is_dir()):
        raise SystemExit("--archive-root must be an existing local relay archive directory")
    if args.direct_archive and args.archive_root is None:
        raise SystemExit("--direct-archive requires --archive-root")
    args.output.mkdir(parents=True, exist_ok=True)

    def restore_since(since_ns: int) -> None:
        query = urllib.parse.urlencode({"since_ns": since_ns, "stream": args.stream, "symbol": args.symbol})
        try:
            with request(f"{base}/v1/captures?{query}", key) as response:
                index = json.load(response)
        except (urllib.error.URLError, json.JSONDecodeError) as error:
            raise RuntimeError(f"could not fetch relay index: {error}") from error
        if index.get("schema") != "td_market_data_relay_index_v1" or not isinstance(index.get("captures"), list):
            raise RuntimeError("relay returned an unsupported index schema")
        losses = index.get("losses", [])
        if not isinstance(losses, list):
            raise RuntimeError("relay returned an invalid capture-loss inventory")
        new_losses = record_remote_losses(args.output, losses)
        if losses:
            print(f"WARNING: relay reports early-discarded captures={len(losses)}; "
                  f"newly recorded={new_losses} in {args.output / '.relay-losses.jsonl'}", flush=True)
        restored = skipped = 0
        for entry in index["captures"]:
            stream_name = entry.get("stream")
            symbol = entry.get("symbol")
            relative_parts = Path(entry.get("path", "")).parts
            filename = relative_parts[-1] if len(relative_parts) == 3 else ""
            if (stream_name not in SUPPORTED_STREAMS
                    or not isinstance(symbol, str) or re.fullmatch(r"[A-Z0-9]{1,32}", symbol) is None
                    or relative_parts[:2] != (stream_name, symbol)
                    or re.fullmatch(rf"{re.escape(symbol)}-[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{32}}(?:\.jsonl(?:\.zst)?)?", filename) is None
                    or not isinstance(entry.get("compressed"), bool)
                    or re.fullmatch(r"[0-9a-f]{32}", str(entry.get("id", ""))) is None
                    or not isinstance(entry.get("sha256"), str)
                    or not isinstance(entry.get("raw_sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
                    or re.fullmatch(r"[0-9a-f]{64}", entry["raw_sha256"]) is None
                    or not isinstance(entry.get("size_bytes"), int)
                    or entry["size_bytes"] < 0 or entry["size_bytes"] > 1024**3):
                raise RuntimeError("relay returned an invalid or unsafe capture manifest")
            if bool(entry["compressed"]) != filename.endswith(".zst"):
                raise RuntimeError("relay compression flag disagrees with capture path")
            raw_filename = filename.removesuffix(".zst")
            if args.direct_archive:
                archive_target = (args.archive_root / stream_name / symbol
                                  / f"{raw_filename}.zst")
                if restore_to_archive(base, key, entry, archive_target, relay_host):
                    restored += 1
                    print(f"archived {stream_name}/{symbol}/{archive_target.name}", flush=True)
                else:
                    skipped += 1
                continue
            target = args.output / stream_name / symbol / raw_filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) == entry["raw_sha256"]:
                    if capture_is_closed(target):
                        write_recovery_provenance(target, entry, relay_host)
                        skipped += 1
                    else:
                        skipped += 1
                        print(f"deferred incomplete remote capture: {stream_name}/{symbol}/{filename}", flush=True)
                    continue
                raise RuntimeError(f"local capture differs from relay copy; left untouched: {target}")
            if args.archive_root is not None:
                archive_path = args.archive_root / stream_name / symbol / f"{raw_filename}.zst"
                if archived_capture_matches(archive_path, entry):
                    skipped += 1
                    continue
            temp = target.with_name(target.name + f".download-{os.getpid()}")
            remove_abandoned_temp(temp)
            route = "/v1/captures/" + urllib.parse.quote(str(entry["id"]), safe="")
            try:
                with request(base + route, key, timeout=180) as response, temp.open("xb") as output:
                    declared_length = int(response.headers.get("Content-Length", "0"))
                    if declared_length != entry["size_bytes"]:
                        raise RuntimeError(f"relay size changed for capture {entry['id']}")
                    h = hashlib.sha256()
                    while block := response.read(1024 * 1024):
                        h.update(block)
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
                    if h.hexdigest() != entry["sha256"]:
                        raise RuntimeError(f"download hash mismatch for capture {entry['id']}")
                if entry["compressed"]:
                    import zstandard

                    decoded = target.with_name(target.name + f".decoded-{os.getpid()}")
                    remove_abandoned_temp(decoded)
                    uncompressed_size = 0
                    try:
                        with temp.open("rb") as source, decoded.open("xb") as output:
                            with zstandard.ZstdDecompressor().stream_reader(source) as reader:
                                while block := reader.read(1024 * 1024):
                                    uncompressed_size += len(block)
                                    if uncompressed_size > 256 * 1024 * 1024:
                                        raise RuntimeError("decompressed capture exceeds the 256 MiB cap")
                                    output.write(block)
                            output.flush()
                            os.fsync(output.fileno())
                        if digest(decoded) != entry["raw_sha256"]:
                            raise RuntimeError(f"decompressed capture hash mismatch for {entry['id']}")
                        if not capture_is_closed(decoded):
                            skipped += 1
                            print(f"deferred incomplete remote capture: {stream_name}/{symbol}/{filename}", flush=True)
                            continue
                        publish_without_overwrite(decoded, target)
                    finally:
                        decoded.unlink(missing_ok=True)
                else:
                    if digest(temp) != entry["raw_sha256"]:
                        raise RuntimeError(f"raw capture hash mismatch for {entry['id']}")
                    if not capture_is_closed(temp):
                        skipped += 1
                        print(f"deferred incomplete remote capture: {stream_name}/{symbol}/{filename}", flush=True)
                        continue
                    publish_without_overwrite(temp, target)
                write_recovery_provenance(target, entry, relay_host)
                restored += 1
                print(f"restored {stream_name}/{symbol}/{target.name}", flush=True)
            finally:
                temp.unlink(missing_ok=True)
        print(f"sync complete: restored={restored} already_present={skipped}", flush=True)

    state_path = args.state_file
    cursor = args.since_ns
    if args.continuous and state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            if state.get("schema") != "td_market_data_sync_cursor_v1":
                raise ValueError("invalid cursor schema")
            cursor = max(cursor, int(state["cursor_ns"]))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid sync cursor: {error}") from error
    while True:
        scan_from = cursor
        if args.continuous:
            scan_from = continuous_scan_start(time.time_ns(), args.since_ns)
        try:
            restore_since(scan_from)
            if not args.continuous:
                break
            cursor = time.time_ns()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_state = state_path.with_name(state_path.name + f".tmp-{os.getpid()}")
            remove_abandoned_temp(temp_state)
            try:
                with temp_state.open("x") as stream:
                    stream.write(json.dumps({"schema": "td_market_data_sync_cursor_v1", "cursor_ns": cursor}) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temp_state, state_path)
            finally:
                temp_state.unlink(missing_ok=True)
            time.sleep(args.poll_seconds)
        except (OSError, RuntimeError, urllib.error.URLError) as error:
            if not args.continuous:
                raise SystemExit(str(error)) from error
            print(f"sync attempt failed; cursor kept for retry: {error}", flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
