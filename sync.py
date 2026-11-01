#!/usr/bin/env python3
"""Restore retained VPS captures into a local append-only archive."""

from __future__ import annotations

import argparse
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
from pathlib import Path

from collector import SUPPORTED_STREAMS as COLLECTOR_SUPPORTED_STREAMS

SUPPORTED_STREAMS = COLLECTOR_SUPPORTED_STREAMS
RELAY_RETENTION_NS = 73 * 3600 * 1_000_000_000
LOSS_SCHEMA = "td_market_data_relay_early_discard_v1"
MAX_LOSS_LEDGER_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_LINE_BYTES = 2 * 1024 * 1024


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("TD_RELAY_BASE_URL"),
                        help="Cloudflare Tunnel URL for the relay (or TD_RELAY_BASE_URL)")
    parser.add_argument("--output", required=True, type=Path)
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
    if args.continuous and (args.state_file is None or args.poll_seconds < 30):
        raise SystemExit("continuous mode requires --state-file and --poll-seconds >= 30")
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
            target = args.output / stream_name / symbol / filename.removesuffix(".zst")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(target) == entry["raw_sha256"]:
                    if capture_is_closed(target):
                        skipped += 1
                    else:
                        skipped += 1
                        print(f"deferred incomplete remote capture: {stream_name}/{symbol}/{filename}", flush=True)
                    continue
                raise RuntimeError(f"local capture differs from relay copy; left untouched: {target}")
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
                restored += 1
                print(f"restored {stream_name}/{symbol}/{target.name}")
            finally:
                temp.unlink(missing_ok=True)
        print(f"sync complete: restored={restored} already_present={skipped}")

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
