#!/usr/bin/env python3
"""Restore retained VPS captures into a local append-only archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from collector import SUPPORTED_STREAMS as COLLECTOR_SUPPORTED_STREAMS

SUPPORTED_STREAMS = COLLECTOR_SUPPORTED_STREAMS
RELAY_RETENTION_NS = 73 * 3600 * 1_000_000_000


def continuous_scan_start(now_ns: int, since_ns: int) -> int:
    """Reconcile every remotely retained capture on each successful poll."""
    return max(since_ns, now_ns - RELAY_RETENTION_NS, 0)


def request(url: str, api_key: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    return urllib.request.urlopen(req, timeout=timeout)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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
                    skipped += 1
                    continue
                raise RuntimeError(f"local capture differs from relay copy; left untouched: {target}")
            temp = target.with_name(target.name + f".download-{os.getpid()}")
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
                        publish_without_overwrite(decoded, target)
                    finally:
                        decoded.unlink(missing_ok=True)
                else:
                    if digest(temp) != entry["raw_sha256"]:
                        raise RuntimeError(f"raw capture hash mismatch for {entry['id']}")
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
