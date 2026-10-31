#!/usr/bin/env python3
"""Small authenticated archive and recovery API for public capture files."""

from __future__ import annotations

import hashlib
import hmac
import json
import fcntl
import os
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import zstandard


HOT = Path(os.environ.get("HOT_ROOT", "/var/lib/td-market-data/hot"))
ARCHIVE = Path(os.environ.get("ARCHIVE_ROOT", "/var/lib/td-market-data/archive"))
API_KEY = os.environ.get("RELAY_API_KEY", "")
HOT_SECONDS = int(os.environ.get("HOT_SECONDS", "3600"))
DELETE_SECONDS = int(os.environ.get("DELETE_SECONDS", "262800"))
MAX_ARCHIVE_BYTES = int(os.environ.get("MAX_ARCHIVE_BYTES", str(4 * 1024**3)))
MAX_HOT_BYTES = int(os.environ.get("MAX_HOT_BYTES", str(1024**3)))
MAX_REQUEST_BYTES = 1024
INDEX_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256(path: Path, modified_ns: int, size: int) -> str:
    key = (str(path), modified_ns, size)
    digest = INDEX_HASH_CACHE.get(key)
    if digest is None:
        digest = sha256_file(path)
        if path.stat().st_mtime_ns != modified_ns:
            raise OSError("capture changed while indexing")
        for stale in [item for item in INDEX_HASH_CACHE if item[0] == str(path)]:
            INDEX_HASH_CACHE.pop(stale, None)
        if len(INDEX_HASH_CACHE) >= 4096:
            INDEX_HASH_CACHE.pop(next(iter(INDEX_HASH_CACHE)))
        INDEX_HASH_CACHE[key] = digest
    return digest


def capture_started_at(path: Path) -> float:
    try:
        with path.open("rb") as stream:
            first = json.loads(stream.readline(16 * 1024))
        return int(first["recorded_at_unix_ns"]) / 1_000_000_000
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return path.stat().st_mtime


def compress_old_captures() -> None:
    now = time.time()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    HOT.mkdir(parents=True, exist_ok=True)
    # Oldest archives are discarded first when the explicitly bounded quota is reached.
    archived = sorted((p for p in ARCHIVE.rglob("*.jsonl.zst") if p.is_file()), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in archived)
    for path in archived:
        if now - path.stat().st_mtime > DELETE_SECONDS:
            total -= path.stat().st_size
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".meta.json").unlink(missing_ok=True)
    for path in sorted(HOT.rglob("*.jsonl")):
        if not path.is_file() or now - capture_started_at(path) <= HOT_SECONDS:
            continue
        relative = path.relative_to(HOT)
        destination = ARCHIVE / relative.with_suffix(path.suffix + ".zst")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_name(destination.name + f".tmp-{os.getpid()}")
        meta_path = destination.with_name(destination.name + ".meta.json")
        meta_temp = meta_path.with_name(meta_path.name + f".tmp-{os.getpid()}")
        try:
            # A source file still exists if a prior rotation was interrupted.
            # Reuse a complete archive, or rebuild an orphaned/tampered one.
            if destination.exists() and meta_path.exists():
                try:
                    prior = json.loads(meta_path.read_text())
                    prior_valid = (prior.get("compressed_sha256") == sha256_file(destination)
                                   and prior.get("raw_sha256") == sha256_file(path))
                except (OSError, json.JSONDecodeError):
                    prior_valid = False
                if prior_valid:
                    path.unlink()
                    continue
                destination.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
            elif destination.exists():
                destination.unlink()
            with path.open("rb") as src, temp.open("xb") as raw:
                try:
                    fcntl.flock(src.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    temp.unlink(missing_ok=True)
                    continue
                source_digest = hashlib.sha256()
                source_bytes = 0
                with zstandard.ZstdCompressor(level=3, write_checksum=True).stream_writer(raw, closefd=False) as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        source_digest.update(chunk)
                        source_bytes += len(chunk)
                        dst.write(chunk)
                raw.flush()
                os.fsync(raw.fileno())
            size = temp.stat().st_size
            start_time = capture_started_at(path)
            os.utime(temp, (start_time, start_time))
            while total + size > MAX_ARCHIVE_BYTES and archived:
                oldest = archived.pop(0)
                try:
                    old_size = oldest.stat().st_size
                    oldest.unlink()
                    total -= old_size
                except FileNotFoundError:
                    pass
            if total + size > MAX_ARCHIVE_BYTES:
                temp.unlink(missing_ok=True)
                print(f"archive cap could not fit capture: {relative}", flush=True)
                continue
            metadata = {"schema": "td_market_data_archive_metadata_v1",
                        "capture_id": path.name.split("-")[-1].split(".")[0],
                        "raw_sha256": source_digest.hexdigest(), "raw_size_bytes": source_bytes,
                        "compressed_sha256": sha256_file(temp), "compressed_size_bytes": size,
                        "captured_at_unix_seconds": start_time}
            with meta_temp.open("xb") as metadata_file:
                metadata_file.write(json.dumps(metadata, separators=(",", ":")).encode() + b"\n")
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            os.replace(temp, destination)
            os.utime(destination, (start_time, start_time))
            os.replace(meta_temp, meta_path)
            path.unlink()
            total += size
        finally:
            temp.unlink(missing_ok=True)
            meta_temp.unlink(missing_ok=True)
    enforce_hot_cap()


def enforce_hot_cap() -> None:
    files = sorted((path for path in HOT.rglob("*.jsonl") if path.is_file()),
                   key=lambda path: path.stat().st_mtime)
    total = sum(path.stat().st_size for path in files)
    for path in files:
        if total <= MAX_HOT_BYTES:
            break
        try:
            with path.open("rb") as stream:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                size = path.stat().st_size
                path.unlink()
                total -= size
                print(f"hot storage cap discarded oldest capture: {path.relative_to(HOT)}", flush=True)
        except FileNotFoundError:
            continue
    if total > MAX_HOT_BYTES:
        print(f"hot storage remains above cap because captures are active: bytes={total}", flush=True)


def retention_worker() -> None:
    while True:
        try:
            compress_old_captures()
        except Exception as error:
            print(f"retention error: {type(error).__name__}: {error}", flush=True)
        time.sleep(60)


def capture_entry(root: Path, path: Path) -> dict[str, object] | None:
    st = path.stat()
    if root == HOT and time.time() - st.st_mtime < 60:
        return None
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) != 3:
        return None
    stream, symbol = parts[0], parts[1]
    capture_id = path.name.split("-")[-1].split(".")[0]
    if path.suffix == ".zst":
        meta_path = path.with_name(path.name + ".meta.json")
        with meta_path.open("rb") as meta_file:
            metadata = json.loads(meta_file.read(16 * 1024))
        if (metadata.get("schema") != "td_market_data_archive_metadata_v1"
                or metadata.get("capture_id") != capture_id
                or metadata.get("compressed_size_bytes") != st.st_size):
            return None
        file_sha = metadata.get("compressed_sha256")
        raw_sha = metadata.get("raw_sha256")
    else:
        file_sha = cached_sha256(path, st.st_mtime_ns, st.st_size)
        raw_sha = file_sha
    if (not isinstance(file_sha, str) or len(file_sha) != 64
            or any(char not in "0123456789abcdef" for char in file_sha)
            or not isinstance(raw_sha, str) or len(raw_sha) != 64
            or any(char not in "0123456789abcdef" for char in raw_sha)):
        return None
    return {"id": capture_id, "stream": stream, "symbol": symbol, "path": str(relative),
            "compressed": path.suffix == ".zst", "size_bytes": st.st_size,
            "modified_at_unix_ns": st.st_mtime_ns, "sha256": file_sha, "raw_sha256": raw_sha}


def capture_index(since_ns: int, until_ns: int, stream_filter: str, symbol_filter: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for root, suffix in ((HOT, ".jsonl"), (ARCHIVE, ".jsonl.zst")):
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            try:
                entry = capture_entry(root, path)
                if entry is None:
                    continue
                if not since_ns <= int(entry["modified_at_unix_ns"]) <= until_ns:
                    continue
                if stream_filter and entry["stream"] != stream_filter:
                    continue
                if symbol_filter and entry["symbol"] != symbol_filter.upper():
                    continue
                result.append(entry)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
    return sorted(result, key=lambda item: (item["modified_at_unix_ns"], item["path"]))


class Handler(BaseHTTPRequestHandler):
    server_version = "td-market-data-relay/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.client_address[0]} {fmt % args}", flush=True)

    def authorized(self) -> bool:
        given = self.headers.get("Authorization", "")
        expected = f"Bearer {API_KEY}"
        return bool(API_KEY) and hmac.compare_digest(given.encode(), expected.encode())

    def send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/healthz":
            self.send_json(200, {"status": "ok"})
            return
        if not self.authorized():
            self.send_json(401, {"error": "unauthorized"})
            return
        params = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/v1/captures":
            try:
                since = int(params.get("since_ns", ["0"])[0])
                until = int(params.get("until_ns", [str(time.time_ns())])[0])
                if since < 0 or until < since:
                    raise ValueError("invalid time range")
            except ValueError:
                self.send_json(400, {"error": "invalid time range"})
                return
            entries = capture_index(since, until, params.get("stream", [""])[0],
                                    params.get("symbol", [""])[0])
            self.send_json(200, {"schema": "td_market_data_relay_index_v1", "captures": entries})
            return
        if parsed.path.startswith("/v1/captures/"):
            capture_id = parsed.path.removeprefix("/v1/captures/")
            if len(capture_id) != 32 or any(char not in "0123456789abcdef" for char in capture_id):
                self.send_json(404, {"error": "capture not found"})
                return
            match = next((entry for root, suffix in ((HOT, ".jsonl"), (ARCHIVE, ".jsonl.zst"))
                          if root.exists() for path in root.rglob(f"*-{capture_id}{suffix}")
                          for entry in [capture_entry(root, path)] if entry is not None), None)
            if match is None:
                self.send_json(404, {"error": "capture not found"})
                return
            rel = Path(str(match["path"]))
            root = ARCHIVE if bool(match["compressed"]) else HOT
            path = (root / str(match["stream"]) / str(match["symbol"]) / rel.name).resolve()
            if root.resolve() not in path.parents or not path.is_file():
                self.send_json(404, {"error": "capture not found"})
                return
            size = path.stat().st_size
            if size > MAX_REQUEST_BYTES * 1024 * 1024:
                self.send_json(413, {"error": "capture exceeds download limit"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/zstd" if path.suffix == ".zst" else "application/x-ndjson")
            self.send_header("Content-Length", str(size))
            self.send_header("X-Content-SHA256", str(match["sha256"]))
            self.send_header("X-Capture-Id", capture_id)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    self.wfile.write(chunk)
            return
        self.send_json(404, {"error": "not found"})


def main() -> None:
    if not API_KEY or len(API_KEY) < 32:
        raise SystemExit("RELAY_API_KEY must be set to a random value of at least 32 characters")
    if HOT_SECONDS != 3600 or DELETE_SECONDS != 73 * 3600:
        raise SystemExit("retention policy is fixed: raw=1h, archive deletion=73h")
    if MAX_ARCHIVE_BYTES < 512 * 1024 * 1024 or MAX_HOT_BYTES < 256 * 1024 * 1024:
        raise SystemExit("archive and hot storage caps are too small for bounded captures")
    threading.Thread(target=retention_worker, daemon=True).start()
    server = ThreadingHTTPServer((os.environ.get("RELAY_BIND", "0.0.0.0"),
                                  int(os.environ.get("RELAY_PORT", "8080"))), Handler)
    server.daemon_threads = True
    print("relay API listening", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
