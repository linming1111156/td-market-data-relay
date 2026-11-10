#!/usr/bin/env python3
"""Small authenticated archive and recovery API for public capture files."""

from __future__ import annotations

import hashlib
import hmac
import json
import fcntl
import os
import re
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
MAX_LOSS_LEDGER_BYTES = 16 * 1024 * 1024
LOSS_SCHEMA = "td_market_data_relay_early_discard_v1"
INDEX_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _loss_ledger_path() -> Path:
    return ARCHIVE / "early-discard-ledger.jsonl"


def _read_loss_rows() -> list[dict[str, object]]:
    path = _loss_ledger_path()
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_LOSS_LEDGER_BYTES:
        raise OSError("early-discard ledger is not a bounded regular file")
    rows = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 16 * 1024:
                raise OSError(f"early-discard ledger line {line_number} is truncated or oversized")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise OSError(f"early-discard ledger line {line_number} is invalid JSON") from error
            if not isinstance(row, dict) or row.get("schema") != LOSS_SCHEMA:
                raise OSError(f"early-discard ledger line {line_number} has an invalid schema")
            rows.append(row)
    return rows


def _write_loss_rows(rows: list[dict[str, object]]) -> None:
    path = _loss_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(json.dumps(row, separators=(",", ":"), sort_keys=True).encode() + b"\n"
                       for row in rows)
    if len(encoded) > MAX_LOSS_LEDGER_BYTES:
        raise OSError("early-discard ledger reached its 16 MiB safety bound")
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    created = False
    try:
        with temp.open("xb") as stream:
            created = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if created:
            temp.unlink(missing_ok=True)
    parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def prune_loss_ledger(now_ns: int | None = None) -> None:
    path = _loss_ledger_path()
    if not path.exists():
        return
    now_ns = time.time_ns() if now_ns is None else now_ns
    cutoff_ns = now_ns - DELETE_SECONDS * 1_000_000_000
    rows = _read_loss_rows()
    retained = [row for row in rows
                if isinstance(row.get("discarded_at_ns"), int)
                and row["discarded_at_ns"] >= cutoff_ns]
    if len(retained) != len(rows):
        _write_loss_rows(retained)


def capture_lifecycle_bounds(path: Path) -> tuple[int, int]:
    def records(stream):
        first = last = None
        total = 0
        for line_number, raw in enumerate(stream, 1):
            total += len(raw)
            if total > 256 * 1024 * 1024 or len(raw) > 2 * 1024 * 1024 or not raw.endswith(b"\n"):
                raise OSError("capture lifecycle scan encountered a truncated or oversized record")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise OSError("capture lifecycle scan encountered invalid JSON") from error
            if not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record":
                raise OSError("capture lifecycle scan encountered an unsupported record")
            if first is None:
                first = row
            last = row
        if (not isinstance(first, dict) or first.get("record_type") != "capture_started"
                or not isinstance(last, dict) or last.get("record_type") != "capture_stopped"):
            raise OSError("capture lacks a complete start/stop lifecycle")
        started_ns, stopped_ns = first.get("recorded_at_unix_ns"), last.get("recorded_at_unix_ns")
        if (not isinstance(started_ns, int) or isinstance(started_ns, bool)
                or not isinstance(stopped_ns, int) or isinstance(stopped_ns, bool)
                or started_ns < 0 or stopped_ns < started_ns):
            raise OSError("capture lifecycle timestamps are invalid")
        return started_ns, stopped_ns

    if path.suffix == ".zst":
        try:
            with path.open("rb") as source:
                with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
                    import io

                    return records(io.BufferedReader(decoded))
        except zstandard.ZstdError as error:
            raise OSError("archived capture has invalid zstd data") from error
    with path.open("rb") as source:
        return records(source)


def record_early_discard(root: Path, path: Path, reason: str) -> bool:
    relative = path.relative_to(root)
    parts = relative.parts
    if len(parts) != 3:
        raise OSError(f"cannot record early discard for unexpected path: {relative}")
    stream, symbol, filename = parts
    match = re.fullmatch(rf"{re.escape(symbol)}-([0-9]{{8}}T[0-9]{{6}}Z)-([0-9a-f]{{32}})\.jsonl(?:\.zst)?", filename)
    if match is None:
        raise OSError(f"cannot record early discard for malformed capture name: {relative}")
    raw_hash: str | None = None
    started_ns: int | None = None
    stopped_ns: int | None = None
    if root == ARCHIVE:
        meta_path = path.with_name(path.name + ".meta.json")
        try:
            metadata = json.loads(meta_path.read_bytes())
            candidate_hash = metadata.get("raw_sha256")
            if isinstance(candidate_hash, str) and re.fullmatch(r"[0-9a-f]{64}", candidate_hash):
                raw_hash = candidate_hash
            candidate_start, candidate_stop = metadata.get("capture_started_at_ns"), metadata.get("capture_stopped_at_ns")
            if (isinstance(candidate_start, int) and not isinstance(candidate_start, bool)
                    and isinstance(candidate_stop, int) and not isinstance(candidate_stop, bool)
                    and 0 <= candidate_start <= candidate_stop):
                started_ns, stopped_ns = candidate_start, candidate_stop
        except (OSError, json.JSONDecodeError):
            pass
    try:
        if started_ns is None or stopped_ns is None:
            started_ns, stopped_ns = capture_lifecycle_bounds(path)
        if raw_hash is None:
            raw_hash = sha256_file(path) if root == HOT else None
    except OSError as error:
        print(f"early discard refused because capture lifecycle is unverifiable: {error}", flush=True)
        return False
    now_ns = time.time_ns()
    row: dict[str, object] = {"schema": LOSS_SCHEMA, "version": 1,
                              "capture_id": match.group(2), "stream": stream,
                              "symbol": symbol, "path": str(relative),
                              "reason": reason, "discarded_at_ns": now_ns,
                              "capture_started_at_ns": started_ns,
                              "capture_stopped_at_ns": stopped_ns,
                              "raw_sha256": raw_hash}
    existing = _read_loss_rows()
    if any(item.get("capture_id") == match.group(2) and item.get("path") == str(relative)
           and item.get("reason") == reason for item in existing):
        return True
    cutoff_ns = now_ns - DELETE_SECONDS * 1_000_000_000
    retained = [item for item in existing
                if isinstance(item.get("discarded_at_ns"), int)
                and item["discarded_at_ns"] >= cutoff_ns]
    retained.append(row)
    try:
        _write_loss_rows(retained)
    except OSError as error:
        print(f"early discard refused because loss ledger could not be updated: {error}", flush=True)
        return False
    return True


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


def _capture_quality_record(raw: bytes) -> dict[str, object]:
    if len(raw) > 2 * 1024 * 1024 or not raw.endswith(b"\n"):
        raise OSError("capture quality record is truncated or oversized")
    try:
        row = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OSError("capture quality record is invalid JSON") from error
    if (not isinstance(row, dict)
            or row.get("schema") != "td_public_market_capture_record"
            or row.get("version") != 1):
        raise OSError("capture quality record has an unsupported schema")
    return row


def capture_quality(path: Path) -> dict[str, object]:
    """Read lifecycle quality without parsing every market frame.

    Raw hot files use their first and last JSONL records. Archived files fall
    back to a streaming scan only when older metadata lacks quality fields.
    """
    first: dict[str, object] | None = None
    last: dict[str, object] | None = None
    if path.suffix == ".zst":
        try:
            with path.open("rb") as source:
                with zstandard.ZstdDecompressor().stream_reader(source) as decoded:
                    import io

                    stream = io.BufferedReader(decoded)
                    for raw in stream:
                        row = _capture_quality_record(raw)
                        if first is None:
                            first = row
                        last = row
        except zstandard.ZstdError as error:
            raise OSError("archived capture has invalid zstd data") from error
    else:
        with path.open("rb") as source:
            first = _capture_quality_record(source.readline(2 * 1024 * 1024 + 1))
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - 2 * 1024 * 1024), os.SEEK_SET)
            tail = source.read()
        if not tail.endswith(b"\n"):
            raise OSError("capture quality footer is incomplete")
        last_raw = tail.splitlines()[-1] if tail.splitlines() else b""
        last = _capture_quality_record(last_raw + b"\n")
    if first is None or first.get("record_type") != "capture_started":
        raise OSError("capture quality header is missing")
    started_ns = first.get("recorded_at_unix_ns")
    stopped = last if last and last.get("record_type") == "capture_stopped" else None
    stopped_ns = stopped.get("recorded_at_unix_ns") if stopped else None
    if (not isinstance(started_ns, int) or isinstance(started_ns, bool)
            or started_ns < 0):
        raise OSError("capture quality start timestamp is invalid")
    closed = stopped is not None
    if stopped_ns is not None and (
            not isinstance(stopped_ns, int) or isinstance(stopped_ns, bool)
            or stopped_ns < started_ns):
        raise OSError("capture quality stop timestamp is invalid")
    gap_count = stopped.get("gap_count", 0) if stopped else None
    disconnect_count = stopped.get("disconnect_count", 0) if stopped else None
    stop_reason = stopped.get("reason") if stopped else None
    if (gap_count is not None and (not isinstance(gap_count, int) or gap_count < 0)
            or disconnect_count is not None
            and (not isinstance(disconnect_count, int) or disconnect_count < 0)):
        raise OSError("capture quality counters are invalid")
    trainable = bool(
        closed and gap_count == 0 and disconnect_count == 0
        and stop_reason in {"duration_elapsed", "sigterm"}
    )
    return {
        "capture_started_at_ns": started_ns,
        "capture_stopped_at_ns": stopped_ns,
        "gap_count": gap_count,
        "disconnect_count": disconnect_count,
        "stop_reason": stop_reason,
        "closed": closed,
        "trainable": trainable,
    }


def compress_old_captures() -> None:
    now = time.time()
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    HOT.mkdir(parents=True, exist_ok=True)
    cleanup_stale_temps(ARCHIVE)
    try:
        prune_loss_ledger()
    except Exception as error:
        print(f"early-discard ledger pruning error: {type(error).__name__}: {error}", flush=True)
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
        recovery_path = path.with_name(path.name + ".recovery.json")
        recovery_provenance = None
        if recovery_path.exists() or recovery_path.is_symlink():
            try:
                if recovery_path.is_symlink() or not recovery_path.is_file():
                    raise ValueError("recovery provenance is not a regular file")
                recovery_provenance = json.loads(recovery_path.read_text())
                if (recovery_provenance.get("schema") != "td_market_data_relay_recovery_provenance_v1"
                        or recovery_provenance.get("version") != 1
                        or recovery_provenance.get("capture_id") != path.name.split("-")[-1].split(".")[0]
                        or recovery_provenance.get("origin") != "vps_recovered"
                        or not isinstance(recovery_provenance.get("recovered_at_unix_ns"), int)
                        or isinstance(recovery_provenance.get("recovered_at_unix_ns"), bool)
                        or recovery_provenance["recovered_at_unix_ns"] <= 0
                        or not isinstance(recovery_provenance.get("raw_sha256"), str)
                        or len(recovery_provenance["raw_sha256"]) != 64):
                    raise ValueError("recovery provenance fields are invalid")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"capture recovery provenance rejected; leaving raw capture: {relative}: {error}", flush=True)
                continue
        meta_temp = meta_path.with_name(meta_path.name + f".tmp-{os.getpid()}")
        try:
            # A source file still exists if a prior rotation was interrupted.
            # Reuse a complete archive, or rebuild an orphaned/tampered one.
            if destination.exists() and meta_path.exists():
                try:
                    prior = json.loads(meta_path.read_text())
                    prior_valid = (prior.get("schema") == "td_market_data_archive_metadata_v1"
                                   and prior.get("capture_id") == path.name.split("-")[-1].split(".")[0]
                                   and prior.get("compressed_sha256") == sha256_file(destination)
                                   and prior.get("raw_sha256") == sha256_file(path)
                                   and prior.get("recovery_provenance") == recovery_provenance)
                except (OSError, json.JSONDecodeError):
                    prior_valid = False
                if prior_valid:
                    path.unlink()
                    recovery_path.unlink(missing_ok=True)
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
            if (recovery_provenance is not None
                    and recovery_provenance.get("raw_sha256") != source_digest.hexdigest()):
                temp.unlink(missing_ok=True)
                print(f"capture recovery hash mismatch; leaving raw capture: {relative}", flush=True)
                continue
            start_time = capture_started_at(path)
            os.utime(temp, (start_time, start_time))
            while total + size > MAX_ARCHIVE_BYTES and archived:
                oldest = archived.pop(0)
                try:
                    old_size = oldest.stat().st_size
                    if not record_early_discard(ARCHIVE, oldest, "archive_quota"):
                        break
                    oldest.unlink()
                    oldest.with_name(oldest.name + ".meta.json").unlink(missing_ok=True)
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
                        "captured_at_unix_seconds": start_time,
                        **capture_quality(path),
                        **({"recovery_provenance": recovery_provenance}
                           if recovery_provenance is not None else {})}
            with meta_temp.open("xb") as metadata_file:
                metadata_file.write(json.dumps(metadata, separators=(",", ":")).encode() + b"\n")
                metadata_file.flush()
                os.fsync(metadata_file.fileno())
            os.replace(temp, destination)
            os.utime(destination, (start_time, start_time))
            os.replace(meta_temp, meta_path)
            path.unlink()
            recovery_path.unlink(missing_ok=True)
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
                if not record_early_discard(HOT, path, "hot_storage_quota"):
                    continue
                path.unlink()
                path.with_name(path.name + ".recovery.json").unlink(missing_ok=True)
                total -= size
                print(f"hot storage cap discarded oldest capture: {path.relative_to(HOT)}", flush=True)
        except FileNotFoundError:
            continue
    if total > MAX_HOT_BYTES:
        print(f"hot storage remains above cap because captures are active: bytes={total}", flush=True)


def cleanup_stale_temps(root: Path) -> None:
    for path in root.rglob("*.tmp-*"):
        match = re.search(r"\.tmp-(\d+)$", path.name)
        if match is None:
            continue
        owner_pid = int(match.group(1))
        if owner_pid == os.getpid():
            continue
        try:
            os.kill(owner_pid, 0)
            continue
        except ProcessLookupError:
            path.unlink(missing_ok=True)
        except PermissionError:
            continue


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
        quality = {key: metadata[key] for key in (
            "capture_started_at_ns", "capture_stopped_at_ns", "gap_count",
            "disconnect_count", "stop_reason", "closed", "trainable")
                   if key in metadata}
        if len(quality) != 7:
            quality = capture_quality(path)
    else:
        file_sha = cached_sha256(path, st.st_mtime_ns, st.st_size)
        raw_sha = file_sha
        quality = capture_quality(path)
    if (not isinstance(file_sha, str) or len(file_sha) != 64
            or any(char not in "0123456789abcdef" for char in file_sha)
            or not isinstance(raw_sha, str) or len(raw_sha) != 64
            or any(char not in "0123456789abcdef" for char in raw_sha)):
        return None
    return {"id": capture_id, "stream": stream, "symbol": symbol, "path": str(relative),
            "compressed": path.suffix == ".zst", "size_bytes": st.st_size,
            "modified_at_unix_ns": st.st_mtime_ns, "sha256": file_sha, "raw_sha256": raw_sha,
            **quality}


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


def capture_losses(since_ns: int, until_ns: int, stream_filter: str,
                   symbol_filter: str) -> list[dict[str, object]]:
    rows = []
    for row in _read_loss_rows():
        discarded_ns = row.get("discarded_at_ns")
        if not isinstance(discarded_ns, int) or isinstance(discarded_ns, bool):
            raise OSError("early-discard ledger has an invalid timestamp")
        if not since_ns <= discarded_ns <= until_ns:
            continue
        if stream_filter and row.get("stream") != stream_filter:
            continue
        if symbol_filter and row.get("symbol") != symbol_filter.upper():
            continue
        rows.append(row)
    return rows


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
            losses = capture_losses(since, until, params.get("stream", [""])[0],
                                    params.get("symbol", [""])[0])
            self.send_json(200, {"schema": "td_market_data_relay_index_v1",
                                 "captures": entries, "losses": losses})
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
