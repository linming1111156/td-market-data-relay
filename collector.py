#!/usr/bin/env python3
"""Future-only public Binance Spot and USD-M market-data capture."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime as dt
import decimal
import fcntl
import hashlib
import http.client
import io
import json
import os
import signal
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from python_socks.sync import Proxy as SocksProxy
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


MAX_FRAME_BYTES = 1_048_576
QUEUE_LIMIT = 256
UTC = dt.timezone.utc
SOCKS5_PROXY_ENV = "TD_MARKET_DATA_SOCKS5_PROXY"
SUPPORTED_STREAMS = frozenset({
    "spot-depth", "spot-trades", "spot-klines-1m",
    "usdm-depth", "usdm-funding-mark", "usdm-finalized-funding",
    "usdm-aggtrades", "usdm-klines-1m",
})


def now_ns() -> int:
    return time.time_ns()


def utc_stamp(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1_000_000_000, UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def configured_socks5_proxy() -> str | None:
    proxy_url = os.environ.get(SOCKS5_PROXY_ENV, "").strip()
    if not proxy_url:
        return None
    try:
        parsed = urllib.parse.urlsplit(proxy_url)
        port = parsed.port
    except ValueError as error:
        raise ValueError(f"{SOCKS5_PROXY_ENV} is not a valid SOCKS5 URL") from error
    if (parsed.scheme not in {"socks5", "socks5h"} or not parsed.hostname or port is None
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError(
            f"{SOCKS5_PROXY_ENV} must be socks5://HOST:PORT or socks5h://HOST:PORT"
        )
    return proxy_url


@contextlib.contextmanager
def open_request(request: urllib.request.Request, timeout: int):
    proxy_url = configured_socks5_proxy()
    if proxy_url is None:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            yield response
        return

    target = urllib.parse.urlsplit(request.full_url)
    if target.scheme != "https" or not target.hostname:
        raise ValueError("SOCKS5 collector requests must use HTTPS")
    port = target.port or 443
    proxy_parts = urllib.parse.urlsplit(proxy_url)
    socks5_url = urllib.parse.urlunsplit((
        "socks5", proxy_parts.netloc, "", "", ""
    ))
    proxy = SocksProxy.from_url(socks5_url, rdns=proxy_parts.scheme == "socks5h")
    proxy_socket = proxy.connect(dest_host=target.hostname, dest_port=port, timeout=timeout)
    try:
        tls_socket = ssl.create_default_context().wrap_socket(
            proxy_socket, server_hostname=target.hostname
        )
    except BaseException:
        proxy_socket.close()
        raise

    connection = http.client.HTTPConnection(target.hostname, port=port, timeout=timeout)
    connection.sock = tls_socket
    request_target = target.path or "/"
    if target.query:
        request_target += f"?{target.query}"
    try:
        connection.request(
            request.get_method(), request_target, headers=dict(request.header_items())
        )
        response = connection.getresponse()
        if response.status >= 400:
            error_body = response.read(64 * 1024)
            raise urllib.error.HTTPError(
                request.full_url,
                response.status,
                response.reason,
                response.headers,
                io.BytesIO(error_body),
            )
        try:
            yield response
        finally:
            response.close()
    finally:
        connection.close()


class CaptureLimit(Exception):
    pass


class SequenceGap(Exception):
    pass


class CaptureFile:
    def __init__(self, root: Path, kind: str, symbol: str, max_bytes: int,
                 source_revision: str, max_gap_ms: int):
        self.root = root.resolve(strict=True)
        self.kind = kind
        self.symbol = symbol
        self.max_bytes = max_bytes
        self.bytes = 0
        self.observations = 0
        self.gaps = 0
        self.disconnects = 0
        self.previous_frame_wall_ns: int | None = None
        self.previous_frame_monotonic_ns: int | None = None
        stamp = utc_stamp(now_ns())
        self.path = self.root / kind / symbol / f"{symbol}-{stamp}-{uuid.uuid4().hex}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("xb", buffering=0)
        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._write({
            "schema": "td_public_market_capture_record",
            "version": 1,
            "record_type": "capture_started",
            "recorded_at_unix_ns": now_ns(),
            "recorded_at_monotonic_ns": time.monotonic_ns(),
            "stream": kind,
            "symbol": symbol,
            "source_url": ("https://fapi.binance.com/fapi/v1/fundingRate"
                           if kind == "usdm-finalized-funding" else websocket_url(kind, symbol)),
            "source_host_id": os.environ.get("SOURCE_HOST_ID", "vps-market-data"),
            "source_revision": source_revision,
            "collector_executable_path": str(Path(__file__).resolve()),
            "collector_executable_sha256": sha256_file(Path(__file__).resolve()),
            "network_proxy_mode": (
                urllib.parse.urlsplit(proxy_url).scheme
                if (proxy_url := configured_socks5_proxy()) else "system_default"
            ),
            "max_gap_ms": max_gap_ms,
            "max_output_bytes": max_bytes,
        }, reserve=4096)

    def _write(self, record: dict[str, Any], reserve: int = 0) -> None:
        line = compact_json(record)
        if self.bytes + len(line) + reserve > self.max_bytes:
            raise CaptureLimit("capture byte limit reached")
        self.file.write(line)
        self.bytes += len(line)

    def record(self, record_type: str, **fields: Any) -> None:
        self._write({
            "schema": "td_public_market_capture_record",
            "version": 1,
            "record_type": record_type,
            **fields,
        }, reserve=4096)
        if self.bytes % (1024 * 1024) < 2048:
            os.fsync(self.file.fileno())

    def frame(self, received_ns: int, monotonic_ns: int, raw: str, **fields: Any) -> None:
        raw_bytes = raw.encode("utf-8")
        if len(raw_bytes) > MAX_FRAME_BYTES:
            raise ValueError("WebSocket frame exceeds one MiB")
        if self.previous_frame_wall_ns is not None and self.previous_frame_monotonic_ns is not None:
            wall_delta = received_ns - self.previous_frame_wall_ns
            monotonic_delta = monotonic_ns - self.previous_frame_monotonic_ns
            if wall_delta < 0 or monotonic_delta < 0 or abs(wall_delta - monotonic_delta) > 2_000_000_000:
                self.record("clock_discontinuity", recorded_at_unix_ns=received_ns,
                            wall_delta_ns=wall_delta, monotonic_delta_ns=monotonic_delta)
        self.previous_frame_wall_ns = received_ns
        self.previous_frame_monotonic_ns = monotonic_ns
        self.record(
            "market_frame",
            received_at_unix_ns=received_ns,
            receive_monotonic_ns=monotonic_ns,
            raw_frame=raw,
            raw_frame_sha256=sha256_bytes(raw_bytes),
            **fields,
        )
        self.observations += 1

    def gap(self, reason: str, **fields: Any) -> None:
        self.gaps += 1
        self.record("gap_detected", detected_at_unix_ns=now_ns(), reason=reason, **fields)

    def close(self, reason: str) -> None:
        if self.file.closed:
            return
        self._write({
            "schema": "td_public_market_capture_record",
            "version": 1,
            "record_type": "capture_stopped",
            "recorded_at_unix_ns": now_ns(),
            "recorded_at_monotonic_ns": time.monotonic_ns(),
            "reason": reason,
            "observation_count": self.observations,
            "gap_count": self.gaps,
            "disconnect_count": self.disconnects,
            "bytes_before_stop_record": self.bytes,
        })
        os.fsync(self.file.fileno())
        self.file.close()


def snapshot_request(kind: str, symbol: str) -> tuple[str, str]:
    if kind == "spot-depth":
        return "https://api.binance.com/api/v3/depth", "limit=5000"
    if kind == "usdm-depth":
        return "https://fapi.binance.com/fapi/v1/depth", "limit=1000"
    raise ValueError("snapshot requested for a non-depth stream")


def get_snapshot(kind: str, symbol: str) -> tuple[dict[str, Any], str]:
    endpoint, suffix = snapshot_request(kind, symbol)
    url = f"{endpoint}?{urllib.parse.urlencode({'symbol': symbol, 'limit': suffix.split('=')[1]})}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with open_request(request, timeout=15) as response:
        body = response.read(4 * 1024 * 1024 + 1)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError("depth snapshot exceeds four MiB")
    value = json.loads(body)
    if not isinstance(value, dict) or not isinstance(value.get("lastUpdateId"), int):
        raise ValueError("depth snapshot is missing lastUpdateId")
    if not isinstance(value.get("bids"), list) or not isinstance(value.get("asks"), list):
        raise ValueError("depth snapshot is missing bids or asks")
    return value, body.decode("utf-8")


def fetch_finalized_funding(symbol: str, start_time_ms: int) -> tuple[list[dict[str, Any]], str, str, int]:
    query = urllib.parse.urlencode({"symbol": symbol, "startTime": start_time_ms, "limit": 1000})
    url = f"https://fapi.binance.com/fapi/v1/fundingRate?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with open_request(request, timeout=15) as response:
        status = response.status
        raw = response.read(MAX_FRAME_BYTES + 1)
    if status != 200:
        raise ValueError(f"finalized Funding endpoint returned HTTP {status}")
    if len(raw) > MAX_FRAME_BYTES:
        raise ValueError("finalized Funding response exceeds one MiB")
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("finalized Funding response must be an array of at most 1000 rows")
    previous_time: int | None = None
    for row in rows:
        if not isinstance(row, dict) or row.get("symbol") != symbol:
            raise ValueError("finalized Funding row has an unexpected symbol")
        funding_time = row.get("fundingTime")
        if not isinstance(funding_time, int) or funding_time < start_time_ms:
            raise ValueError("finalized Funding row is outside the requested future-only cursor")
        rate = row.get("fundingRate")
        if not isinstance(rate, str) or not decimal.Decimal(rate).is_finite():
            raise ValueError("finalized Funding row has an invalid funding rate")
        mark_price = row.get("markPrice")
        if mark_price is not None and (
            not isinstance(mark_price, str) or not decimal.Decimal(mark_price).is_finite()
        ):
            raise ValueError("finalized Funding row has an invalid mark price")
        if previous_time is not None and funding_time <= previous_time:
            raise ValueError("finalized Funding rows are not in strict time order")
        previous_time = funding_time
    return rows, raw.decode("utf-8"), url, status


def read_or_initialize_funding_cursor(path: Path) -> tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        start_time_ms = time.time_ns() // 1_000_000
        temporary = path.with_name(path.name + ".new")
        temporary.unlink(missing_ok=True)
        with temporary.open("x", encoding="utf-8") as output:
            json.dump({"schema": "td_usdm_finalized_funding_cursor", "version": 1,
                       "start_time_ms": start_time_ms, "last_funding_time_ms": None}, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        return start_time_ms, start_time_ms
    if (not isinstance(value, dict) or value.get("schema") != "td_usdm_finalized_funding_cursor"
            or value.get("version") != 1 or not isinstance(value.get("start_time_ms"), int)):
        raise ValueError("Funding cursor has an unsupported or malformed schema")
    last_time = value.get("last_funding_time_ms")
    if last_time is not None and not isinstance(last_time, int):
        raise ValueError("Funding cursor's last settlement time is invalid")
    first_time = value["start_time_ms"]
    return first_time if last_time is None else last_time + 1, first_time


def write_funding_cursor(path: Path, start_time_ms: int, last_funding_time_ms: int) -> None:
    temporary = path.with_name(path.name + ".new")
    temporary.unlink(missing_ok=True)
    with temporary.open("x", encoding="utf-8") as output:
        json.dump({"schema": "td_usdm_finalized_funding_cursor", "version": 1,
                   "start_time_ms": start_time_ms,
                   "last_funding_time_ms": last_funding_time_ms}, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


async def run_finalized_funding_capture(args: argparse.Namespace, shutdown: asyncio.Event) -> None:
    symbol = args.symbol.upper()
    hot_root = Path(args.hot_root)
    hot_root.mkdir(parents=True, exist_ok=True)
    cursor_root = Path(args.cursor_root)
    cursor_root.mkdir(parents=True, exist_ok=True)
    cursor_path = cursor_root / f"usdm-finalized-funding-{symbol}.json"
    lock_path = cursor_path.with_suffix(".lock")
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another finalized Funding collector owns this symbol cursor") from error
        cursor_start_ms, cursor_initial_start_ms = read_or_initialize_funding_cursor(cursor_path)
        source_revision = os.environ.get("SOURCE_REVISION", "unreleased")
        capture = CaptureFile(hot_root, "usdm-finalized-funding", symbol, args.max_bytes,
                              source_revision, args.poll_seconds * 2_000)
        capture.record("funding_cursor_started", recorded_at_unix_ns=now_ns(),
                       request_start_time_ms=cursor_start_ms,
                       cursor_path=str(cursor_path), future_only=True)
        stop_at = time.monotonic() + args.duration_seconds
        next_poll = time.monotonic()
        stop_reason = "duration_elapsed"
        try:
            while time.monotonic() < stop_at and not shutdown.is_set():
                delay = next_poll - time.monotonic()
                if delay > 0:
                    try:
                        await asyncio.wait_for(shutdown.wait(), min(delay, stop_at - time.monotonic()))
                        if shutdown.is_set():
                            break
                    except asyncio.TimeoutError:
                        pass
                received_ns = now_ns()
                monotonic_ns = time.monotonic_ns()
                try:
                    rows, raw_response, url, http_status = await asyncio.to_thread(
                        fetch_finalized_funding, symbol, cursor_start_ms)
                    received_ns = now_ns()
                    monotonic_ns = time.monotonic_ns()
                    capture.record("funding_api_page", request_url=url,
                                   request_start_time_ms=cursor_start_ms,
                                   received_at_unix_ns=received_ns,
                                   receive_monotonic_ns=monotonic_ns,
                                   http_status=http_status, response_bytes=len(raw_response.encode()),
                                   raw_response=raw_response,
                                   raw_response_sha256=sha256_bytes(raw_response.encode()),
                                   record_count=len(rows), funding_rows=rows)
                    if rows:
                        last_funding_time_ms = rows[-1]["fundingTime"]
                        write_funding_cursor(cursor_path, cursor_initial_start_ms, last_funding_time_ms)
                        cursor_start_ms = last_funding_time_ms + 1
                    next_poll = time.monotonic() + args.poll_seconds
                except Exception as error:
                    capture.gap("finalized_funding_api_poll_failed", request_start_time_ms=cursor_start_ms,
                                error=f"{type(error).__name__}: {error}")
                    capture.record("funding_api_error", recorded_at_unix_ns=received_ns,
                                   request_start_time_ms=cursor_start_ms,
                                   error=f"{type(error).__name__}: {error}")
                    next_poll = time.monotonic() + min(args.poll_seconds, 30)
            if shutdown.is_set():
                stop_reason = "sigterm"
        except CaptureLimit:
            stop_reason = "capture_byte_limit"
        finally:
            capture.close(stop_reason)
            print(f"finalized Funding capture stopped: {capture.path} "
                  f"records={capture.observations} gaps={capture.gaps}")


def websocket_url(kind: str, symbol: str) -> str:
    lowered = symbol.lower()
    if kind == "spot-depth":
        return f"wss://stream.binance.com:9443/ws/{lowered}@depth@100ms"
    if kind == "usdm-depth":
        return f"wss://fstream.binance.com/public/ws/{lowered}@depth@100ms"
    if kind == "spot-trades":
        return f"wss://stream.binance.com:9443/ws/{lowered}@trade"
    if kind == "spot-klines-1m":
        return f"wss://stream.binance.com:9443/ws/{lowered}@kline_1m"
    if kind == "usdm-funding-mark":
        return f"wss://fstream.binance.com/market/ws/{lowered}@markPrice@1s"
    if kind == "usdm-aggtrades":
        return f"wss://fstream.binance.com/market/ws/{lowered}@aggTrade"
    if kind == "usdm-klines-1m":
        return f"wss://fstream.binance.com/market/ws/{lowered}@kline_1m"
    raise ValueError(f"unsupported stream: {kind}")


async def receive_depth_session(
    capture: CaptureFile,
    kind: str,
    symbol: str,
    max_gap_ms: int,
    stop_at: float,
    shutdown: asyncio.Event,
) -> None:
    url = websocket_url(kind, symbol)
    async with connect(
        url,
        proxy=configured_socks5_proxy() or True,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=0.5,
        max_size=MAX_FRAME_BYTES,
        max_queue=32,
        compression=None,
    ) as websocket:
        session_id = uuid.uuid4().hex
        capture.record("session_connected", recorded_at_unix_ns=now_ns(), session_id=session_id, url=url)
        queue: asyncio.Queue[tuple[int, int, str]] = asyncio.Queue(maxsize=QUEUE_LIMIT)
        disconnected = asyncio.Event()
        receiver_error: list[BaseException] = []

        async def receive_frames() -> None:
            try:
                async for raw in websocket:
                    wall = now_ns()
                    mono = time.monotonic_ns()
                    try:
                        queue.put_nowait((wall, mono, raw))
                    except asyncio.QueueFull as error:
                        receiver_error.append(error)
                        break
            except Exception as error:
                receiver_error.append(error)
            finally:
                disconnected.set()

        receiver = asyncio.create_task(receive_frames())
        try:
            snapshot, raw_snapshot = await asyncio.to_thread(get_snapshot, kind, symbol)
            snapshot_id = snapshot["lastUpdateId"]
            capture.record(
                "depth_snapshot",
                received_at_unix_ns=now_ns(),
                snapshot_update_id=snapshot_id,
                raw_snapshot=raw_snapshot,
                raw_snapshot_sha256=sha256_bytes(raw_snapshot.encode()),
                bids=snapshot["bids"],
                asks=snapshot["asks"],
                session_id=session_id,
            )
            last_id = snapshot_id
            bridged = False
            last_receive_mono = time.monotonic_ns()
            while time.monotonic() < stop_at and not shutdown.is_set():
                if receiver_error:
                    raise ConnectionError(f"depth receive queue overflow: {receiver_error[0]}")
                stop_remaining = stop_at - time.monotonic()
                gap_remaining_ns = max_gap_ms * 1_000_000 - (time.monotonic_ns() - last_receive_mono)
                if gap_remaining_ns <= 0:
                    gap_ns = time.monotonic_ns() - last_receive_mono
                    capture.gap("websocket_receive_timeout", receive_gap_ns=gap_ns,
                                max_gap_ms=max_gap_ms, session_id=session_id)
                    raise ConnectionError("depth WebSocket receive gap exceeded configured limit")
                remaining = min(stop_remaining, gap_remaining_ns / 1_000_000_000)
                try:
                    received_ns, monotonic_ns, raw = await asyncio.wait_for(
                        queue.get(), min(remaining, 1.0)
                    )
                except asyncio.TimeoutError:
                    if shutdown.is_set() or time.monotonic() >= stop_at:
                        break
                    gap_ns = time.monotonic_ns() - last_receive_mono
                    if gap_ns >= max_gap_ms * 1_000_000:
                        capture.gap("websocket_receive_timeout", receive_gap_ns=gap_ns,
                                    max_gap_ms=max_gap_ms, session_id=session_id)
                        raise ConnectionError("depth WebSocket receive gap exceeded configured limit")
                    continue
                delay_ns = monotonic_ns - last_receive_mono
                if delay_ns > max_gap_ms * 1_000_000:
                    capture.gap("receive_gap_exceeded", receive_gap_ns=delay_ns, session_id=session_id)
                    raise ConnectionError("depth WebSocket receive gap exceeded configured limit")
                last_receive_mono = monotonic_ns
                update = json.loads(raw)
                first_id = update.get("U")
                final_id = update.get("u")
                previous_id = update.get("pu")
                if not isinstance(first_id, int) or not isinstance(final_id, int):
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  accepted_for_book=False, reject_reason="update_ids_missing")
                    raise ValueError("depth frame is missing integer U/u update IDs")
                if final_id <= last_id:
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  first_update_id=first_id, final_update_id=final_id,
                                  previous_final_update_id=previous_id, accepted_for_book=False,
                                  reject_reason="older_than_snapshot_or_duplicate")
                    continue
                if not bridged:
                    if not (first_id <= last_id + 1 <= final_id):
                        capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                      first_update_id=first_id, final_update_id=final_id,
                                      previous_final_update_id=previous_id, accepted_for_book=False,
                                      reject_reason="snapshot_bridge_missing")
                        capture.gap("snapshot_bridge_missing", snapshot_update_id=snapshot_id,
                                    first_update_id=first_id, final_update_id=final_id, session_id=session_id)
                        raise SequenceGap("first depth event did not bridge the REST snapshot")
                    bridged = True
                elif kind == "usdm-depth" and previous_id != last_id:
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  first_update_id=first_id, final_update_id=final_id,
                                  previous_final_update_id=previous_id, accepted_for_book=False,
                                  reject_reason="previous_update_id_mismatch")
                    capture.gap("depth_sequence_discontinuity", expected_previous_final_id=last_id,
                                received_previous_final_id=previous_id, session_id=session_id)
                    raise SequenceGap("USD-M pu does not match prior final update ID")
                elif kind == "spot-depth" and first_id > last_id + 1:
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  first_update_id=first_id, final_update_id=final_id,
                                  accepted_for_book=False, reject_reason="update_id_gap")
                    capture.gap("depth_sequence_discontinuity", expected_first_update_id=last_id + 1,
                                received_first_update_id=first_id, session_id=session_id)
                    raise SequenceGap("Spot U does not bridge the previous update ID")
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), first_update_id=first_id,
                              final_update_id=final_id, previous_final_update_id=previous_id,
                              accepted_for_book=True)
                last_id = final_id
            if not bridged:
                capture.gap("no_snapshot_bridge_before_capture_end", snapshot_update_id=snapshot_id,
                            session_id=session_id)
        finally:
            receiver.cancel()
            try:
                await receiver
            except asyncio.CancelledError:
                pass
        if disconnected.is_set() and time.monotonic() < stop_at and not shutdown.is_set():
            error = receiver_error[0] if receiver_error else ConnectionClosed(1006, "connection closed")
            raise ConnectionError(f"WebSocket ended: {error}")
        capture.record("session_disconnected", recorded_at_unix_ns=now_ns(), session_id=session_id,
                       reason="local_shutdown" if shutdown.is_set() else
                       "hourly_rotation" if time.monotonic() >= stop_at else "session_end")


async def receive_observation_session(
    capture: CaptureFile,
    kind: str,
    symbol: str,
    max_gap_ms: int,
    stop_at: float,
    last_trade_id: list[int | None],
    last_closed_kline_open_ms: list[int | None],
    shutdown: asyncio.Event,
) -> None:
    url = websocket_url(kind, symbol)
    async with connect(url, open_timeout=10, ping_interval=20, ping_timeout=20,
                       close_timeout=0.5, max_size=MAX_FRAME_BYTES,
                       max_queue=32, compression=None,
                       proxy=configured_socks5_proxy() or True) as websocket:
        session_id = uuid.uuid4().hex
        capture.record("session_connected", recorded_at_unix_ns=now_ns(), session_id=session_id, url=url)
        last_receive_mono = time.monotonic_ns()
        while time.monotonic() < stop_at and not shutdown.is_set():
            stop_remaining = stop_at - time.monotonic()
            gap_remaining_ns = max_gap_ms * 1_000_000 - (time.monotonic_ns() - last_receive_mono)
            if gap_remaining_ns <= 0:
                gap_ns = time.monotonic_ns() - last_receive_mono
                capture.gap("websocket_receive_timeout", receive_gap_ns=gap_ns,
                            max_gap_ms=max_gap_ms, session_id=session_id)
                raise ConnectionError("observation WebSocket receive gap exceeded configured limit")
            remaining = min(stop_remaining, gap_remaining_ns / 1_000_000_000)
            try:
                raw = await asyncio.wait_for(websocket.recv(), min(remaining, 1.0))
            except asyncio.TimeoutError:
                if shutdown.is_set() or time.monotonic() >= stop_at:
                    break
                gap_ns = time.monotonic_ns() - last_receive_mono
                if gap_ns >= max_gap_ms * 1_000_000:
                    capture.gap("websocket_receive_timeout", receive_gap_ns=gap_ns,
                                max_gap_ms=max_gap_ms, session_id=session_id)
                    raise ConnectionError("observation WebSocket receive gap exceeded configured limit")
                continue
            received_ns = now_ns()
            monotonic_ns = time.monotonic_ns()
            delay_ns = monotonic_ns - last_receive_mono
            if delay_ns > max_gap_ms * 1_000_000:
                capture.gap("receive_gap_exceeded", receive_gap_ns=delay_ns, session_id=session_id)
                raise ConnectionError("observation receive gap exceeded configured limit")
            last_receive_mono = monotonic_ns
            update = json.loads(raw)
            if kind in {"spot-trades", "usdm-aggtrades"}:
                expected_event = "trade" if kind == "spot-trades" else "aggTrade"
                id_field = "t" if kind == "spot-trades" else "a"
                market_name = "Spot" if kind == "spot-trades" else "USD-M aggregate trade"
                if update.get("e") != expected_event or update.get("s") != symbol:
                    raise ValueError(f"unexpected {market_name} event identity")
                trade_id = update.get(id_field)
                if not isinstance(trade_id, int):
                    raise ValueError(f"{market_name} event has no integer trade ID")
                previous = last_trade_id[0]
                if previous is not None and trade_id != previous + 1:
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  event_time_ms=update.get("E"), trade_id=trade_id,
                                  accepted_for_sequence=False)
                    capture.gap(f"{kind}_id_discontinuity", expected_trade_id=previous + 1,
                                received_trade_id=trade_id, session_id=session_id)
                    # Keep capturing after a sequence gap. Reconnecting while
                    # retaining the old ID makes every subsequent session
                    # fail against the same stale expected_trade_id forever.
                    # Rebase only on a forward ID; an old/duplicate frame must
                    # not move the high-water mark backwards.
                    if trade_id > previous:
                        last_trade_id[0] = trade_id
                    continue
                last_trade_id[0] = trade_id
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), trade_time_ms=update.get("T"),
                              trade_id=trade_id, first_trade_id=update.get("f"),
                              last_trade_id=update.get("l"), buyer_is_maker=update.get("m"),
                              accepted_for_sequence=True)
            elif kind in {"spot-klines-1m", "usdm-klines-1m"}:
                kline = update.get("k")
                if (update.get("e") != "kline" or update.get("s") != symbol
                        or not isinstance(kline, dict) or kline.get("s") != symbol
                        or kline.get("i") != "1m"):
                    raise ValueError(f"unexpected {kind} kline event identity")
                open_ms, close_ms, is_closed = kline.get("t"), kline.get("T"), kline.get("x")
                if (not isinstance(open_ms, int) or not isinstance(close_ms, int)
                        or not isinstance(is_closed, bool) or close_ms != open_ms + 59_999):
                    raise ValueError(f"{kind} kline has invalid window fields")
                if is_closed:
                    previous = last_closed_kline_open_ms[0]
                    if previous is not None and open_ms != previous + 60_000:
                        capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                      event_time_ms=update.get("E"), kline_open_time_ms=open_ms,
                                      kline_close_time_ms=close_ms, kline_closed=True,
                                      accepted_for_sequence=False)
                        capture.gap(f"{kind}_minute_discontinuity",
                                    expected_open_time_ms=previous + 60_000,
                                    received_open_time_ms=open_ms, session_id=session_id)
                        raise SequenceGap(f"{kind} finalized 1m kline sequence is discontinuous")
                    last_closed_kline_open_ms[0] = open_ms
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), kline_open_time_ms=open_ms,
                              kline_close_time_ms=close_ms, kline_interval="1m",
                              kline_closed=is_closed, accepted_for_sequence=True)
            else:
                if update.get("e") != "markPriceUpdate" or update.get("s") != symbol:
                    raise ValueError("unexpected USD-M Mark Price event identity")
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), mark_price=update.get("p"),
                              index_price=update.get("i"), funding_rate=update.get("r"),
                              next_funding_time_ms=update.get("T"))
        capture.record("session_disconnected", recorded_at_unix_ns=now_ns(), session_id=session_id,
                       reason="local_shutdown" if shutdown.is_set() else "hourly_rotation")


async def run_capture(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    root = Path(args.hot_root)
    root.mkdir(parents=True, exist_ok=True)
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)
    if args.kind == "usdm-finalized-funding":
        await run_finalized_funding_capture(args, shutdown)
        return
    source_revision = os.environ.get("SOURCE_REVISION", "unreleased")
    capture = CaptureFile(root, args.kind, symbol, args.max_bytes, source_revision, args.max_gap_ms)
    stop_at = time.monotonic() + args.duration_seconds
    retry = 1.0
    attempt = 0
    last_trade_id: list[int | None] = [None]
    last_closed_kline_open_ms: list[int | None] = [None]
    stop_reason = "duration_elapsed"
    try:
        while time.monotonic() < stop_at and not shutdown.is_set():
            attempt += 1
            try:
                if args.kind in {"spot-depth", "usdm-depth"}:
                    await receive_depth_session(capture, args.kind, symbol, args.max_gap_ms,
                                                stop_at, shutdown)
                else:
                    await receive_observation_session(capture, args.kind, symbol, args.max_gap_ms,
                                                      stop_at, last_trade_id, last_closed_kline_open_ms,
                                                      shutdown)
                retry = 1.0
            except CaptureLimit:
                stop_reason = "capture_byte_limit"
                break
            except Exception as error:
                capture.disconnects += 1
                capture.record("session_disconnected", recorded_at_unix_ns=now_ns(),
                               attempt=attempt, reason=f"{type(error).__name__}: {error}",
                               reconnect_delay_ms=int(retry * 1000))
                if shutdown.is_set() or time.monotonic() >= stop_at:
                    break
                try:
                    await asyncio.wait_for(shutdown.wait(), retry)
                except asyncio.TimeoutError:
                    pass
                retry = min(retry * 2, 60.0)
        if shutdown.is_set():
            stop_reason = "sigterm"
    except CaptureLimit:
        stop_reason = "capture_byte_limit"
    finally:
        capture.close(stop_reason)
        print(f"capture stopped: {capture.path} records={capture.observations} gaps={capture.gaps} disconnects={capture.disconnects}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=sorted(SUPPORTED_STREAMS))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hot-root", default=os.environ.get("HOT_ROOT", "/var/lib/td-market-data/hot"))
    parser.add_argument("--max-gap-ms", type=int, default=5_000)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--duration-seconds", type=int, default=3_600)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--cursor-root", default=os.environ.get("CURSOR_ROOT", "/var/lib/td-market-data/state"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_gap_ms <= 0 or args.max_bytes < 8192 or args.duration_seconds <= 0 or args.poll_seconds <= 0:
        raise SystemExit("max gap, capture size, duration, and poll interval must be positive and bounded")
    configured_socks5_proxy()
    asyncio.run(run_capture(args))


if __name__ == "__main__":
    main()
