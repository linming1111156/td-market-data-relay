#!/usr/bin/env python3
"""Future-only public Binance Spot and USD-M market-data capture."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed


MAX_FRAME_BYTES = 1_048_576
QUEUE_LIMIT = 256
UTC = dt.timezone.utc


def now_ns() -> int:
    return time.time_ns()


def utc_stamp(ns: int) -> str:
    return dt.datetime.fromtimestamp(ns / 1_000_000_000, UTC).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


class CaptureLimit(Exception):
    pass


class SequenceGap(Exception):
    pass


class CaptureFile:
    def __init__(self, root: Path, kind: str, symbol: str, max_bytes: int, source_revision: str):
        self.root = root.resolve(strict=True)
        self.kind = kind
        self.symbol = symbol
        self.max_bytes = max_bytes
        self.bytes = 0
        self.observations = 0
        self.gaps = 0
        self.disconnects = 0
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
            "stream": kind,
            "symbol": symbol,
            "source_host_id": os.environ.get("SOURCE_HOST_ID", "vps-market-data"),
            "source_revision": source_revision,
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
    with urllib.request.urlopen(request, timeout=15) as response:
        body = response.read(4 * 1024 * 1024 + 1)
    if len(body) > 4 * 1024 * 1024:
        raise ValueError("depth snapshot exceeds four MiB")
    value = json.loads(body)
    if not isinstance(value, dict) or not isinstance(value.get("lastUpdateId"), int):
        raise ValueError("depth snapshot is missing lastUpdateId")
    if not isinstance(value.get("bids"), list) or not isinstance(value.get("asks"), list):
        raise ValueError("depth snapshot is missing bids or asks")
    return value, body.decode("utf-8")


def websocket_url(kind: str, symbol: str) -> str:
    lowered = symbol.lower()
    if kind == "spot-depth":
        return f"wss://stream.binance.com:9443/ws/{lowered}@depth@100ms"
    if kind == "usdm-depth":
        return f"wss://fstream.binance.com/public/ws/{lowered}@depth@100ms"
    if kind == "spot-trades":
        return f"wss://stream.binance.com:9443/ws/{lowered}@trade"
    if kind == "usdm-funding-mark":
        return f"wss://fstream.binance.com/market/ws/{lowered}@markPrice@1s"
    raise ValueError(f"unsupported stream: {kind}")


async def receive_depth_session(
    capture: CaptureFile,
    kind: str,
    symbol: str,
    max_gap_ms: int,
    stop_at: float,
) -> None:
    url = websocket_url(kind, symbol)
    async with connect(
        url,
        open_timeout=10,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=10,
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
            while time.monotonic() < stop_at:
                if receiver_error:
                    raise ConnectionError(f"depth receive queue overflow: {receiver_error[0]}")
                remaining = min(max_gap_ms / 1000, max(0.05, stop_at - time.monotonic()))
                try:
                    received_ns, monotonic_ns, raw = await asyncio.wait_for(queue.get(), remaining)
                except asyncio.TimeoutError:
                    if time.monotonic() >= stop_at:
                        break
                    capture.gap("websocket_receive_timeout", max_gap_ms=max_gap_ms, session_id=session_id)
                    raise ConnectionError("depth WebSocket receive gap exceeded configured limit")
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
        if disconnected.is_set() and time.monotonic() < stop_at:
            error = receiver_error[0] if receiver_error else ConnectionClosed(1006, "connection closed")
            raise ConnectionError(f"WebSocket ended: {error}")
        capture.record("session_disconnected", recorded_at_unix_ns=now_ns(), session_id=session_id,
                       reason="hourly_rotation" if time.monotonic() >= stop_at else "session_end")


async def receive_observation_session(
    capture: CaptureFile,
    kind: str,
    symbol: str,
    max_gap_ms: int,
    stop_at: float,
    last_trade_id: list[int | None],
) -> None:
    url = websocket_url(kind, symbol)
    async with connect(url, open_timeout=10, ping_interval=20, ping_timeout=20,
                       close_timeout=10, max_size=MAX_FRAME_BYTES,
                       max_queue=32, compression=None) as websocket:
        session_id = uuid.uuid4().hex
        capture.record("session_connected", recorded_at_unix_ns=now_ns(), session_id=session_id, url=url)
        last_receive_mono = time.monotonic_ns()
        while time.monotonic() < stop_at:
            remaining = min(max_gap_ms / 1000, max(0.05, stop_at - time.monotonic()))
            try:
                raw = await asyncio.wait_for(websocket.recv(), remaining)
            except asyncio.TimeoutError:
                if time.monotonic() >= stop_at:
                    break
                capture.gap("websocket_receive_timeout", max_gap_ms=max_gap_ms, session_id=session_id)
                raise ConnectionError("observation WebSocket receive gap exceeded configured limit")
            received_ns = now_ns()
            monotonic_ns = time.monotonic_ns()
            delay_ns = monotonic_ns - last_receive_mono
            if delay_ns > max_gap_ms * 1_000_000:
                capture.gap("receive_gap_exceeded", receive_gap_ns=delay_ns, session_id=session_id)
                raise ConnectionError("observation receive gap exceeded configured limit")
            last_receive_mono = monotonic_ns
            update = json.loads(raw)
            if kind == "spot-trades":
                if update.get("e") != "trade" or update.get("s") != symbol:
                    raise ValueError("unexpected Spot trade event identity")
                trade_id = update.get("t")
                if not isinstance(trade_id, int):
                    raise ValueError("Spot trade event has no integer trade ID")
                previous = last_trade_id[0]
                if previous is not None and trade_id != previous + 1:
                    capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                                  event_time_ms=update.get("E"), trade_id=trade_id,
                                  accepted_for_sequence=False)
                    capture.gap("spot_trade_id_discontinuity", expected_trade_id=previous + 1,
                                received_trade_id=trade_id, session_id=session_id)
                    raise SequenceGap("Spot trade ID is discontinuous")
                last_trade_id[0] = trade_id
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), trade_time_ms=update.get("T"),
                              trade_id=trade_id, accepted_for_sequence=True)
            else:
                if update.get("e") != "markPriceUpdate" or update.get("s") != symbol:
                    raise ValueError("unexpected USD-M Mark Price event identity")
                capture.frame(received_ns, monotonic_ns, raw, session_id=session_id,
                              event_time_ms=update.get("E"), mark_price=update.get("p"),
                              index_price=update.get("i"), funding_rate=update.get("r"),
                              next_funding_time_ms=update.get("T"))
        capture.record("session_disconnected", recorded_at_unix_ns=now_ns(), session_id=session_id,
                       reason="hourly_rotation")


async def run_capture(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    root = Path(args.hot_root)
    root.mkdir(parents=True, exist_ok=True)
    source_revision = os.environ.get("SOURCE_REVISION", "unreleased")
    capture = CaptureFile(root, args.kind, symbol, args.max_bytes, source_revision)
    stop_at = time.monotonic() + args.duration_seconds
    shutdown = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)
    retry = 1.0
    attempt = 0
    last_trade_id: list[int | None] = [None]
    stop_reason = "duration_elapsed"
    try:
        while time.monotonic() < stop_at and not shutdown.is_set():
            attempt += 1
            try:
                if args.kind in {"spot-depth", "usdm-depth"}:
                    await receive_depth_session(capture, args.kind, symbol, args.max_gap_ms, stop_at)
                else:
                    await receive_observation_session(capture, args.kind, symbol, args.max_gap_ms,
                                                      stop_at, last_trade_id)
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
            stop_reason = "sigterm" if not args.once else "sigint_or_sigterm"
    except CaptureLimit:
        stop_reason = "capture_byte_limit"
    finally:
        capture.close(stop_reason)
        print(f"capture stopped: {capture.path} records={capture.observations} gaps={capture.gaps} disconnects={capture.disconnects}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("spot-depth", "usdm-depth", "spot-trades", "usdm-funding-mark"))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--hot-root", default=os.environ.get("HOT_ROOT", "/var/lib/td-market-data/hot"))
    parser.add_argument("--max-gap-ms", type=int, default=5_000)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--duration-seconds", type=int, default=3_600)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_gap_ms <= 0 or args.max_bytes < 8192 or args.duration_seconds <= 0:
        raise SystemExit("max gap, capture size, and duration must be positive and bounded")
    asyncio.run(run_capture(args))


if __name__ == "__main__":
    main()
