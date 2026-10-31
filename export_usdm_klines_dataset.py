#!/usr/bin/env python3
"""Publish complete, finalized USD-M one-minute candles from raw captures."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from export_spot_depth_dataset import canonical, file_sha256, source_revision


MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_DATASET_BYTES = 128 * 1024 * 1024
MAX_BARS = 1_000_000
MAX_LINE_BYTES = 2 * 1024 * 1024
BAR_MS = 60_000


class KlinesRejected(ValueError):
    pass


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None or renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def parse_time(value: str) -> int:
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("timestamp must be RFC3339 UTC") from error
    if stamp.tzinfo != dt.timezone.utc:
        raise argparse.ArgumentTypeError("timestamp must use UTC")
    return int(stamp.timestamp()) * 1_000_000_000 + stamp.microsecond * 1_000


def uint(value: Any, field: str, line: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise KlinesRejected(f"line {line}: {field} must be a nonnegative integer")
    return value


def dec(value: Any, field: str, line: int, *, positive: bool = False) -> tuple[str, Decimal]:
    if not isinstance(value, str):
        raise KlinesRejected(f"line {line}: {field} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise KlinesRejected(f"line {line}: invalid {field}") from error
    if not parsed.is_finite() or (positive and parsed <= 0) or (not positive and parsed < 0):
        raise KlinesRejected(f"line {line}: invalid {field}")
    return value, parsed


def publish(captures: list[Path], output: Path, symbol: str,
            start_ns: int, end_ns: int, market: str = "USD_M") -> dict[str, Any]:
    if not captures or not output.is_absolute() or start_ns >= end_ns:
        raise KlinesRejected("supply captures, an absolute output, and a positive window")
    if market not in {"SPOT", "USD_M"}:
        raise KlinesRejected("market must be SPOT or USD_M")
    if (start_ns % (BAR_MS * 1_000_000) or end_ns % (BAR_MS * 1_000_000)
            or not symbol.isascii() or not symbol.isalnum()):
        raise KlinesRejected("window must be minute-aligned and symbol alphanumeric")
    resolved = [path.resolve(strict=True) for path in captures]
    if len(set(resolved)) != len(resolved):
        raise KlinesRejected("duplicate capture paths are not allowed")
    for path in resolved:
        if path.stat().st_size > MAX_CAPTURE_BYTES:
            raise KlinesRejected(f"capture exceeds 256 MiB: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    repository = Path(__file__).resolve().parent
    if output.exists() or output == repository or repository in output.parents:
        raise KlinesRejected("output exists or is inside the source repository")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        return _publish(resolved, output, parent, stage, symbol.upper(), start_ns, end_ns, market)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _publish(captures: list[Path], output: Path, parent: Path, stage: Path,
             symbol: str, start_ns: int, end_ns: int, market: str) -> dict[str, Any]:
    start_ms, end_ms = start_ns // 1_000_000, end_ns // 1_000_000
    expected_bars = (end_ms - start_ms) // BAR_MS
    if expected_bars <= 0 or expected_bars > MAX_BARS:
        raise KlinesRejected("window bar count exceeds configured bound")
    logical = hashlib.sha256(f"td-market-data-{market.lower()}-klines-1m-v1\0".encode() + canonical({
        "symbol": symbol, "start_ns": start_ns, "end_ns": end_ns}))
    events_hash = hashlib.sha256()
    event_count = event_bytes = frame_count = 0
    previous_received_ns = previous_mono_ns = previous_event_ms = None
    previous_open_ms: int | None = None
    previous_closed_open_ms: int | None = None
    previous_inside_receive_ns: int | None = None
    first_inside: dict[str, Any] | None = None
    last_inside: dict[str, Any] | None = None
    pre_evidence: dict[str, Any] | None = None
    post_evidence: dict[str, Any] | None = None
    max_chain_gap_ms = max_dataset_gap_ms = 0
    sources: list[dict[str, Any]] = []
    chain_start_ns = chain_stop_ns = last_capture_start_ns = None
    capture_stream_name = "spot-klines-1m" if market == "SPOT" else "usdm-klines-1m"
    source_url = (f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_1m" if market == "SPOT"
                  else f"wss://fstream.binance.com/market/ws/{symbol.lower()}@kline_1m")

    with (stage / "events.jsonl").open("xb", buffering=0) as target:
        for capture in captures:
            source_hash = file_sha256(capture)
            start: dict[str, Any] | None = None
            footer: dict[str, Any] | None = None
            session: str | None = None
            connected = disconnected = False
            connect_ns = disconnect_ns = None
            max_gap_ms = 0
            local_frames = 0
            bytes_seen = 0
            with capture.open("rb") as stream:
                for line_number, raw in enumerate(stream, 1):
                    before_record = bytes_seen
                    bytes_seen += len(raw)
                    if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n"):
                        raise KlinesRejected(f"{capture}:{line_number}: truncated/oversized capture line")
                    try:
                        row = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise KlinesRejected(f"{capture}:{line_number}: invalid capture JSON") from error
                    if (not isinstance(row, dict) or row.get("schema") != "td_public_market_capture_record"
                            or row.get("version") != 1):
                        raise KlinesRejected(f"{capture}:{line_number}: unsupported capture schema")
                    kind = row.get("record_type")
                    if line_number == 1:
                        if (kind != "capture_started" or row.get("stream") != capture_stream_name
                                or row.get("symbol") != symbol or row.get("source_url") != source_url):
                            raise KlinesRejected(f"capture identity/source mismatch: {capture}")
                        start = row
                        started_ns = uint(row.get("recorded_at_unix_ns"), "capture start", line_number)
                        max_gap_ms = uint(row.get("max_gap_ms"), "receive gap policy", line_number)
                        code_hash, revision = row.get("collector_executable_sha256"), row.get("source_revision")
                        if max_gap_ms <= 0 or not isinstance(revision, str) or not revision:
                            raise KlinesRejected(f"capture source policy/revision invalid: {capture}")
                        if (not isinstance(code_hash, str) or len(code_hash) != 64
                                or any(char not in "0123456789abcdef" for char in code_hash)):
                            raise KlinesRejected(f"capture code hash invalid: {capture}")
                        if last_capture_start_ns is not None and started_ns <= last_capture_start_ns:
                            raise KlinesRejected("capture files must be supplied in chronological order")
                        if chain_stop_ns is not None and started_ns < chain_stop_ns:
                            raise KlinesRejected("capture files overlap in their lifecycle")
                        if chain_start_ns is None:
                            chain_start_ns = started_ns
                        last_capture_start_ns = started_ns
                    elif kind == "session_connected":
                        if connected or disconnected:
                            raise KlinesRejected(f"{capture}:{line_number}: reconnect is not admissible")
                        session = row.get("session_id")
                        connect_ns = uint(row.get("recorded_at_unix_ns"), "connect time", line_number)
                        if not isinstance(session, str) or not session or row.get("url") != start["source_url"]:
                            raise KlinesRejected(f"{capture}:{line_number}: session identity mismatch")
                        connected = True
                    elif kind == "market_frame":
                        local_frames += 1
                        frame_count += 1
                        if not connected or disconnected or row.get("session_id") != session:
                            raise KlinesRejected(f"{capture}:{line_number}: frame outside connected session")
                        raw_frame = row.get("raw_frame")
                        if (not isinstance(raw_frame, str)
                                or hashlib.sha256(raw_frame.encode()).hexdigest() != row.get("raw_frame_sha256")):
                            raise KlinesRejected(f"{capture}:{line_number}: raw frame hash mismatch")
                        try:
                            frame = json.loads(raw_frame)
                        except json.JSONDecodeError as error:
                            raise KlinesRejected(f"{capture}:{line_number}: invalid raw frame") from error
                        bar = frame.get("k") if isinstance(frame, dict) else None
                        if (not isinstance(frame, dict) or frame.get("e") != "kline" or frame.get("s") != symbol
                                or not isinstance(bar, dict) or bar.get("s") != symbol or bar.get("i") != "1m"):
                            raise KlinesRejected(f"{capture}:{line_number}: exchange event identity mismatch")
                        event_ms = uint(frame.get("E"), "exchange event time", line_number)
                        open_ms, close_ms = uint(bar.get("t"), "kline open time", line_number), uint(bar.get("T"), "kline close time", line_number)
                        received_ns = uint(row.get("received_at_unix_ns"), "receive time", line_number)
                        mono_ns = uint(row.get("receive_monotonic_ns"), "monotonic receive time", line_number)
                        is_closed = bar.get("x")
                        if (not isinstance(is_closed, bool) or close_ms != open_ms + BAR_MS - 1
                                or row.get("event_time_ms") != event_ms or row.get("kline_open_time_ms") != open_ms
                                or row.get("kline_close_time_ms") != close_ms or row.get("kline_interval") != "1m"
                                or row.get("kline_closed") is not is_closed or row.get("accepted_for_sequence") is not True):
                            raise KlinesRejected(f"{capture}:{line_number}: parsed fields disagree with raw frame")
                        if received_ns < (connect_ns or 0) or event_ms * 1_000_000 > received_ns + 2_000_000_000:
                            raise KlinesRejected(f"{capture}:{line_number}: invalid receive-time ordering")
                        if previous_open_ms is not None and open_ms < previous_open_ms:
                            raise KlinesRejected(f"{capture}:{line_number}: kline open time regressed")
                        if previous_event_ms is not None and event_ms < previous_event_ms:
                            raise KlinesRejected(f"{capture}:{line_number}: exchange event time regressed")
                        if previous_received_ns is not None:
                            if received_ns <= previous_received_ns or mono_ns <= previous_mono_ns:
                                raise KlinesRejected(f"{capture}:{line_number}: receive clock regressed")
                            wall_delta, mono_delta = received_ns - previous_received_ns, mono_ns - previous_mono_ns
                            max_chain_gap_ms = max(max_chain_gap_ms, wall_delta // 1_000_000)
                            if mono_delta > max_gap_ms * 1_000_000 or abs(wall_delta - mono_delta) > 2_000_000_000:
                                raise KlinesRejected(f"{capture}:{line_number}: receive gap/clock discontinuity")
                        if is_closed:
                            if previous_closed_open_ms is not None and open_ms != previous_closed_open_ms + BAR_MS:
                                raise KlinesRejected(f"{capture}:{line_number}: finalized minute gap")
                            if received_ns < (close_ms + 1) * 1_000_000:
                                raise KlinesRejected(f"{capture}:{line_number}: finalized candle available before close")
                            values = {key: dec(bar.get(key), key, line_number,
                                               positive=key in {"o", "c", "h", "l"})[0]
                                      for key in ("o", "h", "l", "c", "v", "q", "V", "Q")}
                            nums = {key: Decimal(value) for key, value in values.items()}
                            if (nums["h"] < max(nums["o"], nums["c"], nums["l"])
                                    or nums["l"] > min(nums["o"], nums["c"], nums["h"])
                                    or nums["V"] > nums["v"] or nums["Q"] > nums["q"]):
                                raise KlinesRejected(f"{capture}:{line_number}: OHLC/volume constraints invalid")
                            trade_count = uint(bar.get("n"), "kline trade count", line_number)
                            first_trade_id = bar.get("f")
                            last_trade_id = bar.get("L")
                            if (not isinstance(first_trade_id, int) or isinstance(first_trade_id, bool)
                                    or not isinstance(last_trade_id, int) or isinstance(last_trade_id, bool)
                                    or first_trade_id < -1 or last_trade_id < -1):
                                raise KlinesRejected(f"{capture}:{line_number}: kline trade-ID range invalid")
                            normalized = {"record_type": "finalized_kline", "symbol": symbol,
                                          "interval": "1m", "open_time_ms": open_ms,
                                          "close_time_ms": close_ms, "exchange_event_time_ms": event_ms,
                                          "available_at_ns": received_ns,
                                          "open": values["o"], "high": values["h"], "low": values["l"],
                                          "close": values["c"], "base_volume": values["v"],
                                          "quote_volume": values["q"], "taker_buy_base_volume": values["V"],
                                          "taker_buy_quote_volume": values["Q"], "trade_count": trade_count,
                                          "first_trade_id": first_trade_id, "last_trade_id": last_trade_id,
                                          "raw_frame_sha256": row["raw_frame_sha256"]}
                            if open_ms < start_ms:
                                pre_evidence = pre_evidence or normalized
                            elif open_ms >= end_ms:
                                post_evidence = post_evidence or normalized
                            else:
                                if previous_inside_receive_ns is not None:
                                    max_dataset_gap_ms = max(max_dataset_gap_ms,
                                        (received_ns - previous_inside_receive_ns) // 1_000_000)
                                previous_inside_receive_ns = received_ns
                                encoded = canonical(normalized) + b"\n"
                                event_bytes += len(encoded)
                                if event_bytes > MAX_DATASET_BYTES:
                                    raise KlinesRejected("normalized Dataset exceeds 128 MiB")
                                target.write(encoded)
                                events_hash.update(encoded)
                                logical.update(encoded)
                                first_inside = first_inside or normalized
                                last_inside = normalized
                                event_count += 1
                            previous_closed_open_ms = open_ms
                        previous_open_ms, previous_event_ms = open_ms, event_ms
                        previous_received_ns, previous_mono_ns = received_ns, mono_ns
                    elif kind == "session_disconnected":
                        if (not connected or disconnected or row.get("session_id") != session
                                or row.get("reason") not in {"hourly_rotation", "local_shutdown"}):
                            raise KlinesRejected(f"{capture}:{line_number}: unclean session stop")
                        disconnect_ns = uint(row.get("recorded_at_unix_ns"), "disconnect time", line_number)
                        if previous_received_ns is not None and disconnect_ns < previous_received_ns:
                            raise KlinesRejected(f"{capture}:{line_number}: disconnect precedes last frame")
                        disconnected = True
                    elif kind == "capture_stopped":
                        footer = row
                        stopped_ns = uint(row.get("recorded_at_unix_ns"), "capture stop", line_number)
                        if (not disconnected or row.get("reason") not in {"duration_elapsed", "sigterm"}
                                or row.get("observation_count") != local_frames or row.get("gap_count") != 0
                                or row.get("disconnect_count") != 0 or row.get("bytes_before_stop_record") != before_record
                                or stopped_ns < (disconnect_ns or 0)):
                            raise KlinesRejected(f"{capture}:{line_number}: capture footer mismatch")
                        if chain_stop_ns is not None and started_ns - chain_stop_ns > max_gap_ms * 1_000_000:
                            raise KlinesRejected("receive gap across capture rotation")
                        chain_stop_ns = stopped_ns
                    else:
                        raise KlinesRejected(f"{capture}:{line_number}: inadmissible record {kind!r}")
            if not (start and footer and connected and disconnected and local_frames > 0):
                raise KlinesRejected(f"capture incomplete or empty: {capture}")
            if file_sha256(capture) != source_hash:
                raise KlinesRejected(f"capture changed during export: {capture}")
            sources.append({"path": str(capture), "sha256": source_hash,
                            "capture_started_at_ns": uint(start["recorded_at_unix_ns"], "capture start", 1),
                            "capture_stopped_at_ns": uint(footer["recorded_at_unix_ns"], "capture stop", 1),
                            "collector_code_sha256": start["collector_executable_sha256"],
                            "collector_source_revision": start["source_revision"],
                            "observation_count": local_frames})
        target.flush()
        os.fsync(target.fileno())
    if event_count != expected_bars or pre_evidence is None or post_evidence is None:
        raise KlinesRejected("requested window lacks exact finalized-minute coverage or boundary candles")
    if chain_start_ns is None or chain_stop_ns is None or chain_start_ns > start_ns or chain_stop_ns < end_ns:
        raise KlinesRejected("capture lifecycle does not cover requested window")
    builder = source_revision()
    manifest = {"schema": f"td_market_data_{market.lower()}_klines_dataset", "version": 1,
                "exchange": "BINANCE", "market": market, "symbol": symbol, "interval": "1m",
                "window": {"start_ns": start_ns, "end_ns": end_ns}, "event_count": event_count,
                "first_open_time_ms": first_inside["open_time_ms"],
                "last_open_time_ms": last_inside["open_time_ms"],
                "first_available_at_ns": first_inside["available_at_ns"],
                "last_available_at_ns": last_inside["available_at_ns"],
                "max_receive_gap_ms": max_dataset_gap_ms,
                "max_capture_chain_receive_gap_ms": max_chain_gap_ms,
                "boundary_coverage": {"pre_window_candle": {"open_time_ms": pre_evidence["open_time_ms"],
                                                                "available_at_ns": pre_evidence["available_at_ns"],
                                                                "raw_frame_sha256": pre_evidence["raw_frame_sha256"]},
                                      "post_window_candle": {"open_time_ms": post_evidence["open_time_ms"],
                                                                 "available_at_ns": post_evidence["available_at_ns"],
                                                                 "raw_frame_sha256": post_evidence["raw_frame_sha256"]}},
                "events_file": "events.jsonl", "events_sha256": events_hash.hexdigest(),
                "logical_fingerprint_sha256": logical.hexdigest(), "capture_sources": sources,
                "source_capture_set_sha256": hashlib.sha256(canonical(
                    [item["sha256"] for item in sources])).hexdigest(),
                "collector_code_sha256": sorted({item["collector_code_sha256"] for item in sources}),
                "converter_code_sha256": file_sha256(Path(__file__).resolve()),
                "builder_source": builder, "replay_usable": builder["tree_sha256"] is not None}
    with (stage / "manifest.json").open("xb") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    for source in sources:
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise KlinesRejected(f"source capture changed during publication: {source['path']}")
    stage_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(stage_fd)
    finally:
        os.close(stage_fd)
    rename_noreplace(stage, output)
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return manifest


def main(default_market: str = "USD_M") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="append", required=True, type=Path)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--market", choices=("SPOT", "USD_M"), default=default_market)
    parser.add_argument("--start", required=True, type=parse_time)
    parser.add_argument("--end", required=True, type=parse_time)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = publish(args.capture, args.output, args.symbol.upper(), args.start, args.end, args.market)
    except (KlinesRejected, OSError, ValueError) as error:
        raise SystemExit(f"{args.market} Kline Dataset rejected: {error}") from error
    print(f"{args.market} finalized Kline Dataset published: bars={manifest['event_count']} "
          f"logical_fingerprint={manifest['logical_fingerprint_sha256']}")


if __name__ == "__main__":
    main()
