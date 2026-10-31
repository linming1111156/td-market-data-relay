from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_spot_trades_dataset import SpotTradesRejected, publish


START_NS = 1_900_000_000_000_000_000


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def record(kind: str, **fields: object) -> dict[str, object]:
    return {"schema": "td_public_market_capture_record", "version": 1,
            "record_type": kind, **fields}


def make_capture(path: Path, *, gap: bool = False, tamper: bool = False) -> None:
    session = "spot-trade-session"
    rows: list[dict[str, object]] = [record(
        "capture_started", stream="spot-trades", symbol="BTCUSDT", max_gap_ms=30_000,
        recorded_at_unix_ns=START_NS, recorded_at_monotonic_ns=1_000_000_000,
        collector_executable_sha256="a" * 64, source_revision="fixture-revision")]
    rows.append(record("session_connected", session_id=session,
                       recorded_at_unix_ns=START_NS + 500_000_000,
                       url="wss://stream.binance.com:9443/ws/btcusdt@trade"))
    for index, (offset_ms, price) in enumerate(((2_500, "100"), (3_000, "101"),
                                                (4_000, "102"), (5_500, "103"))):
        trade_id = 10 + index + (1 if gap and index >= 1 else 0)
        frame = {"e": "trade", "E": 1_900_000_000_000 + offset_ms,
                 "s": "BTCUSDT", "t": trade_id, "p": price, "q": "0.01",
                 "T": 1_900_000_000_000 + offset_ms - 1, "m": index % 2 == 0, "M": True}
        raw_frame = compact(frame)
        receive_ns = START_NS + offset_ms * 1_000_000
        rows.append(record("market_frame", session_id=session,
                           received_at_unix_ns=receive_ns,
                           receive_monotonic_ns=2_000_000_000 + index * 1_000_000_000,
                           raw_frame=raw_frame,
                           raw_frame_sha256=("0" * 64 if tamper and index == 2
                                             else hashlib.sha256(raw_frame.encode()).hexdigest()),
                           event_time_ms=frame["E"], trade_time_ms=frame["T"], trade_id=trade_id,
                           accepted_for_sequence=True))
    rows.append(record("session_disconnected", session_id=session, reason="hourly_rotation",
                       recorded_at_unix_ns=START_NS + 5_800_000_000))
    before_stop = sum(len((compact(row) + "\n").encode()) for row in rows)
    rows.append(record("capture_stopped", reason="duration_elapsed", observation_count=4,
                       bytes_before_stop_record=before_stop,
                       recorded_at_unix_ns=START_NS + 6_000_000_000,
                       recorded_at_monotonic_ns=7_000_000_000))
    path.write_bytes(b"".join((compact(row) + "\n").encode() for row in rows))


class SpotTradesDatasetTests(unittest.TestCase):
    def test_publishes_windowed_trades_with_separate_times_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            manifest = publish(capture, output, "BTCUSDT", START_NS + 3_000_000_000,
                               START_NS + 5_000_000_000)
            events = [json.loads(line) for line in (output / "events.jsonl").read_bytes().splitlines()]
            self.assertEqual(manifest["event_count"], 2)
            self.assertEqual(manifest["first_trade_id"], 11)
            self.assertEqual(manifest["last_trade_id"], 12)
            self.assertEqual(events[0]["event_time_ns"], 1_900_000_003_000_000_000)
            self.assertEqual(events[0]["trade_time_ns"], 1_900_000_002_999_000_000)
            self.assertEqual(events[0]["receive_time_ns"], 1_900_000_003_000_000_000)
            self.assertEqual(events[0]["aggressor_side"], "BUY")
            self.assertEqual(manifest["source_capture_sha256"], hashlib.sha256(capture.read_bytes()).hexdigest())

    def test_rejects_trade_id_gap_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, gap=True)
            with self.assertRaises(SpotTradesRejected):
                publish(capture, output, "BTCUSDT", START_NS + 3_000_000_000,
                        START_NS + 5_000_000_000)
            self.assertFalse(output.exists())

    def test_rejects_raw_frame_hash_mismatch_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, tamper=True)
            with self.assertRaises(SpotTradesRejected):
                publish(capture, output, "BTCUSDT", START_NS + 3_000_000_000,
                        START_NS + 5_000_000_000)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
