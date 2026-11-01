from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_spot_depth_dataset import CaptureRejected, publish


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def record(kind: str, **fields: object) -> dict[str, object]:
    return {"schema": "td_public_market_capture_record", "version": 1,
            "record_type": kind, **fields}


def make_capture(path: Path, *, gap: bool = False,
                 receive_offsets_ns: tuple[int, ...] | None = None) -> None:
    session = "spot-session-1"
    rows: list[dict[str, object]] = [record(
        "capture_started", stream="spot-depth", symbol="BTCUSDT", max_gap_ms=30_000,
        recorded_at_unix_ns=1_000_000_000, recorded_at_monotonic_ns=100,
        collector_executable_sha256="a" * 64, source_revision="test")]
    rows.append(record("session_connected", session_id=session,
                       url="wss://stream.binance.com:9443/ws/btcusdt@depth@100ms"))
    snapshot = {"lastUpdateId": 100, "bids": [["99", "1"]], "asks": [["101", "1"]]}
    raw_snapshot = compact(snapshot)
    rows.append(record("depth_snapshot", session_id=session, received_at_unix_ns=2_000_000_000,
                       raw_snapshot=raw_snapshot,
                       raw_snapshot_sha256=hashlib.sha256(raw_snapshot.encode()).hexdigest(),
                       bids=snapshot["bids"], asks=snapshot["asks"]))
    updates = [
        (101 if not gap else 102, 101 if not gap else 102, 2_500, [["99", "2"]], []),
        (102 if not gap else 103, 102 if not gap else 103, 3_000, [], [["101", "2"]]),
        (103 if not gap else 104, 103 if not gap else 104, 4_000, [], [["101", "0"], ["102", "1"]]),
        (104 if not gap else 105, 104 if not gap else 105, 5_500, [], [["101", "3"]]),
    ]
    for index, (first, final, event_ms, bids, asks) in enumerate(updates):
        frame = {"e": "depthUpdate", "s": "BTCUSDT", "E": event_ms,
                 "U": first, "u": final, "b": bids, "a": asks}
        raw_frame = compact(frame)
        received_ns = event_ms * 1_000_000 + (receive_offsets_ns[index] if receive_offsets_ns else 0)
        rows.append(record("market_frame", session_id=session,
                           received_at_unix_ns=received_ns,
                           receive_monotonic_ns=1_000_000_000 + index * 1_000_000_000,
                           raw_frame=raw_frame,
                           raw_frame_sha256=hashlib.sha256(raw_frame.encode()).hexdigest(),
                           event_time_ms=event_ms, first_update_id=first,
                           final_update_id=final, accepted_for_book=True))
    rows.extend([
        record("session_disconnected", session_id=session, reason="hourly_rotation"),
        record("capture_stopped", reason="duration_elapsed", observation_count=4,
               recorded_at_unix_ns=6_000_000_000, recorded_at_monotonic_ns=4_000_000_000),
    ])
    path.write_bytes(b"".join((compact(row) + "\n").encode() for row in rows))


class SpotDepthDatasetTests(unittest.TestCase):
    def test_publishes_windowed_dataset_and_rolls_snapshot_forward(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            manifest = publish(capture, output, "BTCUSDT", 3_000_000_000, 5_000_000_000)
            snapshot = json.loads((output / "snapshot.json").read_bytes())
            events = [json.loads(line) for line in (output / "events.jsonl").read_bytes().splitlines()]
            self.assertEqual(manifest["event_count"], 2)
            self.assertEqual(snapshot["last_update_id"], 101)
            self.assertEqual(snapshot["bids"], [["99", "2"]])
            self.assertEqual(events[0]["event_ts_ns"], 3_000_000_000)
            self.assertEqual(events[0]["receive_ts_ns"], 3_000_000_000)
            self.assertEqual(manifest["source_capture_sha256"], hashlib.sha256(capture.read_bytes()).hexdigest())

    def test_keeps_pre_window_exchange_event_that_arrives_after_window_opens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, receive_offsets_ns=(600_000_000, 300_000_000,
                                                       200_000_000, 100_000_000))
            manifest = publish(capture, output, "BTCUSDT", 3_000_000_000, 5_000_000_000)
            snapshot = json.loads((output / "snapshot.json").read_bytes())
            events = [json.loads(line) for line in (output / "events.jsonl").read_bytes().splitlines()]
            self.assertEqual(snapshot["last_update_id"], 100)
            self.assertEqual(manifest["event_count"], 3)
            self.assertEqual(events[0]["event_ts_ns"], 2_500_000_000)
            self.assertEqual(events[0]["receive_ts_ns"], 3_100_000_000)

    def test_rejects_sequence_gap_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, gap=True)
            with self.assertRaises(CaptureRejected):
                publish(capture, output, "BTCUSDT", 3_000_000_000, 5_000_000_000)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
