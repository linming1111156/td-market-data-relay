import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_usdm_depth_dataset import CaptureRejected, publish_dataset


SYMBOL = "BTCUSDT"
URL = "wss://fstream.binance.com/public/ws/btcusdt@depth@100ms"
SESSION = "0123456789abcdef0123456789abcdef"
START_NS = 1_800_000_002_000_000_000
END_NS = 1_800_000_005_000_000_000


def encode(row):
    return (json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def make_capture(path, add_gap=False, *, add_unaccepted_stale=False,
                 receive_offsets_ns=None):
    raw_snapshot = json.dumps({"lastUpdateId": 100, "bids": [["50000.0", "1.0"]],
                               "asks": [["50001.0", "1.0"]]}, separators=(",", ":"))
    rows = [{"schema": "td_public_market_capture_record", "version": 1,
             "record_type": "capture_started", "recorded_at_unix_ns": 1_799_999_999_000_000_000,
             "recorded_at_monotonic_ns": 1_000_000_000, "stream": "usdm-depth",
             "symbol": SYMBOL, "source_host_id": "test-vps", "source_revision": "fixture-revision",
             "collector_executable_path": "/app/collector.py",
             "collector_executable_sha256": "a" * 64, "max_gap_ms": 5_000,
             "max_output_bytes": 256 * 1024 * 1024},
            {"schema": "td_public_market_capture_record", "version": 1,
             "record_type": "session_connected", "recorded_at_unix_ns": 1_800_000_000_000_000_000,
             "session_id": SESSION, "url": URL},
            {"schema": "td_public_market_capture_record", "version": 1,
             "record_type": "depth_snapshot", "received_at_unix_ns": 1_799_999_999_900_000_000,
             "snapshot_update_id": 100, "raw_snapshot": raw_snapshot,
             "raw_snapshot_sha256": hashlib.sha256(raw_snapshot.encode()).hexdigest(),
             "bids": [["50000.0", "1.0"]], "asks": [["50001.0", "1.0"]],
             "session_id": SESSION}]
    if add_unaccepted_stale:
        frame = {"e": "depthUpdate", "E": 1_799_999_999_900, "s": SYMBOL,
                 "U": 98, "u": 99, "pu": 97, "b": [["50000.0", "1.2"]], "a": []}
        raw = json.dumps(frame, separators=(",", ":"))
        rows.append({"schema": "td_public_market_capture_record", "version": 1,
                     "record_type": "market_frame", "session_id": SESSION,
                     "received_at_unix_ns": 1_800_000_000_100_000_000,
                     "receive_monotonic_ns": 1_900_000_000,
                     "raw_frame": raw, "raw_frame_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                     "event_time_ms": None, "first_update_id": 98, "final_update_id": 99,
                     "previous_final_update_id": 97, "accepted_for_book": False,
                     "reject_reason": "older_than_snapshot_or_duplicate"})
    changes = [
        (1_800_000_000_000, 100, 101, 99, [["50000.0", "0.9"]], [["50001.0", "0.8"]]),
        (1_800_000_001_000, 102, 102, 101, [["49999.0", "2.0"]], []),
        (1_800_000_002_000, 103, 103, 102, [["50000.0", "1.1"]], []),
    ]
    for i, (event_ms, first, final, previous, bids, asks) in enumerate(changes):
        frame = {"e": "depthUpdate", "E": event_ms, "s": SYMBOL, "U": first,
                 "u": final, "pu": previous, "b": bids, "a": asks}
        raw = json.dumps(frame, separators=(",", ":"))
        rows.append({"schema": "td_public_market_capture_record", "version": 1,
                     "record_type": "market_frame", "session_id": SESSION,
                     "received_at_unix_ns": event_ms * 1_000_000 + (
                         receive_offsets_ns[i] if receive_offsets_ns else 0),
                     "receive_monotonic_ns": 2_000_000_000 + i * 1_000_000_000,
                     "raw_frame": raw, "raw_frame_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                     "event_time_ms": event_ms, "first_update_id": first,
                     "final_update_id": final, "previous_final_update_id": previous,
                     "accepted_for_book": True})
    if add_gap:
        rows.append({"schema": "td_public_market_capture_record", "version": 1,
                     "record_type": "gap_detected", "reason": "fixture gap"})
    disconnect_ns = (1_800_000_004_050_000_000 if receive_offsets_ns
                     else 1_800_000_003_000_000_000)
    stop_ns = disconnect_ns + 100_000_000
    rows.append({"schema": "td_public_market_capture_record", "version": 1,
                 "record_type": "session_disconnected", "recorded_at_unix_ns": disconnect_ns,
                 "session_id": SESSION, "reason": "hourly_rotation"})
    before_stop = sum(len(encode(row)) for row in rows)
    rows.append({"schema": "td_public_market_capture_record", "version": 1,
                 "record_type": "capture_stopped", "recorded_at_unix_ns": stop_ns,
                 "recorded_at_monotonic_ns": 5_100_000_000 if receive_offsets_ns else 4_100_000_000,
                 "reason": "duration_elapsed",
                 "observation_count": 3 + int(add_unaccepted_stale),
                 "bytes_before_stop_record": before_stop})
    path.write_bytes(b"".join(encode(row) for row in rows))


class UsdmDepthDatasetExportTests(unittest.TestCase):
    def test_clean_capture_publishes_immutable_replay_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture.jsonl"
            output = root / "dataset"
            make_capture(capture)
            manifest = publish_dataset(capture, output, SYMBOL, START_NS, END_NS)
            self.assertTrue(manifest["replay_usable"])
            self.assertEqual(manifest["event_count"], 3)
            self.assertEqual(manifest["warmup_event_count"], 2)
            self.assertEqual(manifest["last_update_id"], 103)
            self.assertEqual(manifest["source_capture_sha256"], hashlib.sha256(capture.read_bytes()).hexdigest())
            self.assertEqual({path.name for path in output.iterdir()},
                             {"snapshot.json", "events.jsonl", "health.jsonl", "manifest.json"})

    def test_gap_capture_is_rejected_without_publishing_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture.jsonl"
            output = root / "dataset"
            make_capture(capture, add_gap=True)
            with self.assertRaises(CaptureRejected):
                publish_dataset(capture, output, SYMBOL, START_NS, END_NS)
            self.assertFalse(output.exists())

    def test_keeps_exchange_event_before_window_when_received_after_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "capture.jsonl"
            output = root / "dataset"
            make_capture(capture, add_unaccepted_stale=True,
                         receive_offsets_ns=(200_000_000, 2_000_000_000, 2_000_000_000))
            window_start = START_NS + 500_000_000
            manifest = publish_dataset(capture, output, SYMBOL, window_start, END_NS)
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["event_count"], 3)
            self.assertEqual(manifest["warmup_event_count"], 1)
            self.assertLess(events[1]["event_ts_ns"], window_start)
            self.assertGreater(events[1]["receive_ts_ns"], window_start)


if __name__ == "__main__":
    unittest.main()
