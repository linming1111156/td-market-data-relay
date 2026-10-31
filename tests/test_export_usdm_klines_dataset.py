import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_usdm_klines_dataset import KlinesRejected, publish
from verify_usdm_klines_dataset import DatasetRejected, verify_dataset


SYMBOL = "BTCUSDT"
BASE_MS = 1_800_000_000_000
URL = f"wss://fstream.binance.com/market/ws/{SYMBOL.lower()}@kline_1m"
BAR_MS = 60_000


def line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def make_capture(path, opens=None):
    if opens is None:
        opens = [BASE_MS - BAR_MS, BASE_MS, BASE_MS + BAR_MS, BASE_MS + 2 * BAR_MS]
    start_ns = opens[0] * 1_000_000 - 1_000_000_000
    records = [{"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "capture_started", "recorded_at_unix_ns": start_ns,
                "recorded_at_monotonic_ns": 1_000_000_000,
                "stream": "usdm-klines-1m", "symbol": SYMBOL, "source_url": URL,
                "source_host_id": "fixture-vps", "source_revision": "fixture-rev",
                "collector_executable_path": "/app/collector.py",
                "collector_executable_sha256": "a" * 64,
                "max_gap_ms": 120_000, "max_output_bytes": 256 * 1024 * 1024},
               {"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "session_connected", "recorded_at_unix_ns": start_ns + 1_000_000,
                "session_id": "session-1", "url": URL}]
    for index, open_ms in enumerate(opens):
        close_ms = open_ms + BAR_MS - 1
        event_ms = close_ms + 1
        kline = {"t": open_ms, "T": close_ms, "s": SYMBOL, "i": "1m", "f": index,
                 "L": index + 2, "o": "10.0", "c": "11.0", "h": "12.0", "l": "9.0",
                 "v": "5.0", "n": 3, "x": True, "q": "50.0", "V": "2.0", "Q": "20.0"}
        frame = {"e": "kline", "E": event_ms, "s": SYMBOL, "k": kline}
        raw = json.dumps(frame, separators=(",", ":"))
        receive_ns = event_ms * 1_000_000 + 1_000_000
        records.append({"schema": "td_public_market_capture_record", "version": 1,
                        "record_type": "market_frame", "session_id": "session-1",
                        "received_at_unix_ns": receive_ns,
                        "receive_monotonic_ns": 2_000_000_000 + index * BAR_MS * 1_000_000,
                        "event_time_ms": event_ms, "kline_open_time_ms": open_ms,
                        "kline_close_time_ms": close_ms, "kline_interval": "1m",
                        "kline_closed": True, "accepted_for_sequence": True,
                        "raw_frame": raw, "raw_frame_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    stopped_ns = (opens[-1] + BAR_MS + 2) * 1_000_000
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "session_disconnected", "recorded_at_unix_ns": stopped_ns - 1_000_000,
                    "session_id": "session-1", "reason": "hourly_rotation"})
    prior_bytes = sum(len(line(record)) for record in records)
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "capture_stopped", "recorded_at_unix_ns": stopped_ns,
                    "recorded_at_monotonic_ns": 2_000_000_000 + len(opens) * BAR_MS * 1_000_000,
                    "reason": "duration_elapsed", "observation_count": len(opens),
                    "gap_count": 0, "disconnect_count": 0,
                    "bytes_before_stop_record": prior_bytes})
    path.write_bytes(b"".join(line(record) for record in records))


class UsdmKlinesDatasetTests(unittest.TestCase):
    def test_publishes_exact_finalized_minute_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            manifest = publish([capture], output, SYMBOL,
                               BASE_MS * 1_000_000,
                               (BASE_MS + 2 * BAR_MS) * 1_000_000)
            rows = [json.loads(raw) for raw in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["event_count"], 2)
            self.assertEqual([row["open_time_ms"] for row in rows], [BASE_MS, BASE_MS + BAR_MS])
            self.assertTrue(all(row["available_at_ns"] > row["close_time_ms"] * 1_000_000 for row in rows))
            self.assertEqual(verify_dataset(output, [capture])["event_count"], 2)

    def test_missing_finalized_minute_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, [BASE_MS - BAR_MS, BASE_MS, BASE_MS + 2 * BAR_MS])
            with self.assertRaises(KlinesRejected):
                publish([capture], output, SYMBOL,
                        BASE_MS * 1_000_000,
                        (BASE_MS + 2 * BAR_MS) * 1_000_000)
            self.assertFalse(output.exists())

    def test_raw_frame_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            rows = [json.loads(raw) for raw in capture.read_text().splitlines()]
            rows[2]["raw_frame"] += " "
            capture.write_bytes(b"".join(line(row) for row in rows))
            with self.assertRaisesRegex(KlinesRejected, "raw frame hash"):
                publish([capture], output, SYMBOL,
                        BASE_MS * 1_000_000,
                        (BASE_MS + 2 * BAR_MS) * 1_000_000)

    def test_independent_verifier_rejects_dataset_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            publish([capture], output, SYMBOL,
                    BASE_MS * 1_000_000,
                    (BASE_MS + 2 * BAR_MS) * 1_000_000)
            records = [json.loads(raw) for raw in (output / "events.jsonl").read_text().splitlines()]
            records[0]["close"] = "99"
            (output / "events.jsonl").write_bytes(b"".join(line(row) for row in records))
            with self.assertRaises(DatasetRejected):
                verify_dataset(output)


if __name__ == "__main__":
    unittest.main()
