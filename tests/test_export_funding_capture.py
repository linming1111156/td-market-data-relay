import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_funding_capture import InvalidCapture, export_capture


SYMBOL = "BTCUSDT"
URL = "wss://fstream.binance.com/market/ws/btcusdt@markPrice@1s"


def record_bytes(value):
    return (json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def fixture_records():
    raw_frames = [
        json.dumps({"e": "markPriceUpdate", "E": 1_800_000_000_000, "s": SYMBOL,
                    "p": "50000.0", "i": "50001.0", "P": "50002.0",
                    "r": "0.0001", "T": 1_800_028_800_000}, separators=(",", ":")),
        json.dumps({"e": "markPriceUpdate", "E": 1_800_000_001_000, "s": SYMBOL,
                    "p": "50001.0", "i": "50002.0", "P": "50003.0",
                    "r": "0.0001", "T": 1_800_028_800_000}, separators=(",", ":")),
    ]
    records = [{
        "schema": "td_public_market_capture_record", "version": 1,
        "record_type": "capture_started", "recorded_at_unix_ns": 1_800_000_000_000_000_000,
        "recorded_at_monotonic_ns": 10_000, "stream": "usdm-funding-mark", "symbol": SYMBOL,
        "source_host_id": "test-host", "source_revision": "abc123",
        "source_url": URL, "collector_executable_path": "/app/collector.py",
        "collector_executable_sha256": "a" * 64, "max_gap_ms": 5_000,
        "max_output_bytes": 256 * 1024 * 1024,
    }, {
        "schema": "td_public_market_capture_record", "version": 1,
        "record_type": "session_connected", "recorded_at_unix_ns": 1_800_000_000_000_000_000,
        "session_id": "session-1", "url": URL,
    }]
    for i, raw in enumerate(raw_frames):
        data = json.loads(raw)
        records.append({
            "schema": "td_public_market_capture_record", "version": 1,
            "record_type": "market_frame", "session_id": "session-1",
            "received_at_unix_ns": 1_800_000_000_000_000_000 + i * 1_000_000_000,
            "receive_monotonic_ns": 20_000 + i * 1_000_000_000,
            "event_time_ms": data["E"], "raw_frame": raw,
            "raw_frame_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "mark_price": data["p"], "index_price": data["i"],
            "funding_rate": data["r"], "next_funding_time_ms": data["T"],
        })
    records.append({
        "schema": "td_public_market_capture_record", "version": 1,
        "record_type": "session_disconnected", "recorded_at_unix_ns": 1_800_000_002_000_000_000,
        "session_id": "session-1", "reason": "hourly_rotation",
    })
    prior_bytes = sum(len(record_bytes(record)) for record in records)
    records.append({
        "schema": "td_public_market_capture_record", "version": 1,
        "record_type": "capture_stopped", "recorded_at_unix_ns": 1_800_000_002_100_000_000,
        "recorded_at_monotonic_ns": 2_100_010_000, "reason": "duration_elapsed",
        "observation_count": 2, "bytes_before_stop_record": prior_bytes,
    })
    return records


def write_fixture(path, records):
    path.write_bytes(b"".join(record_bytes(record) for record in records))


class FundingCaptureExportTests(unittest.TestCase):
    def test_clean_capture_exports_rust_compatible_lifecycle_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "validated.jsonl"
            write_fixture(source, fixture_records())
            digest, count = export_capture(source, output, SYMBOL, None)
            self.assertEqual(count, 2)
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), digest)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["record_type"], "capture_started")
            self.assertEqual(rows[0]["source_url"], URL)
            self.assertEqual(rows[0]["source_capture_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual([row["record_type"] for row in rows].count("session_connected"), 1)
            self.assertEqual([row["record_type"] for row in rows].count("mark_price_funding_observation"), 2)
            self.assertEqual(rows[-2]["reason"], "local_shutdown")
            self.assertEqual(rows[-1]["bytes_written"], len(b"".join(record_bytes(row) for row in rows[:-1])))

    def test_gap_record_is_rejected_and_no_output_is_left(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "validated.jsonl"
            records = fixture_records()
            records.insert(-2, {
                "schema": "td_public_market_capture_record", "version": 1,
                "record_type": "gap_detected", "reason": "fixture gap",
            })
            write_fixture(source, records)
            with self.assertRaises(InvalidCapture):
                export_capture(source, output, SYMBOL, None)
            self.assertFalse(output.exists())

    def test_raw_frame_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            output = Path(directory) / "validated.jsonl"
            records = fixture_records()
            records[2]["raw_frame_sha256"] = "0" * 64
            write_fixture(source, records)
            with self.assertRaises(InvalidCapture):
                export_capture(source, output, SYMBOL, None)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
