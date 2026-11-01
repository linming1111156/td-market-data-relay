import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from export_usdm_aggtrades_dataset import AggregateTradesRejected, publish
from verify_usdm_aggtrades_dataset import DatasetRejected, verify_dataset


SYMBOL = "BTCUSDT"
BASE_MS = 1_800_000_000_000
URL = f"wss://fstream.binance.com/market/ws/{SYMBOL.lower()}@aggTrade"


def line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def make_capture(path, ids=(100, 101, 102)):
    started_ns = BASE_MS * 1_000_000 - 2_000_000_000
    records = [{"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "capture_started", "recorded_at_unix_ns": started_ns,
                "recorded_at_monotonic_ns": 1_000_000_000,
                "stream": "usdm-aggtrades", "symbol": SYMBOL, "source_url": URL,
                "source_host_id": "fixture-vps", "source_revision": "fixture-rev",
                "collector_executable_path": "/app/collector.py",
                "collector_executable_sha256": "a" * 64,
                "max_gap_ms": 5_000, "max_output_bytes": 256 * 1024 * 1024},
               {"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "session_connected", "recorded_at_unix_ns": started_ns + 1_000_000,
                "session_id": "session-1", "url": URL}]
    for index, (trade_id, trade_ms) in enumerate(zip(ids, (BASE_MS, BASE_MS + 1_500, BASE_MS + 3_000), strict=True)):
        frame = {"e": "aggTrade", "E": trade_ms, "s": SYMBOL, "a": trade_id,
                 "p": "50000.25", "q": "0.003", "f": trade_id * 10,
                 "l": trade_id * 10, "T": trade_ms, "m": index % 2 == 0, "M": True}
        raw_frame = json.dumps(frame, separators=(",", ":"))
        received_ns = trade_ms * 1_000_000 + 100_000_000
        records.append({"schema": "td_public_market_capture_record", "version": 1,
                        "record_type": "market_frame", "session_id": "session-1",
                        "received_at_unix_ns": received_ns,
                        "receive_monotonic_ns": 2_000_000_000 + index * 1_000_000_000,
                        "event_time_ms": trade_ms, "trade_time_ms": trade_ms,
                        "trade_id": trade_id, "first_trade_id": trade_id * 10,
                        "last_trade_id": trade_id * 10, "buyer_is_maker": frame["m"],
                        "accepted_for_sequence": True, "raw_frame": raw_frame,
                        "raw_frame_sha256": hashlib.sha256(raw_frame.encode()).hexdigest()})
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "session_disconnected", "recorded_at_unix_ns": BASE_MS * 1_000_000 + 4_000_000_000,
                    "session_id": "session-1", "reason": "hourly_rotation"})
    prior_bytes = sum(len(line(record)) for record in records)
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "capture_stopped", "recorded_at_unix_ns": BASE_MS * 1_000_000 + 4_100_000_000,
                    "recorded_at_monotonic_ns": 6_000_000_000, "reason": "duration_elapsed",
                    "observation_count": 3, "gap_count": 0, "disconnect_count": 0,
                    "bytes_before_stop_record": prior_bytes})
    path.write_bytes(b"".join(line(record) for record in records))


class UsdmAggregateTradesDatasetTests(unittest.TestCase):
    def test_publishes_point_in_time_trade_window_with_aggressor_side(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            manifest = publish([capture], output, SYMBOL,
                               (BASE_MS + 1_000) * 1_000_000,
                               (BASE_MS + 2_000) * 1_000_000)
            rows = [json.loads(row) for row in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["event_count"], 1)
            self.assertEqual(rows[0]["aggregate_trade_id"], 101)
            self.assertEqual(rows[0]["aggressor_side"], "BUY")
            self.assertGreater(rows[0]["received_at_ns"], rows[0]["trade_time_ns"])
            self.assertEqual(verify_dataset(output, [capture])["event_count"], 1)

    def test_id_gap_is_rejected_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture, ids=(100, 104, 105))
            with self.assertRaisesRegex(AggregateTradesRejected, "ID gap"):
                publish([capture], output, SYMBOL,
                        (BASE_MS + 1_000) * 1_000_000,
                        (BASE_MS + 2_000) * 1_000_000)
            self.assertFalse(output.exists())

    def test_independent_verifier_rejects_event_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            publish([capture], output, SYMBOL,
                    (BASE_MS + 1_000) * 1_000_000,
                    (BASE_MS + 2_000) * 1_000_000)
            record = json.loads((output / "events.jsonl").read_text())
            record["price"] = "1"
            (output / "events.jsonl").write_text(json.dumps(record, separators=(",", ":")) + "\n")
            with self.assertRaises(DatasetRejected):
                verify_dataset(output)

    def test_raw_frame_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            records = [json.loads(raw) for raw in capture.read_text().splitlines()]
            records[2]["raw_frame"] += " "
            capture.write_bytes(b"".join(line(record) for record in records))
            with self.assertRaisesRegex(AggregateTradesRejected, "hash mismatch"):
                publish([capture], output, SYMBOL,
                        (BASE_MS + 1_000) * 1_000_000,
                        (BASE_MS + 2_000) * 1_000_000)

    def test_recovered_capture_preserves_receive_time_and_delays_availability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture, output = root / "capture.jsonl", root / "dataset"
            make_capture(capture)
            raw_hash = hashlib.sha256(capture.read_bytes()).hexdigest()
            recovery_ns = (BASE_MS + 3_000) * 1_000_000
            (root / "capture.jsonl.recovery.json").write_text(json.dumps({
                "schema": "td_market_data_relay_recovery_provenance_v1", "version": 1,
                "capture_id": raw_hash[:32], "origin": "vps_recovered",
                "recovered_at_unix_ns": recovery_ns, "raw_sha256": raw_hash,
                "relay_host": "relay.example",
            }))
            publish([capture], output, SYMBOL,
                    (BASE_MS + 1_000) * 1_000_000,
                    (BASE_MS + 2_000) * 1_000_000)
            event = json.loads((output / "events.jsonl").read_text().splitlines()[0])
            self.assertLess(event["received_at_ns"], event["available_at_ns"])
            self.assertEqual(event["available_at_ns"], recovery_ns)
            verify_dataset(output, [capture])


if __name__ == "__main__":
    unittest.main()
