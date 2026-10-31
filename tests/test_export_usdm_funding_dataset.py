import hashlib
import json
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from export_usdm_funding_dataset import FundingDatasetRejected, publish
from verify_usdm_funding_dataset import DatasetRejected, verify_dataset


SYMBOL = "BTCUSDT"
FUNDING_MS = 1_800_000_000_000
FUNDING_NS = FUNDING_MS * 1_000_000


def line(record):
    return json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def write_capture(path, records):
    path.write_bytes(b"".join(line(record) for record in records))


def header(stream, source_url, started_ns, max_gap_ms):
    return {"schema": "td_public_market_capture_record", "version": 1,
            "record_type": "capture_started", "recorded_at_unix_ns": started_ns,
            "recorded_at_monotonic_ns": 1_000_000_000, "stream": stream,
            "symbol": SYMBOL, "source_host_id": "fixture-vps", "source_revision": "fixture-rev",
            "collector_executable_path": "/app/collector.py",
            "collector_executable_sha256": "a" * 64, "source_url": source_url,
            "max_gap_ms": max_gap_ms, "max_output_bytes": 256 * 1024 * 1024}


def make_mark_capture(path, start_ns, end_ns):
    url = f"wss://fstream.binance.com/market/ws/{SYMBOL.lower()}@markPrice@1s"
    records = [header("usdm-funding-mark", url, start_ns - 1_000_000, 5_000),
               {"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "session_connected", "recorded_at_unix_ns": start_ns - 1_000_000,
                "session_id": "mark-session", "url": url}]
    start_ms, end_ms = start_ns // 1_000_000, end_ns // 1_000_000
    previous_mono = 1_000_000_000
    count = 0
    for received_ms in range(start_ms, end_ms + 1_000, 1_000):
        next_time = FUNDING_MS if received_ms < FUNDING_MS else FUNDING_MS + 28_800_000
        raw = json.dumps({"e": "markPriceUpdate", "E": received_ms, "s": SYMBOL,
                          "p": "50000.0", "i": "50001.0", "r": "0.0001",
                          "T": next_time}, separators=(",", ":"))
        received_ns = received_ms * 1_000_000
        records.append({"schema": "td_public_market_capture_record", "version": 1,
                        "record_type": "market_frame", "session_id": "mark-session",
                        "received_at_unix_ns": received_ns,
                        "receive_monotonic_ns": previous_mono,
                        "event_time_ms": received_ms, "next_funding_time_ms": next_time,
                        "raw_frame": raw, "raw_frame_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        previous_mono += 1_000_000_000
        count += 1
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "session_disconnected", "recorded_at_unix_ns": end_ns + 2_000_000,
                    "session_id": "mark-session", "reason": "hourly_rotation"})
    prior_size = sum(map(len, (line(row) for row in records)))
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "capture_stopped", "recorded_at_unix_ns": end_ns + 3_000_000,
                    "recorded_at_monotonic_ns": previous_mono, "reason": "duration_elapsed",
                    "observation_count": count, "gap_count": 0, "disconnect_count": 0,
                    "bytes_before_stop_record": prior_size})
    write_capture(path, records)


def make_funding_capture(path, start_ns, available_ns, include_settlement=True):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    records = [header("usdm-finalized-funding", url, start_ns - 1_000_000, 120_000),
               {"schema": "td_public_market_capture_record", "version": 1,
                "record_type": "funding_cursor_started", "recorded_at_unix_ns": start_ns,
                "request_start_time_ms": start_ns // 1_000_000, "future_only": True}]
    values = ([{"symbol": SYMBOL, "fundingTime": FUNDING_MS, "fundingRate": "0.0001",
                "markPrice": "50000.0", "rateType": "Regular"}] if include_settlement else [])
    raw = json.dumps(values, separators=(",", ":"))
    request_url = ("https://fapi.binance.com/fapi/v1/fundingRate?"
                   + urllib.parse.urlencode({"symbol": SYMBOL, "startTime": start_ns // 1_000_000,
                                             "limit": 1000}))
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "funding_api_page", "request_url": request_url,
                    "request_start_time_ms": start_ns // 1_000_000,
                    "received_at_unix_ns": available_ns, "receive_monotonic_ns": 3_000_000_000,
                    "raw_response": raw, "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "record_count": len(values), "funding_rows": values})
    prior_size = sum(map(len, (line(row) for row in records)))
    records.append({"schema": "td_public_market_capture_record", "version": 1,
                    "record_type": "capture_stopped", "recorded_at_unix_ns": available_ns + 1_000_000,
                    "recorded_at_monotonic_ns": 4_000_000_000, "reason": "duration_elapsed",
                    "observation_count": 0, "gap_count": 0, "disconnect_count": 0,
                    "bytes_before_stop_record": prior_size})
    write_capture(path, records)


class UsdmFundingDatasetTests(unittest.TestCase):
    def test_matches_finalized_funding_to_fresh_pre_settlement_mark_and_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_ns = FUNDING_NS - 10_000_000_000
            end_ns = FUNDING_NS + 3_000_000_000
            mark = root / "mark.jsonl"
            funding = root / "funding.jsonl"
            make_mark_capture(mark, start_ns, end_ns)
            make_funding_capture(funding, start_ns, FUNDING_NS + 1_000_000_000)
            output = root / "dataset"
            manifest = publish([mark], [funding], output, SYMBOL, start_ns, end_ns)
            self.assertEqual(manifest["event_count"], 1)
            event = json.loads((output / "events.jsonl").read_text())
            self.assertEqual(event["funding_time_ns"], FUNDING_NS)
            self.assertGreater(event["available_at_ns"], event["funding_time_ns"])
            self.assertEqual(manifest["expected_settlement_times_ms"], [FUNDING_MS])
            self.assertTrue(manifest["logical_fingerprint_sha256"])
            verified = verify_dataset(output, [mark, funding])
            self.assertEqual(verified["event_count"], 1)

    def test_missing_final_settlement_is_rejected_without_publishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_ns = FUNDING_NS - 10_000_000_000
            end_ns = FUNDING_NS + 3_000_000_000
            mark, funding = root / "mark.jsonl", root / "funding.jsonl"
            make_mark_capture(mark, start_ns, end_ns)
            make_funding_capture(funding, start_ns, FUNDING_NS + 1_000_000_000, False)
            output = root / "dataset"
            with self.assertRaisesRegex(FundingDatasetRejected, "does not exactly match"):
                publish([mark], [funding], output, SYMBOL, start_ns, end_ns)
            self.assertFalse(output.exists())

    def test_stale_mark_evidence_cannot_define_expected_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_ns = FUNDING_NS - 10_000_000_000
            end_ns = FUNDING_NS + 3_000_000_000
            mark, funding = root / "mark.jsonl", root / "funding.jsonl"
            make_mark_capture(mark, start_ns, end_ns)
            make_funding_capture(funding, start_ns, FUNDING_NS + 1_000_000_000)
            records = [json.loads(raw) for raw in mark.read_text().splitlines()]
            for record in records:
                if record.get("record_type") == "market_frame" and record.get("next_funding_time_ms") == FUNDING_MS:
                    record["received_at_unix_ns"] = FUNDING_NS - 10_000_000_000
                    frame = json.loads(record["raw_frame"])
                    frame["E"] = record["received_at_unix_ns"] // 1_000_000
                    record["raw_frame"] = json.dumps(frame, separators=(",", ":"))
                    record["raw_frame_sha256"] = hashlib.sha256(record["raw_frame"].encode()).hexdigest()
            # Keep the fixture's capture bytes/stop summary coherent after modifying records.
            footer = records[-1]
            footer["bytes_before_stop_record"] = sum(len(line(row)) for row in records[:-1])
            write_capture(mark, records)
            with self.assertRaises(FundingDatasetRejected):
                publish([mark], [funding], root / "dataset", SYMBOL, start_ns, end_ns)

    def test_independent_verifier_rejects_event_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start_ns = FUNDING_NS - 10_000_000_000
            end_ns = FUNDING_NS + 3_000_000_000
            mark, funding, output = root / "mark.jsonl", root / "funding.jsonl", root / "dataset"
            make_mark_capture(mark, start_ns, end_ns)
            make_funding_capture(funding, start_ns, FUNDING_NS + 1_000_000_000)
            publish([mark], [funding], output, SYMBOL, start_ns, end_ns)
            event = json.loads((output / "events.jsonl").read_text())
            event["funding_rate"] = "-0.5"
            (output / "events.jsonl").write_text(json.dumps(event, separators=(",", ":")) + "\n")
            with self.assertRaises(DatasetRejected):
                verify_dataset(output)


if __name__ == "__main__":
    unittest.main()
