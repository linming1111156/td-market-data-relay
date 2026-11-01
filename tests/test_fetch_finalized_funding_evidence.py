import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_finalized_funding_evidence import FundingEvidenceRejected, write_evidence


class Response:
    status = 200

    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _limit):
        return self.raw


class FetchFinalizedFundingEvidenceTests(unittest.TestCase):
    def schedule(self):
        settled_ns = (time.time_ns() // 1_000_000) * 1_000_000
        return {"schema": "td_gateway_usdm_funding_settlement_schedule", "version": 1,
                "symbol": "BTCUSDT", "capture_stopped_at_unix_ns": settled_ns,
                "expected_settlement_times_ns": [settled_ns]}

    def test_fetches_exact_settlement_and_publishes_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = root / "schedule.json"
            schedule.write_text(json.dumps(self.schedule()))
            funding_time_ms = json.loads(schedule.read_text())["expected_settlement_times_ns"][0] // 1_000_000
            raw = json.dumps([{"symbol": "BTCUSDT", "fundingTime": funding_time_ms,
                           "fundingRate": "0.0001", "markPrice": "50000.25"}]).encode()
            output = root / "evidence"
            with patch("fetch_finalized_funding_evidence.urllib.request.urlopen",
                       return_value=Response(raw)) as urlopen:
                manifest = write_evidence(schedule, output)
            self.assertIn(f"startTime={funding_time_ms}", manifest["request_url"])
            self.assertIn(f"endTime={funding_time_ms}", manifest["request_url"])
            self.assertEqual(manifest["http_status"], 200)
            self.assertEqual(manifest["response_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(manifest["response_bytes"], len(raw))
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual((output / "funding-response.json").read_bytes(), raw)
            self.assertEqual(json.loads((output / "funding-source-manifest.json").read_text()), manifest)
            urlopen.assert_called_once()

    def test_rejects_missing_or_wrong_scheduled_record_without_output(self):
        raw = b"[]"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = root / "schedule.json"
            value = self.schedule()
            schedule.write_text(json.dumps(value))
            output = root / "evidence"
            with patch("fetch_finalized_funding_evidence.urllib.request.urlopen",
                       return_value=Response(raw)):
                with self.assertRaisesRegex(FundingEvidenceRejected, "exactly the scheduled"):
                    write_evidence(schedule, output)
            self.assertFalse(output.exists())

    def test_rejects_more_than_one_scheduled_settlement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schedule = root / "schedule.json"
            value = self.schedule()
            settled_ns = value["expected_settlement_times_ns"][0]
            value["capture_stopped_at_unix_ns"] = settled_ns + 1_000_000_000
            value["expected_settlement_times_ns"].append(settled_ns + 1_000_000_000)
            schedule.write_text(json.dumps(value))
            with patch("fetch_finalized_funding_evidence.urllib.request.urlopen") as urlopen:
                with self.assertRaisesRegex(FundingEvidenceRejected, "requires one"):
                    write_evidence(schedule, root / "evidence")
            urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
