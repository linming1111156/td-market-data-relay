from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from export_spot_trades_dataset import publish
from test_export_spot_trades_dataset import START_NS, make_capture
from verify_spot_trades_dataset import verify
from export_spot_trades_dataset import SpotTradesRejected


class SpotTradesDatasetVerifierTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        capture, dataset = root / "capture.jsonl", root / "dataset"
        make_capture(capture)
        publish(capture, dataset, "BTCUSDT", START_NS + 3_000_000_000,
                START_NS + 5_000_000_000)
        return dataset

    def test_reverifies_published_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            result = verify(dataset)
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["event_count"], 2)

    def test_rejects_tampered_event_file_and_logical_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            events = dataset / "events.jsonl"
            events.write_bytes(events.read_bytes() + b"{}\n")
            with self.assertRaises(SpotTradesRejected):
                verify(dataset)

        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            manifest_path = dataset / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["logical_fingerprint_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(SpotTradesRejected):
                verify(dataset)


if __name__ == "__main__":
    unittest.main()
