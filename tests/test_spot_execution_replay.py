from __future__ import annotations

import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from export_spot_depth_dataset import publish
from spot_execution_replay import ReplayRejected, replay, write_create_only
from test_export_spot_depth_dataset import make_capture


class SpotExecutionReplayTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> Path:
        capture, dataset = root / "capture.jsonl", root / "dataset"
        make_capture(capture)
        publish(capture, dataset, "BTCUSDT", 3_000_000_000, 5_000_000_000)
        return dataset

    def test_point_in_time_partial_fill_fees_and_common_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            config = root / "config.json"
            orders = root / "orders.json"
            config.write_text(json.dumps({
                "schema": "td_spot_execution_replay_config", "version": 1,
                "initial_quote_balance": "1000", "initial_base_balance": "0",
                "taker_fee_bps": "10", "latency_ms": 500,
            }))
            orders.write_text(json.dumps([{"order_id": "take-1", "type": "MARKET_IOC",
                                           "side": "BUY", "quantity": "2.5",
                                           "decision_ts_ns": 3_500_000_000}]))
            report = replay(dataset, config, orders)
            self.assertEqual(report, replay(dataset, config, orders))
            strategy = report["accounts"]["strategy"]
            self.assertEqual(strategy["orders"][0]["status"], "PARTIALLY_FILLED")
            self.assertEqual(Decimal(strategy["orders"][0]["filled_quantity"]), Decimal("1"))
            self.assertEqual(Decimal(strategy["total_fees_quote"]), Decimal("0.102"))
            self.assertEqual(Decimal(strategy["turnover_quote"]), Decimal("102"))
            self.assertEqual(strategy["fills"][0]["price"], "102")
            self.assertEqual(Decimal(report["accounts"]["NOOP"]["net_pnl_quote"]), Decimal("0"))
            self.assertEqual(Decimal(report["accounts"]["BUY_AND_HOLD"]["total_fees_quote"]), Decimal("0.202"))
            self.assertEqual(report["dataset_logical_fingerprint_sha256"],
                             json.loads((dataset / "manifest.json").read_bytes())["logical_fingerprint_sha256"])
            self.assertEqual(len(report["replay_code_sha256"]), 64)
            self.assertEqual(len(report["config_sha256"]), 64)
            self.assertEqual(len(report["orders_sha256"]), 64)
            output = root / "report.json"
            write_create_only(output, report)
            with self.assertRaises(ReplayRejected):
                write_create_only(output, report)

    def test_rejects_tampered_dataset_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = self.make_dataset(root)
            with (dataset / "events.jsonl").open("ab") as stream:
                stream.write(b"{}\n")
            config, orders = root / "config.json", root / "orders.json"
            config.write_text(json.dumps({"schema": "td_spot_execution_replay_config", "version": 1,
                                          "initial_quote_balance": "100", "initial_base_balance": "0",
                                          "taker_fee_bps": "10", "latency_ms": 0}))
            orders.write_text("[]")
            with self.assertRaises(ReplayRejected):
                replay(dataset, config, orders)


if __name__ == "__main__":
    unittest.main()
