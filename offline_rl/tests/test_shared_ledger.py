import json
import tempfile
import unittest
from pathlib import Path

from td_market_data_rl.contracts import MINUTE_NS
from td_market_data_rl.learner import simulate_episode


class SharedLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tdrl-ledger-")
        self.root = Path(self.temp.name)
        self.features = self.root / "features.jsonl"
        self.ledger = self.root / "ledger.jsonl"
        with self.features.open("w") as feature_file, self.ledger.open("w") as ledger_file:
            for minute in range(16):
                timestamp = minute * MINUTE_NS
                feature_file.write(json.dumps({
                    "decision_at_ns": timestamp, "decision_due": minute % 15 == 0,
                    "feature_ready": True, "mark_price": 60_000.0,
                    "features": {"mark_log_return_15m": 0.0},
                }) + "\n")
                settlements = ([{"funding_time_ns": timestamp, "available_at_ns": timestamp,
                                 "mark_price": 60_000.0, "funding_rate": 0.001}]
                               if minute == 1 else [])
                ledger_file.write(json.dumps({
                    "minute_open_ns": timestamp, "execution_window_end_ns": timestamp + MINUTE_NS,
                    "execution_vwap": 60_000.0, "execution_base_volume": 1.0,
                    "funding_settlements": settlements,
                }) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def run_policy(self, action):
        return simulate_episode(self.features, self.ledger,
                                {"start_ns": 0, "end_ns": 16 * MINUTE_NS},
                                lambda feature, position: (action(feature, position), None),
                                fee_bps=10, slippage_bps=5, memory_limit_mb=4096)[1]

    def test_noop_has_zero_cost_and_funding(self):
        stats = self.run_policy(lambda _feature, _position: 1)
        self.assertEqual(stats["fills"], 0)
        self.assertEqual(stats["funding_pnl_usdt"], 0.0)
        self.assertEqual(stats["ending_equity_usdt"], 10_000.0)

    def test_buy_and_hold_uses_same_fee_slippage_and_funding_ledger(self):
        stats = self.run_policy(lambda _feature, _position: 2)
        self.assertEqual(stats["fills"], 1)
        self.assertAlmostEqual(stats["fees_usdt"], 0.06)
        self.assertAlmostEqual(stats["slippage_usdt"], 0.03)
        self.assertAlmostEqual(stats["funding_pnl_usdt"], -0.06)
        self.assertAlmostEqual(stats["ending_equity_usdt"], 9_999.85)


if __name__ == "__main__":
    unittest.main()
