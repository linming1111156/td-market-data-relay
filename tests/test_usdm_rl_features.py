import json
import unittest
from pathlib import Path

from build_usdm_rl_features import MINUTE_NS, mark_price_at


class UsdmRlFeatureTests(unittest.TestCase):
    def test_mark_endpoint_is_asof_and_rejects_stale_or_future_data(self):
        values = {
            9: (9 * MINUTE_NS + 59_000_000_000, 10 * MINUTE_NS, 100.0),
            10: (10 * MINUTE_NS + 10_000_000_000, 10 * MINUTE_NS + 10_000_000_000, 101.0),
        }
        self.assertEqual(mark_price_at(values, 10 * MINUTE_NS, 11 * MINUTE_NS), 100.0)
        self.assertIsNone(mark_price_at(values, 10 * MINUTE_NS, 9 * MINUTE_NS))
        self.assertIsNone(mark_price_at({}, 10 * MINUTE_NS, 11 * MINUTE_NS))

    def test_feature_protocol_freezes_factor_order_and_separates_execution(self):
        protocol = json.loads(Path("offline-usdm-feature-ledger-v1.json").read_text())
        self.assertEqual(protocol["policy_state"]["action_targets_btc"], [-0.001, 0.0, 0.001])
        self.assertEqual(protocol["execution"]["participation_cap_fraction_of_next_minute_base_volume"], 0.01)
        self.assertEqual(protocol["cost_scenarios"]["reference"], {
            "taker_fee_bps": 10, "adverse_slippage_bps": 5})
        self.assertEqual(len(protocol["features"]), 7)


if __name__ == "__main__":
    unittest.main()
