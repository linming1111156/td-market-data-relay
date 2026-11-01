import unittest

from td_market_data_rl.learner import apply_fill


class FillLedgerTests(unittest.TestCase):
    def test_entry_respects_entry_cap_and_applies_costs(self):
        ledger = {"execution_vwap": 60_000.0, "execution_base_volume": 1.0}
        position, cost, fill, lots = apply_fill(0, 1, ledger, fee_bps=10, slippage_bps=5)
        self.assertEqual(position, 1)
        self.assertEqual(lots, 1)
        self.assertAlmostEqual(fill, 60_030.0)
        self.assertAlmostEqual(cost, 0.09)

    def test_entry_above_notional_cap_is_rejected(self):
        ledger = {"execution_vwap": 100_001.0, "execution_base_volume": 1.0}
        self.assertEqual(apply_fill(0, 1, ledger, 10, 5), (0, 0.0, 0.0, 0))

    def test_missing_vwap_never_fills(self):
        ledger = {"execution_vwap": None, "execution_base_volume": 5.0}
        self.assertEqual(apply_fill(0, 1, ledger, 10, 5), (0, 0.0, 0.0, 0))

    def test_reducing_position_is_not_subject_to_entry_minimum(self):
        ledger = {"execution_vwap": 40_000.0, "execution_base_volume": 0.2}
        position, cost, _, lots = apply_fill(1, 0, ledger, fee_bps=10, slippage_bps=5)
        self.assertEqual((position, lots), (0, -1))
        self.assertGreater(cost, 0.0)

    def test_fill_does_not_flip_through_flat_in_one_execution(self):
        ledger = {"execution_vwap": 60_000.0, "execution_base_volume": 1.0}
        position, _, _, lots = apply_fill(1, -1, ledger, fee_bps=10, slippage_bps=5)
        self.assertEqual((position, lots), (0, -1))


if __name__ == "__main__":
    unittest.main()
