import unittest

from td_market_data_rl.cli import read_role_plan_from_object
from td_market_data_rl.contracts import MINUTE_NS


class RolePlanTests(unittest.TestCase):
    def setUp(self):
        self.start = (1_760_000_000_000_000_000 // MINUTE_NS) * MINUTE_NS
        self.end = self.start + 120 * MINUTE_NS
        self.features = {"window": {"start_ns": self.start, "end_ns": self.end}}

    def plan(self, validation_start=None):
        split = self.start + 90 * MINUTE_NS if validation_start is None else validation_start
        return {"windows": {
            "train": {"start_ns": self.start, "end_ns": split},
            "validation": {"start_ns": split, "end_ns": self.end},
        }}

    def test_contiguous_chronological_roles_cover_the_dataset(self):
        read_role_plan_from_object(self.plan(), self.features, {})

    def test_gap_between_roles_is_rejected(self):
        plan = self.plan()
        plan["windows"]["train"]["end_ns"] -= MINUTE_NS
        with self.assertRaisesRegex(ValueError, "cover the fixed Dataset"):
            read_role_plan_from_object(plan, self.features, {})

    def test_train_longer_than_one_year_is_rejected(self):
        features = {"window": {"start_ns": self.start, "end_ns": self.start + 600_000 * MINUTE_NS}}
        plan = {"windows": {
            "train": {"start_ns": self.start, "end_ns": self.start + 525_601 * MINUTE_NS},
            "validation": {"start_ns": self.start + 525_601 * MINUTE_NS,
                           "end_ns": features["window"]["end_ns"]},
        }}
        with self.assertRaisesRegex(ValueError, "one-year cap"):
            read_role_plan_from_object(plan, features, {})


if __name__ == "__main__":
    unittest.main()
