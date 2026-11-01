import json
import tempfile
import unittest
from pathlib import Path

from td_market_data_rl.contracts import FEATURE_NAMES
from td_market_data_rl.learner import fit_normalizer


class TrainOnlyNormalizerTests(unittest.TestCase):
    def test_validation_values_do_not_change_train_statistics(self):
        with tempfile.TemporaryDirectory(prefix="tdrl-normalizer-") as directory:
            path = Path(directory) / "features.jsonl"
            rows = [
                {"decision_at_ns": 0, "feature_ready": True,
                 "features": {name: 1.0 for name in FEATURE_NAMES}},
                {"decision_at_ns": 60, "feature_ready": True,
                 "features": {name: 3.0 for name in FEATURE_NAMES}},
                {"decision_at_ns": 120, "feature_ready": True,
                 "features": {name: 1_000_000.0 for name in FEATURE_NAMES}},
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            means, scales, count = fit_normalizer(path, 0, 120)
        self.assertEqual(count, 2)
        self.assertEqual(means, [2.0] * len(FEATURE_NAMES))
        self.assertEqual(scales, [1.0] * len(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
