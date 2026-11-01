import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from run_verified_usdm_replay import ReplayBundleRejected, run, verify_input_compatibility


REPORT = {"schema": "td_gateway_futures_depth_comparative_replay_v2",
          "version": 2, "candidate": {"execution": {"symbol": "BTCUSDT"}},
          "comparison_fingerprint_sha256": "a" * 64}
WINDOW = {"start_ns": 1_800_000_000_000_000_000,
          "end_ns": 1_800_000_060_000_000_000}


def manifest_for(name):
    if name == "depth":
        return {"schema": "td_gateway_usdm_depth_dataset", "market": "USD_M",
                "contract": "PERPETUAL", "symbol": "BTCUSDT", "window": WINDOW,
                "replay_usable": True}
    if name == "mark":
        return {"schema": "td_gateway_futures_supplemental_dataset", "market": "USD_M",
                "contract": "PERPETUAL", "kind": "mark_price_observation", "symbol": "BTCUSDT",
                "window": WINDOW, "replay_usable": True, "coverage_known": True}
    return {"schema": "td_gateway_futures_supplemental_dataset", "market": "USD_M",
            "contract": "PERPETUAL", "kind": "funding_rate", "symbol": "BTCUSDT",
            "window": WINDOW, "settlement_usable": True}


class VerifiedReplayRunnerTests(unittest.TestCase):
    def test_rejects_misaligned_dataset_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in ("depth", "mark", "funding"):
                dataset = root / name
                dataset.mkdir()
                manifest = manifest_for(name)
                if name == "funding":
                    manifest["window"] = {**WINDOW, "end_ns": WINDOW["end_ns"] + 1}
                (dataset / "manifest.json").write_text(json.dumps(manifest))
                paths.append(dataset)
            with self.assertRaisesRegex(ReplayBundleRejected, "windows differ"):
                verify_input_compatibility(*paths)

    def test_runs_and_repeats_the_exact_read_only_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = []
            for name in ("depth", "mark", "funding"):
                dataset = root / name
                dataset.mkdir()
                (dataset / "manifest.json").write_text(json.dumps(manifest_for(name)))
                datasets.append(dataset)
            config = root / "config.json"
            orders = root / "orders.json"
            config.write_text("{}")
            orders.write_text("[]")
            calls = root / "calls.jsonl"
            executable = root / "fake-replay.py"
            executable.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                f"calls = pathlib.Path({str(calls)!r})\n"
                "calls.open('a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                "args = sys.argv[1:]\n"
                "if '--output' in args:\n"
                f" pathlib.Path(args[args.index('--output') + 1]).write_text(json.dumps({REPORT!r}))\n"
                "else:\n"
                " assert '--verify-report' in args\n"
                " assert json.loads(pathlib.Path(args[args.index('--verify-report') + 1]).read_text())['comparison_fingerprint_sha256'] == 'a' * 64\n"
            )
            executable.chmod(0o755)
            report, evidence = root / "report.json", root / "evidence.json"
            result = run(Namespace(replay_executable=executable, depth=datasets[0], mark=datasets[1],
                                   funding=datasets[2], config=config, orders=orders,
                                   selection_manifest=None, report=report, evidence=evidence))
            self.assertEqual(len(calls.read_text().splitlines()), 2)
            self.assertEqual(result["symbol"], "BTCUSDT")
            self.assertTrue(result["deterministic_rerun"]["passed"])
            self.assertEqual(json.loads(evidence.read_text())["report"]["sha256"],
                             result["report"]["sha256"])

    def test_rejects_report_path_inside_an_input_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            depth = root / "depth"
            depth.mkdir()
            (depth / "manifest.json").write_text(json.dumps(manifest_for("depth")))
            other = root / "other"
            other.mkdir()
            for name in ("manifest.json",):
                (other / name).write_text(json.dumps(manifest_for("mark")))
            funding = root / "funding"
            funding.mkdir()
            (funding / "manifest.json").write_text(json.dumps(manifest_for("funding")))
            executable = root / "unused"
            executable.write_text("#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            config, orders = root / "config.json", root / "orders.json"
            config.write_text("{}")
            orders.write_text("[]")
            with self.assertRaisesRegex(ReplayBundleRejected, "outside immutable input"):
                run(Namespace(replay_executable=executable, depth=depth, mark=other, funding=funding,
                              config=config, orders=orders, selection_manifest=None,
                              report=depth / "report.json", evidence=root / "evidence.json"))


if __name__ == "__main__":
    unittest.main()
