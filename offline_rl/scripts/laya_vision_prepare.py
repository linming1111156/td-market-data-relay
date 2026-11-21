#!/usr/bin/env python3
"""Build time-separated Laya examples from audited Binance Vision features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import zstandard

from laya_market_continuous import QUESTIONS, QUESTION_ID


ROOT = Path(__file__).resolve().parents[2]
FEATURE_SCHEMA = "td_market_data_usdm_rl_historical_feature_set"
ROW_SCHEMA = "td_market_data_usdm_rl_historical_feature"
FEATURE_NAMES = (
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
)
MINUTE_MS = 60_000
HORIZON_MINUTES = 15
DECISION_INTERVAL_MINUTES = 15
FEE_BPS_EACH_SIDE = 10.0
SLIPPAGE_BPS_EACH_SIDE = 5.0
ROUND_TRIP_COST_BPS = 2 * (FEE_BPS_EACH_SIDE + SLIPPAGE_BPS_EACH_SIDE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_feature_set(root: Path) -> tuple[dict[str, Any], str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"feature set must be a real directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"feature manifest is missing or unsafe: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema") != FEATURE_SCHEMA or manifest.get("version") != 1
            or manifest.get("symbol") != "BTCUSDT" or manifest.get("market") != "USD_M"
            or manifest.get("feature_schema") != "td_futures_research_features"
            or manifest.get("feature_schema_version") != 1
            or manifest.get("feature_names") != list(FEATURE_NAMES)
            or manifest.get("time_basis") != "historical Binance Vision event time; receive/availability time is unknown"):
        raise ValueError(f"unsupported or non-Vision feature set: {root}")
    months = manifest.get("months")
    if not isinstance(months, list) or not months:
        raise ValueError(f"feature manifest has no month shards: {root}")
    for month in months:
        path = root / str(month.get("output_path", ""))
        if path.parent != root or path.is_symlink() or not path.is_file():
            raise ValueError(f"feature shard is missing or unsafe: {path}")
        if (path.stat().st_size != month.get("output_size_bytes")
                or sha256_file(path) != month.get("output_sha256")):
            raise ValueError(f"feature shard hash mismatch: {path}")
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def read_rows(feature_sets: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for feature_root in feature_sets:
        root = feature_root.expanduser().resolve(strict=True)
        manifest, manifest_hash = validate_feature_set(root)
        sources.append({"path": str(root), "manifest_sha256": manifest_hash,
                        "source_manifest_sha256": manifest["source_manifest_sha256"],
                        "months": [{"month": item["month"],
                                    "output_path": item["output_path"],
                                    "output_sha256": item["output_sha256"],
                                    "input_sha256": item["input_sha256"],
                                    "minute_rows": item["minute_rows"],
                                    "feature_ready_rows": item["feature_ready_rows"]}
                                   for item in manifest["months"]]})
        for month in manifest["months"]:
            path = root / month["output_path"]
            with zstandard.open(path, "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from error
                    if (row.get("schema") != ROW_SCHEMA or row.get("version") != 1
                            or row.get("symbol") != "BTCUSDT"):
                        raise ValueError(f"{path}:{line_number}: unsupported feature row")
                    rows.append(row)
    rows.sort(key=lambda row: row["decision_time_ms"])
    timestamps = [row["decision_time_ms"] for row in rows]
    if any(not isinstance(ts, int) for ts in timestamps):
        raise ValueError("feature decision timestamps must be integers")
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("feature sets overlap or contain duplicate decision timestamps")
    return rows, sources


def laya_row(row: dict[str, Any], future_price: float) -> tuple[dict[str, Any] | None, str | None]:
    if row.get("decision_due") is not True:
        return None, "not_a_decision_time"
    if row.get("feature_ready") is not True or row.get("aggtrade_history_complete") is not True:
        return None, "feature_or_trade_history_incomplete"
    features = row.get("features")
    mark = row.get("mark_price")
    if not isinstance(features, dict) or not isinstance(mark, (int, float)) or mark <= 0:
        return None, "missing_decision_features_or_mark"
    if any(not isinstance(features.get(name), (int, float))
           or not math.isfinite(float(features[name])) for name in FEATURE_NAMES):
        return None, "invalid_feature_value"
    forward_bps = math.log(future_price / float(mark)) * 10_000.0
    label = ("long" if forward_bps > ROUND_TRIP_COST_BPS else
             "short" if forward_bps < -ROUND_TRIP_COST_BPS else "flat")
    market_features = {
        "mark_return_1m_bps": round(float(features["mark_log_return_1m"]) * 10_000.0, 6),
        "mark_return_5m_bps": round(float(features["mark_log_return_5m"]) * 10_000.0, 6),
        "mark_return_15m_bps": round(float(features["mark_log_return_15m"]) * 10_000.0, 6),
        "mark_realized_volatility_15m_bps": round(float(features["mark_realized_volatility_15m"]) * 10_000.0, 6),
        "mark_realized_volatility_60m_bps": round(float(features["mark_realized_volatility_60m"]) * 10_000.0, 6),
        "aggtrade_signed_volume_imbalance_1m": round(float(features["aggtrade_signed_volume_imbalance_1m"]), 6),
        "aggtrade_signed_volume_imbalance_5m": round(float(features["aggtrade_signed_volume_imbalance_5m"]), 6),
    }
    timestamp = row["decision_time_ms"]
    state = {
        "market": "BTCUSDT USD-M perpetual",
        "decision_interval_minutes": DECISION_INTERVAL_MINUTES,
        "market_features": market_features,
        "mark_price_usdt": round(float(mark), 4),
        "portfolio": {"current_position_lots": 0, "current_position_btc": 0.0},
        "risk_constraints": {
            "maximum_absolute_position_btc": 0.001,
            "entry_order_btc": 0.001,
            "minimum_notional_usdt": 50,
            "maximum_entry_notional_usdt": 100,
            "taker_fee_bps_each_side": FEE_BPS_EACH_SIDE,
            "adverse_slippage_bps_each_side": SLIPPAGE_BPS_EACH_SIDE,
            "funding_is_applied_from_the_historical_settlement_ledger": False,
        },
        "available_targets": ["short", "flat", "long"],
    }
    return ({"decision_time_unix_ms": timestamp,
             "outcome": {"forward_mark_return_bps": round(forward_bps, 6),
                         "round_trip_cost_bps": ROUND_TRIP_COST_BPS},
             "state": state,
             "questions": QUESTIONS,
             "expected": {QUESTION_ID: label}}, None)


def build_examples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_time = {row["decision_time_ms"]: row for row in rows}
    output: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    horizon_ms = HORIZON_MINUTES * MINUTE_MS
    for row in rows:
        if row.get("decision_due") is not True:
            continue
        future = by_time.get(row["decision_time_ms"] + horizon_ms)
        if future is None:
            skipped["missing_forward_horizon"] += 1
            continue
        future_mark = future.get("mark_price")
        if not isinstance(future_mark, (int, float)) or not math.isfinite(float(future_mark)) or future_mark <= 0:
            skipped["missing_forward_mark"] += 1
            continue
        example, reason = laya_row(row, float(future_mark))
        if reason:
            skipped[reason] += 1
        elif example:
            output.append(example)
    return output, dict(sorted(skipped.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", action="append", type=Path, required=True,
                        help="audited Rust Vision feature set; repeat for multiple training months")
    parser.add_argument("--evaluation-features", action="append", type=Path, required=True,
                        help="audited Rust Vision feature set; repeat for multiple held-out months")
    parser.add_argument("--output", type=Path, required=True,
                        help="new persistent output directory outside the repository")
    args = parser.parse_args()
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya datasets must be prepared inside Kaggle; the local machine is code-only")
    output = args.output.expanduser().absolute()
    resolved_output = output.resolve()
    kaggle_working = Path("/kaggle/working").resolve()
    if (output.exists() or output.is_symlink() or resolved_output == ROOT
            or ROOT in resolved_output.parents
            or (resolved_output != kaggle_working and kaggle_working not in resolved_output.parents)
            or Path("/tmp") == resolved_output or Path("/tmp") in resolved_output.parents
            or Path("/var/tmp") == resolved_output or Path("/var/tmp") in resolved_output.parents):
        raise ValueError("output must be new and under /kaggle/working")
    train_rows, train_sources = read_rows(args.train_features)
    eval_rows, eval_sources = read_rows(args.evaluation_features)
    if not train_rows or not eval_rows:
        raise ValueError("training and evaluation feature sets must both be non-empty")
    train_first, train_last = train_rows[0]["decision_time_ms"], train_rows[-1]["decision_time_ms"]
    eval_first, eval_last = eval_rows[0]["decision_time_ms"], eval_rows[-1]["decision_time_ms"]
    if train_last + HORIZON_MINUTES * MINUTE_MS >= eval_first:
        raise ValueError("evaluation time must begin after the training window and label horizon")
    train_examples, train_skipped = build_examples(train_rows)
    eval_examples, eval_skipped = build_examples(eval_rows)
    if not train_examples or not eval_examples:
        raise ValueError("chronological source windows produced an empty Laya split")
    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", train_examples)
    write_jsonl(output / "evaluation.jsonl", eval_examples)
    label_counts = lambda rows: dict(sorted(Counter(row["expected"][QUESTION_ID] for row in rows).items()))
    report = {
        "schema": "td_market_data_laya_vision_prepare_report", "version": 1,
        "symbol": "BTCUSDT", "market": "USD_M",
        "time_basis": "historical Binance Vision exchange event time; receive/availability time is unknown",
        "decision_interval_minutes": DECISION_INTERVAL_MINUTES,
        "label_horizon_minutes": HORIZON_MINUTES,
        "label_threshold_round_trip_bps": ROUND_TRIP_COST_BPS,
        "fee_bps_each_side": FEE_BPS_EACH_SIDE,
        "slippage_bps_each_side": SLIPPAGE_BPS_EACH_SIDE,
        "train_window_ms": {"start": train_first, "end": train_last},
        "evaluation_window_ms": {"start": eval_first, "end": eval_last},
        "train_rows": len(train_examples), "evaluation_rows": len(eval_examples),
        "train_label_counts": label_counts(train_examples),
        "evaluation_label_counts": label_counts(eval_examples),
        "train_skipped_candidates": train_skipped,
        "evaluation_skipped_candidates": eval_skipped,
        "training_sources": train_sources, "evaluation_sources": eval_sources,
        "limitations": [
            "Vision provides historical exchange event time, not live receive or availability timestamps.",
            "This source contains no funding settlement ledger; generated labels use mark-price return and fixed fee/slippage only.",
            "The evaluation is strictly later than training but currently represents one selected month.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    manifest = {"schema": "td_market_data_laya_vision_prepare_manifest", "version": 1,
                "report_sha256": sha256_file(output / "report.json"),
                "train_sha256": sha256_file(output / "train.jsonl"),
                "evaluation_sha256": sha256_file(output / "evaluation.jsonl"),
                "preparer_sha256": sha256_file(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(output), "train_rows": len(train_examples),
                      "evaluation_rows": len(eval_examples),
                      "train_labels": report["train_label_counts"],
                      "evaluation_labels": report["evaluation_label_counts"],
                      "manifest": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zstandard.ZstdError) as error:
        raise SystemExit(f"Laya Vision preparation failed: {type(error).__name__}: {error}") from error
