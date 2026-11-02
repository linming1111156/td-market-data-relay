#!/usr/bin/env python3
"""Train-only forward-return diagnostics for the frozen local factor registry."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from td_market_data_rl.contracts import MINUTE_NS
from td_market_data_rl.factors import factor_values, registry_sha256, suites
from td_market_data_rl.historical import sha256_file, training_windows, verify_plan
from td_market_data_rl.learner import jsonl


HORIZONS_MINUTES = (15, 60, 240)
ROUND_TRIP_COST_HURDLE_BPS = 30.0  # 10 bps fee + 5 bps slippage on entry and exit.


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2:
        return None
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(np.dot(left_centered, left_centered)
                                * np.dot(right_centered, right_centered)))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def _factor_rows(feature_path: Path, expected_rows: int,
                 factor_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
    timestamps = np.empty(expected_rows, dtype=np.int64)
    prices = np.full(expected_rows, np.nan, dtype=np.float64)
    decision_indices: list[int] = []
    decision_values: list[list[float]] = []
    decision_prices: list[float] = []
    count = 0
    for row in jsonl(feature_path):
        if count >= expected_rows:
            raise ValueError(f"feature shard has more than {expected_rows} rows: {feature_path}")
        timestamp = row.get("decision_at_ns")
        if not isinstance(timestamp, int):
            raise ValueError(f"feature row {count} has an invalid event timestamp")
        timestamps[count] = timestamp
        price = row.get("mark_price")
        if isinstance(price, (int, float)) and np.isfinite(price) and price > 0:
            prices[count] = float(price)
        if row.get("decision_due") and row.get("feature_ready"):
            values = factor_values(row["features"])
            decision_indices.append(count)
            decision_values.append([values[name] for name in factor_names])
            decision_prices.append(prices[count])
        count += 1
    if count != expected_rows:
        raise ValueError(f"feature shard row count {count} != role-plan count {expected_rows}")
    indices = np.asarray(decision_indices, dtype=np.int64)
    matrix = np.asarray(decision_values, dtype=np.float64)
    current_prices = np.asarray(decision_prices, dtype=np.float64)
    return timestamps, prices, indices, matrix, current_prices


def _window_diagnostics(input_root: Path, window: dict[str, Any],
                        factor_names: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    feature_path = input_root / window["features"]["path"]
    ledger_path = input_root / window["ledger_inputs"]["path"]
    for path, entry in ((feature_path, window["features"]), (ledger_path, window["ledger_inputs"])):
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size_bytes"]
                or sha256_file(path) != entry["sha256"]):
            raise ValueError(f"Train shard failed role-plan hash verification: {path}")
    timestamps, prices, indices, values, current_prices = _factor_rows(
        feature_path, window["minute_count"], factor_names)
    decision_times = timestamps[indices]
    by_horizon: list[dict[str, Any]] = []
    packed: list[dict[str, Any]] = []
    for horizon in HORIZONS_MINUTES:
        target_times = decision_times + horizon * MINUTE_NS
        target_indices = np.searchsorted(timestamps, target_times)
        in_range = target_indices < timestamps.size
        valid = in_range.copy()
        valid[in_range] &= timestamps[target_indices[in_range]] == target_times[in_range]
        safe_targets = np.minimum(target_indices, timestamps.size - 1)
        valid &= np.isfinite(current_prices) & (current_prices > 0)
        valid &= np.isfinite(prices[safe_targets]) & (prices[safe_targets] > 0)
        valid_rows = np.flatnonzero(valid)
        forward = np.log(prices[safe_targets[valid]] / current_prices[valid])
        results = []
        for column, name in enumerate(factor_names):
            factor = values[valid, column]
            corr = _correlation(factor, forward)
            if factor.size:
                order = np.argsort(factor, kind="stable")
                decile = max(1, factor.size // 10)
                bottom = float(forward[order[:decile]].mean())
                top = float(forward[order[-decile:]].mean())
                spread = top - bottom
            else:
                bottom = top = spread = 0.0
            results.append({"factor": name,
                            "pearson_corr_with_forward_log_return": corr,
                            "bottom_decile_mean_return_bps": bottom * 10_000,
                            "top_decile_mean_return_bps": top * 10_000,
                            "top_minus_bottom_return_bps": spread * 10_000,
                            "sample_count": int(factor.size)})
        results.sort(key=lambda item: (-(abs(item["pearson_corr_with_forward_log_return"])
                                         if item["pearson_corr_with_forward_log_return"] is not None else -1),
                                       item["factor"]))
        by_horizon.append({"horizon_minutes": horizon, "sample_count": int(valid.sum()),
                           "factors_ranked_by_absolute_pearson": results})
        packed.append({"horizon_minutes": horizon,
                       "timestamps": decision_times[valid_rows],
                       "factor_values": values[valid], "forward_log_returns": forward})
    return ({"window_id": window["id"], "minute_count": window["minute_count"],
             "feature_sha256": window["features"]["sha256"],
             "ledger_sha256": window["ledger_inputs"]["sha256"],
             "ready_decision_count": int(indices.size), "horizons": by_horizon}, packed)


def create_report(historical_inputs: Path, role_plan_path: Path) -> dict[str, Any]:
    input_root = historical_inputs.resolve(strict=True)
    plan_path = role_plan_path.resolve(strict=True)
    plan = verify_plan(plan_path, input_root, verify_input_artifact=False)
    train = training_windows(input_root, plan)
    all_factors = set().union(*suites().values())
    factor_names = sorted(all_factors)
    window_reports = []
    packed_by_horizon: dict[int, list[dict[str, Any]]] = {h: [] for h in HORIZONS_MINUTES}
    for item, window in zip(train, plan["roles"]["train"], strict=True):
        report, packed = _window_diagnostics(input_root, window, factor_names)
        window_reports.append(report)
        for entry in packed:
            packed_by_horizon[entry["horizon_minutes"]].append(entry)
    pooled = []
    for horizon in HORIZONS_MINUTES:
        entries = packed_by_horizon[horizon]
        factors = np.concatenate([entry["factor_values"] for entry in entries], axis=0)
        returns = np.concatenate([entry["forward_log_returns"] for entry in entries])
        rows = []
        for index, name in enumerate(factor_names):
            values = factors[:, index]
            corr = _correlation(values, returns)
            order = np.argsort(values, kind="stable")
            decile = max(1, values.size // 10)
            bottom = float(returns[order[:decile]].mean())
            top = float(returns[order[-decile:]].mean())
            rows.append({"factor": name,
                         "pearson_corr_with_forward_log_return": corr,
                         "bottom_decile_mean_return_bps": bottom * 10_000,
                         "top_decile_mean_return_bps": top * 10_000,
                         "top_minus_bottom_return_bps": (top - bottom) * 10_000,
                         "sample_count": int(values.size)})
        rows.sort(key=lambda item: (-(abs(item["pearson_corr_with_forward_log_return"])
                                      if item["pearson_corr_with_forward_log_return"] is not None else -1),
                                    item["factor"]))
        pooled.append({"horizon_minutes": horizon, "sample_count": int(returns.size),
                       "factors_ranked_by_absolute_pearson": rows})
    script_path = Path(__file__).resolve()
    manifest = json.loads((input_root / "manifest.json").read_bytes())
    return {"schema": "td_market_data_usdm_rl_train_signal_diagnostics", "version": 1,
            "role": "train", "historical_input_manifest_sha256": sha256_file(input_root / "manifest.json"),
            "role_plan_sha256": sha256_file(plan_path),
            "role_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
            "bulk_audit_sha256": manifest["source_manifests"]["bulk_audit_sha256"],
            "factor_registry_sha256": registry_sha256(),
            "diagnostic_code_sha256": sha256_file(script_path),
            "factor_names": factor_names,
            "method": {"decision_filter": "decision_due_and_feature_ready",
                       "target": "forward_mark_log_return_from_decision_timestamp",
                       "horizons_minutes": list(HORIZONS_MINUTES),
                       "correlation": "pearson",
                       "tail_summary": "top and bottom factor deciles, sorted within each horizon",
                       "round_trip_cost_hurdle_bps": ROUND_TRIP_COST_HURDLE_BPS,
                       "cost_hurdle_note": "30 bps is the one-direction entry-plus-exit fee/slippage hurdle; decile spread is descriptive and is not a net strategy return."},
            "train_windows": window_reports, "pooled_train": pooled,
            "limitations": ["All forward labels remain inside their same audited Train window.",
                            "This is a signal-screening diagnostic, not an executable strategy backtest.",
                            "It ignores participation, fill timing, Funding, and path-dependent turnover.",
                            "Validation, Test, and OOS rows are not read."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-inputs", type=Path, required=True)
    parser.add_argument("--role-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise SystemExit("output must be a new create-only file")
    resolved = output.resolve()
    data_root = Path("/mnt/data").resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[2]
    if resolved == data_root or data_root not in resolved.parents:
        raise SystemExit("output must be stored under /mnt/data")
    if resolved == repo_root or repo_root in resolved.parents:
        raise SystemExit("output must be outside the source repository")
    if any(resolved == root or root in resolved.parents for root in (Path("/tmp"), Path("/var/tmp"))):
        raise SystemExit("system temporary directories are not allowed")
    report = create_report(args.historical_inputs, args.role_plan)
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    print(f"Train-only signal diagnostics written: {output}")
    print(f"factors={len(report['factor_names'])} windows={len(report['train_windows'])} "
          f"horizons={len(HORIZONS_MINUTES)}; Validation/Test/OOS not read")


if __name__ == "__main__":
    main()
