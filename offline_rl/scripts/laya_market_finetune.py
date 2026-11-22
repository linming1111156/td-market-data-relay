#!/usr/bin/env python3
"""Prepare time-aligned Laya labels from audited inputs, then fine-tune Laya."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "bulk_engine"
ENGINE_TARGET_DIR = Path(os.environ.get("CARGO_TARGET_DIR", ENGINE_DIR / "target")).expanduser()
ENGINE = ENGINE_TARGET_DIR / "release/td-market-data-bulk-engine"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            resolved = path.resolve(strict=True)
            files.append({"path": path.relative_to(root).as_posix(),
                          "size_bytes": resolved.stat().st_size,
                          "sha256": sha256_file(resolved)})
    if not files:
        raise ValueError(f"base Laya checkpoint has no files: {root}")
    return files


def validate_prepared_data(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prepared data must be a real directory: {root}")
    expected_names = {"train.jsonl", "evaluation.jsonl", "report.json", "manifest.json"}
    entries = list(root.iterdir())
    if (any(path.is_symlink() for path in entries)
            or {path.name for path in entries if path.is_file()} != expected_names
            or any(not path.is_file() and not path.is_dir() for path in entries)):
        raise ValueError("prepared data must contain hash-bound train/evaluation/report/manifest files")
    manifest_path, report_path = root / "manifest.json", root / "report.json"
    train_path, evaluation_path = root / "train.jsonl", root / "evaluation.jsonl"
    if any(path.is_symlink() or not path.is_file()
           for path in (manifest_path, report_path, train_path, evaluation_path)):
        raise ValueError("prepared data files must be regular files")
    manifest = json.loads(manifest_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != "td_market_data_laya_vision_prepare_manifest"
            or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("train_sha256") != sha256_file(train_path)
            or manifest.get("evaluation_sha256") != sha256_file(evaluation_path)
            or report.get("schema") != "td_market_data_laya_vision_prepare_report"
            or report.get("version") != 1
            or report.get("time_basis") != "historical Binance Vision exchange event time; receive/availability time is unknown"):
        raise ValueError("prepared Vision data manifest, hashes, or time basis are invalid")
    for split, path in (("train", train_path), ("evaluation", evaluation_path)):
        expected_rows = report.get(f"{split}_rows")
        seen = 0
        previous_time = None
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                label = row.get("expected", {}).get("target_position")
                timestamp = row.get("decision_time_unix_ms")
                if (label not in {"short", "flat", "long"}
                        or not isinstance(timestamp, int)
                        or not isinstance(row.get("state", {}).get("market_features"), dict)
                        or not isinstance(row.get("outcome"), dict)):
                    raise ValueError(f"invalid prepared {split} row {line_number}")
                if previous_time is not None and timestamp <= previous_time:
                    raise ValueError(f"prepared {split} timestamps are not strictly increasing")
                previous_time = timestamp
                seen += 1
        if seen != expected_rows or seen == 0:
            raise ValueError(f"prepared {split} row count differs from its report")
    if (report["train_window_ms"]["end"] + report["label_horizon_minutes"] * 60_000
            >= report["evaluation_window_ms"]["start"]):
        raise ValueError("prepared Vision train/evaluation windows overlap in time")
    return manifest, report, sha256_file(manifest_path)


def validate_relay_prepared_data(root: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prepared Relay data must be a real directory: {root}")
    expected_names = {"train.jsonl", "evaluation.jsonl", "report.json", "manifest.json"}
    entries = list(root.iterdir())
    if ({path.name for path in entries if path.is_file()} != expected_names
            or any(path.is_symlink() or not path.is_file() for path in entries)):
        raise ValueError("prepared Relay data must contain regular train/evaluation/report/manifest files only")
    manifest_path, report_path = root / "manifest.json", root / "report.json"
    train_path, evaluation_path = root / "train.jsonl", root / "evaluation.jsonl"
    manifest = json.loads(manifest_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != "td_market_data_laya_relay_prepared_manifest"
            or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("train_sha256") != sha256_file(train_path)
            or manifest.get("evaluation_sha256") != sha256_file(evaluation_path)
            or report.get("schema") != "td_market_data_laya_prepare_result"
            or report.get("version") != 1):
        raise ValueError("prepared Relay data manifest, hashes, or report schema are invalid")
    train_rows, evaluation_rows = report.get("train_rows"), report.get("evaluation_rows")
    slicing = report.get("time_slicing")
    horizon = slicing.get("label_horizon_minutes") if isinstance(slicing, dict) else None
    last_train = slicing.get("last_train_minute_utc") if isinstance(slicing, dict) else None
    first_eval = slicing.get("first_evaluation_minute_utc") if isinstance(slicing, dict) else None
    if (not isinstance(train_rows, int) or train_rows <= 0
            or not isinstance(evaluation_rows, int) or evaluation_rows <= 0
            or not isinstance(horizon, int) or horizon < 0
            or not isinstance(last_train, int) or not isinstance(first_eval, int)
            or last_train + horizon > first_eval):
        raise ValueError("prepared Relay report has invalid row counts or overlapping time windows")
    for split, path, expected_rows in (("train", train_path, train_rows),
                                       ("evaluation", evaluation_path, evaluation_rows)):
        seen = 0
        previous_time = None
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                row = json.loads(line)
                label = row.get("expected", {}).get("target_position")
                timestamp = row.get("decision_time_unix_ms")
                if (label not in {"short", "flat", "long"}
                        or not isinstance(timestamp, int)
                        or not isinstance(row.get("state", {}).get("market_features"), dict)
                        or not isinstance(row.get("outcome"), dict)):
                    raise ValueError(f"invalid prepared Relay {split} row {line_number}")
                if previous_time is not None and timestamp <= previous_time:
                    raise ValueError(f"prepared Relay {split} timestamps are not strictly increasing")
                previous_time = timestamp
                seen += 1
        if seen != expected_rows:
            raise ValueError(f"prepared Relay {split} row count differs from its report")
    return manifest, report, sha256_file(manifest_path)


def evaluate_checkpoint_trading(checkpoint: Path, evaluation_path: Path,
                                device: str, batch_size: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("held-out evaluation set is empty")
    questions = rows[0].get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("held-out rows lack the Laya questions schema")
    if any(row.get("questions") != questions for row in rows):
        raise ValueError("held-out rows use inconsistent question schemas")
    from laya import load

    agent = load(str(checkpoint), device=device, vegaml=False)
    try:
        details = agent.decide_batch(
            [row["state"] for row in rows], questions=questions,
            return_details=True, batch_size=min(batch_size, 8))
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    predictions: list[dict[str, Any]] = []
    actions: list[str] = []
    truths: list[str] = []
    timestamps: list[int] = []
    for row, detail in zip(rows, details, strict=True):
        answer = detail.answers.get("target_position", {})
        action = answer.get("choice") if isinstance(answer, dict) else None
        truth = row.get("expected", {}).get("target_position")
        timestamp = row.get("decision_time_unix_ms")
        if action not in {"short", "flat", "long"} or truth not in {"short", "flat", "long"}:
            raise ValueError("Laya held-out prediction or target_position label is invalid")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool):
            raise ValueError("held-out rows need integer decision timestamps")
        actions.append(action)
        truths.append(truth)
        timestamps.append(timestamp)
        predictions.append({
            "decision_time_unix_ms": timestamp,
            "actual": truth,
            "predicted": action,
            "probabilities": detail.probabilities.get("target_position", {}),
            "forward_mark_return_bps": row.get("outcome", {}).get("forward_mark_return_bps"),
            "round_trip_cost_bps": row.get("outcome", {}).get("round_trip_cost_bps"),
        })
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("held-out decision timestamps are not strictly increasing")

    classes = ("short", "flat", "long")

    def summarize(start: int, end: int) -> dict[str, Any]:
        matrix = {truth: {guess: 0 for guess in classes} for truth in classes}
        pnl: list[float] = []
        gross = cost_total = 0.0
        trades = 0
        for index in range(start, end):
            row, action, truth = rows[index], actions[index], truths[index]
            matrix[truth][action] += 1
            outcome = row.get("outcome")
            if not isinstance(outcome, dict):
                raise ValueError("held-out row lacks forward outcome metadata")
            forward = outcome.get("forward_mark_return_bps")
            cost = outcome.get("round_trip_cost_bps")
            if not isinstance(forward, (int, float)) or not isinstance(cost, (int, float)):
                raise ValueError("held-out row has invalid forward return or transaction cost")
            position = 1 if action == "long" else -1 if action == "short" else 0
            trade_gross = position * float(forward)
            trade_cost = float(cost) if position else 0.0
            gross += trade_gross
            cost_total += trade_cost
            pnl.append(trade_gross - trade_cost)
            trades += int(position != 0)
        n = end - start
        recalls = []
        f1s = []
        for label in classes:
            true_positive = matrix[label][label]
            support = sum(matrix[label].values())
            predicted_count = sum(matrix[truth][label] for truth in classes)
            recall = true_positive / support if support else 0.0
            precision = true_positive / predicted_count if predicted_count else 0.0
            recalls.append(recall)
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        equity = peak = max_drawdown = 0.0
        for value in pnl:
            equity += value
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)
        return {
            "decisions": n,
            "accuracy": sum(truths[i] == actions[i] for i in range(start, end)) / n,
            "balanced_accuracy": sum(recalls) / len(classes),
            "macro_f1": sum(f1s) / len(classes),
            "confusion_actual_by_predicted": matrix,
            "trades": trades,
            "turnover_sides": 2 * trades,
            "gross_pnl_bps": round(gross, 4),
            "cost_bps": round(cost_total, 4),
            "net_pnl_bps": round(sum(pnl), 4),
            "net_pnl_bps_per_decision": round(sum(pnl) / n, 4),
            "max_drawdown_bps": round(max_drawdown, 4),
        }

    ranges = []
    start = 0
    for index in range(1, len(rows)):
        if timestamps[index] - timestamps[index - 1] > 30 * 60_000:
            ranges.append((start, index))
            start = index
    ranges.append((start, len(rows)))
    result = {
        "checkpoint": str(checkpoint),
        "device": device,
        "evaluation_rows": len(rows),
        "method": "one independent one-lot decision per 15-minute row; round-trip fees and slippage charged for each non-flat action",
        "overall": summarize(0, len(rows)),
        "time_segments": [
            {"start_time_unix_ms": timestamps[start], "end_time_unix_ms": timestamps[end - 1],
             "metrics": summarize(start, end)}
            for start, end in ranges
        ],
    }
    return result, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path,
                        default=Path("/kaggle/input/datasets/minglinl99/td-market-vps-captures"),
                        help="Kaggle input root containing captures/<stream>/BTCUSDT/*.jsonl.zst")
    parser.add_argument("--additional-archive-root", action="append", type=Path, default=[],
                        help="additional mounted capture dataset root; may be supplied more than once")
    parser.add_argument("--additional-dataset-ref", action="append", default=[],
                        help="Kaggle dataset ref for each additional archive root")
    parser.add_argument("--prepared-data-dir", type=Path,
                        help="verified train/evaluation JSONL from laya_vision_prepare.py")
    parser.add_argument("--relay-prepared-data-dir", type=Path,
                        help="verified reusable train/evaluation artifact from this script's Rust prepare-only run")
    parser.add_argument("--output", type=Path,
                        default=Path("/kaggle/working/laya-market-finetune-v1"))
    parser.add_argument("--base", default="convaiinnovations/laya")
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto",
                        help="auto requires a fresh Kaggle quota check and uses CUDA while quota remains")
    parser.add_argument("--gpu-hours-used-7d", type=float,
                        help="required for GPU training: observed Kaggle GPU hours in the rolling last 7 days")
    parser.add_argument("--gpu-planned-hours", type=float,
                        help="required for GPU training: planned GPU run duration, within the official 30-hour rolling 7-day limit")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 2, 8))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--loss", choices=("rlcd", "soft-ce"), default="rlcd",
                        help="Laya choice-head loss; keep fixed for single-factor comparisons")
    parser.add_argument("--unfreeze-encoder", action="store_true",
                        help="fine-tune the encoder as well as the task head")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--decision-interval-minutes", type=int, default=15)
    parser.add_argument("--horizon-minutes", type=int, default=15)
    parser.add_argument("--fee-bps-each-side", type=float, default=10.0)
    parser.add_argument("--slippage-bps-each-side", type=float, default=5.0)
    parser.add_argument("--edge-bps-round-trip", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true",
                        help="build the labelled chronological JSONL sets without loading Laya")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya data preparation and model runs must happen inside Kaggle; the local machine is code-only")
    prepared_source = None
    prepared_source_manifest = None
    prepared_source_kind = None
    if args.relay_prepared_data_dir:
        if args.prepared_data_dir or args.additional_archive_root or args.additional_dataset_ref:
            raise ValueError("Relay prepared data cannot be combined with Vision prepared data or archive inputs")
        prepared_source = args.relay_prepared_data_dir.expanduser().resolve(strict=True)
        prepared_source_manifest, prepared, prepared_source_manifest_sha256 = validate_relay_prepared_data(prepared_source)
        prepared_source_kind = "relay"
        archive_root = None
        additional_archive_roots = []
    elif args.prepared_data_dir:
        if args.additional_archive_root or args.additional_dataset_ref:
            raise ValueError("prepared Vision data cannot be combined with Relay archive inputs")
        prepared_source = args.prepared_data_dir.expanduser().resolve(strict=True)
        prepared_source_manifest, prepared, prepared_source_manifest_sha256 = validate_prepared_data(prepared_source)
        prepared_source_kind = "vision"
        archive_root = None
        additional_archive_roots = []
    else:
        archive_root = args.archive_root.resolve(strict=True)
        additional_archive_roots = [path.resolve(strict=True) for path in args.additional_archive_root]
        if args.additional_dataset_ref and len(args.additional_dataset_ref) != len(additional_archive_roots):
            raise ValueError("provide one --additional-dataset-ref per --additional-archive-root")
        for path in [archive_root, *additional_archive_roots]:
            if not path.is_dir() or path.is_symlink():
                raise ValueError(f"archive root must be a real directory: {path}")
    output = args.output.absolute()
    if output.exists() or output.is_symlink() or ROOT == output or ROOT in output.parents:
        raise ValueError("output must be a new directory outside the repository")
    if args.workers < 1 or args.workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if min(args.epochs, args.micro_batch, args.grad_accum) < 1:
        raise ValueError("epochs, micro-batch, and grad-accum must be positive")
    gpu_budget = None
    if not args.prepare_only and args.device in ("auto", "cuda"):
        used, planned = args.gpu_hours_used_7d, args.gpu_planned_hours
        quota_verified = (used is not None and planned is not None
                          and math.isfinite(used) and math.isfinite(planned)
                          and 0 <= used <= 30 and planned > 0)
        if not quota_verified:
            raise ValueError("model training requires a fresh Kaggle 7-day GPU quota check and valid --gpu-hours-used-7d/--gpu-planned-hours")
        within_plan = used < 30 and used + planned <= 30
        if used < 30 and not within_plan:
            raise ValueError("GPU quota remains but this plan exceeds it; shorten the GPU plan or wait for quota refresh instead of falling back to CPU")
        if args.device == "cuda" and not within_plan:
            raise ValueError("GPU budget gate failed: planned total must stay within the official 30 h rolling 7-day limit")
        gpu_budget = {"observed_hours_used_rolling_7d": used,
                      "planned_hours": planned,
                      "planned_total_hours": used + planned,
                      "quota_verified": True,
                      "within_plan": within_plan,
                      "plan_limit_hours": 30,
                      "hard_limit_hours": 30,
                      "operator_checked_quota_before_run": True}

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    data_dir = output / "dataset"
    report_path = output / "prepare-report.json"
    request_path = output / "rust-prepare-request.json"
    rust_engine_hash = None
    prepared_data_manifest_sha256 = None
    if prepared_source is not None:
        data_dir.mkdir()
        shutil.copyfile(prepared_source / "train.jsonl", data_dir / "train.jsonl")
        shutil.copyfile(prepared_source / "evaluation.jsonl", data_dir / "evaluation.jsonl")
        shutil.copyfile(prepared_source / "report.json", report_path)
        rust_prepare = prepared if prepared_source_kind == "relay" else None
        prepared_data_manifest_sha256 = prepared_source_manifest_sha256
        rust_engine_hash = (prepared_source_manifest.get("rust_engine_sha256")
                            if prepared_source_kind == "relay" else None)
        prepared_source_path = str(prepared_source)
    else:
        request = {
            "schema": "td_market_data_laya_prepare_request",
            "version": 1,
            "archive_root": str(archive_root),
            "additional_archive_roots": [str(path) for path in additional_archive_roots],
            "output_dir": str(data_dir),
            "report_path": str(report_path),
            "horizon_minutes": args.horizon_minutes,
            "decision_interval_minutes": args.decision_interval_minutes,
            "fee_bps_each_side": args.fee_bps_each_side,
            "slippage_bps_each_side": args.slippage_bps_each_side,
            "edge_bps_round_trip": args.edge_bps_round_trip,
            "train_fraction": args.train_fraction,
            "workers": args.workers,
        }
        request_path.write_text(json.dumps(request, sort_keys=True, indent=2) + "\n")
        cargo = shutil.which("cargo")
        if not cargo:
            raise RuntimeError("Rust toolchain is required to prepare zstd shards")
        subprocess.run([cargo, "build", "--release", "--manifest-path",
                        str(ENGINE_DIR / "Cargo.toml")], cwd=ROOT, check=True)
        subprocess.run([str(ENGINE), "laya-prepare", str(request_path)], cwd=ROOT, check=True)
        rust_prepare = json.loads(report_path.read_bytes())
        # Keep the same summary contract used by the prepared-data path below.
        prepared = rust_prepare
        rust_engine_hash = sha256_file(ENGINE)
        prepared_source_path = None
        shutil.copyfile(report_path, data_dir / "report.json")
        prepared_manifest = {
            "schema": "td_market_data_laya_relay_prepared_manifest",
            "version": 1,
            "report_sha256": sha256_file(data_dir / "report.json"),
            "train_sha256": sha256_file(data_dir / "train.jsonl"),
            "evaluation_sha256": sha256_file(data_dir / "evaluation.jsonl"),
            "rust_engine_sha256": rust_engine_hash,
            "request_sha256": sha256_file(request_path),
        }
        (data_dir / "manifest.json").write_text(
            json.dumps(prepared_manifest, sort_keys=True, indent=2) + "\n")
        prepared_data_manifest_sha256 = sha256_file(data_dir / "manifest.json")

    if prepared_source_kind == "vision":
        source_datasets = ["Binance Vision audited historical feature set"]
    elif prepared_source_kind == "relay":
        source_datasets = ["Kaggle Relay prepared artifact"]
    else:
        source_datasets = ["minglinl99/td-market-vps-captures", *args.additional_dataset_ref]
    run: dict[str, Any] = {
        "schema": "td_market_data_laya_market_finetune",
        "version": 1,
        "source_datasets": source_datasets,
        "archive_root": str(archive_root) if archive_root is not None else None,
        "additional_archive_roots": [str(path) for path in additional_archive_roots],
        "prepared_data_directory": prepared_source_path,
        "prepared_data_manifest_sha256": prepared_data_manifest_sha256,
        "rust_engine_sha256": rust_engine_hash,
        "rust_prepare": rust_prepare,
        "train_jsonl_sha256": sha256_file(data_dir / "train.jsonl"),
        "evaluation_jsonl_sha256": sha256_file(data_dir / "evaluation.jsonl"),
        "gpu_budget": gpu_budget,
    }
    if not args.prepare_only:
        import torch
        from laya.train import TrainConfig, finetune, resolve_checkpoint_dir

        device = args.device
        if args.device == "auto":
            if gpu_budget is None:
                raise ValueError("model training requires a verified Kaggle GPU quota")
            if gpu_budget["within_plan"]:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cpu":
                    gpu_budget["fallback_reason"] = "Kaggle did not provide CUDA; runtime probe confirmed CUDA is unavailable"
            else:
                device = "cpu"
                gpu_budget["fallback_reason"] = "official rolling GPU quota is exhausted"
            gpu_budget["selected_device"] = device
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if gpu_budget is not None:
            gpu_budget["selected_device"] = device
        runtime_probe = {
            "selected_device": device,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_budget": gpu_budget,
        }
        run["runtime_probe"] = runtime_probe
        print("Laya finetune runtime: " + json.dumps(runtime_probe, sort_keys=True), flush=True)
        base = Path(resolve_checkpoint_dir(args.base)).resolve(strict=True)
        base_manifest = checkpoint_files(base)
        checkpoint_dir = output / "checkpoint"
        config = TrainConfig(epochs=args.epochs, micro_batch=args.micro_batch,
                             grad_accum=args.grad_accum,
                             freeze_encoder=not args.unfreeze_encoder,
                             loss=args.loss,
                             seed=args.seed,
                             eval_data=str(data_dir / "evaluation.jsonl"))
        training_started = time.monotonic()
        result = finetune(str(data_dir / "train.jsonl"), str(base), str(checkpoint_dir),
                          config=config, device=device)
        training_elapsed = time.monotonic() - training_started
        run.update({"base_checkpoint": args.base,
                    "resolved_base_checkpoint": str(base),
                    "base_checkpoint_files": base_manifest,
                    "train_config": {"epochs": args.epochs,
                                     "micro_batch": args.micro_batch,
                                     "grad_accum": args.grad_accum,
                                     "freeze_encoder": not args.unfreeze_encoder,
                                     "loss": args.loss,
                                     "seed": args.seed,
                                     "device": device},
                    "training_result": result,
                    "training_elapsed_seconds": round(training_elapsed, 3),
                    "actual_gpu_hours": round(training_elapsed / 3600, 6) if device == "cuda" else 0.0})
        evaluation_started = time.monotonic()
        trading_evaluation, trading_predictions = evaluate_checkpoint_trading(
            checkpoint_dir, data_dir / "evaluation.jsonl", device=device,
            batch_size=args.micro_batch)
        evaluation_elapsed = time.monotonic() - evaluation_started
        prediction_path = output / "trading-evaluation-predictions.json"
        prediction_path.write_text(json.dumps(
            trading_predictions, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        run.update({"trading_evaluation": trading_evaluation,
                    "trading_evaluation_elapsed_seconds": round(evaluation_elapsed, 3),
                    "actual_gpu_hours": round((training_elapsed + evaluation_elapsed) / 3600, 6)
                    if device == "cuda" else 0.0,
                    "trading_evaluation_predictions": str(prediction_path),
                    "trading_evaluation_predictions_sha256": sha256_file(prediction_path)})
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run, ensure_ascii=False, sort_keys=True,
                                   indent=2, default=str) + "\n")
    print(json.dumps({"output": str(output),
                      "prepared_train_rows": prepared["train_rows"],
                      "prepared_evaluation_rows": prepared["evaluation_rows"],
                      "evaluation_labels": prepared["evaluation_label_counts"],
                      "trained": not args.prepare_only}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Laya market fine-tune failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
