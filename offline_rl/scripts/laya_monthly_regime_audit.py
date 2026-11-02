#!/usr/bin/env python3
"""Replay Laya and fixed-position controls in isolated UTC calendar-month Validation slices."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from td_market_data_rl.contracts import MINUTE_NS, feasible_action_mask
from td_market_data_rl.historical import verify_plan
from td_market_data_rl.learner import simulate_episode
from laya_policy_validation import (
    MAX_EVALUATION_RSS_MB, canonical_bytes, checkpoint_inventory,
    current_rss_mb, load_strategy_config, make_state, sha256_file,
    validate_data_path,
)
from laya_simple_baseline_audit import read_laya_artifact


SCRIPT = Path(__file__).resolve()
ARTIFACT_SCHEMA = "td_market_data_laya_monthly_regime_audit_artifact"
REPORT_SCHEMA = "td_market_data_laya_monthly_regime_audit"


def next_month_boundary_ns(timestamp_ns: int) -> int:
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
    if dt.month == 12:
        next_month = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)
    return int(next_month.timestamp() * 1_000_000_000)


def partition_months(window: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    start, end = int(window["start_ns"]), int(window["end_ns"])
    cursor = start
    while cursor < end:
        segment_end = min(next_month_boundary_ns(cursor), end)
        if (segment_end - cursor) % MINUTE_NS:
            raise ValueError("calendar month slice is not aligned to whole UTC minutes")
        month = datetime.fromtimestamp(cursor / 1_000_000_000, timezone.utc).strftime("%Y-%m")
        segments.append({**window, "id": f"{window['id']}:{month}",
                         "start_ns": cursor, "end_ns": segment_end,
                         "minute_count": (segment_end - cursor) // MINUTE_NS,
                         "monthly_segment": month})
        cursor = segment_end
    return segments


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = ("net_return_fraction", "net_pnl_usdt", "gross_price_pnl_usdt",
              "funding_pnl_usdt", "fees_usdt", "slippage_usdt", "fills",
              "traded_notional_usdt", "exposure_minutes", "max_drawdown_fraction")
    return {f"mean_{field}": mean(float(row["metrics"][field]) for row in rows)
            for field in fields}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = validate_data_path(args.historical_inputs, "historical inputs")
    role_plan_path = validate_data_path(args.role_plan, "role plan")
    training = validate_data_path(args.training, "training artifact")
    model_cache = validate_data_path(args.model_cache, "Laya model cache")
    temp_root = validate_data_path(args.temp_root, "temporary workspace")
    _, original, base_manifest_sha = read_laya_artifact(args.laya_artifact)
    config_path = args.laya_config.resolve(strict=True)
    config, config_sha = load_strategy_config(config_path)
    training_manifest_path = training / "manifest.json"
    training_manifest = json.loads(training_manifest_path.read_bytes())
    if (original.get("laya_strategy_config_sha256") != config_sha
            or original.get("role_plan_sha256") != sha256_file(role_plan_path)
            or original.get("historical_input_manifest_sha256") != sha256_file(inputs / "manifest.json")
            or original.get("training_manifest_sha256") != sha256_file(training_manifest_path)):
        raise ValueError("monthly audit inputs differ from the frozen Laya report")
    fee_bps = training_manifest["accounting"]["fee_bps"]
    slippage_bps = training_manifest["accounting"]["slippage_bps"]
    os.environ["HF_HOME"] = str(model_cache)
    os.environ["HF_HUB_CACHE"] = str(model_cache / "hub")
    os.environ["TMPDIR"] = str(temp_root)
    if not torch.cuda.is_available():
        raise ValueError("monthly Laya replay requires CUDA")
    gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    if not 0 < args.max_gpu_memory_mb <= gpu_mb:
        raise ValueError("GPU memory cap must fit the CUDA device")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_mb / gpu_mb, device=0)
    torch.set_num_threads(4)
    from laya import Router
    router = Router(device="cuda", max_loaded=int(config["max_loaded_checkpoints"]),
                    revisions={config["checkpoint"]: config["checkpoint_revision"]})
    plan = verify_plan(role_plan_path, inputs, verify_input_artifact=False)
    windows = plan["roles"].get("validation", [])
    if not windows:
        raise ValueError("role plan has no Validation windows")
    laya_rows, controls = [], {"always_short_v1": [], "always_long_v1": [], "noop_flat_v1": []}
    for window in windows:
        feature_path = inputs / window["features"]["path"]
        ledger_path = inputs / window["ledger_inputs"]["path"]
        for path, metadata in ((feature_path, window["features"]), (ledger_path, window["ledger_inputs"])):
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size != metadata["size_bytes"]
                    or sha256_file(path) != metadata["sha256"]):
                raise ValueError(f"Validation shard hash mismatch: {path}")
        for segment in partition_months(window):
            counts: Counter[str] = Counter()

            def choose_laya(feature: dict[str, Any], position: int):
                mask = feasible_action_mask(feature.get("mark_price"), position)
                if not mask[0] and not mask[2]:
                    counts["flat_forced_by_feasibility_mask"] += 1
                    return 1, None, mask
                state = make_state(feature["features"], feature.get("mark_price"), position)
                result = router.predict(state, {
                    config["question_id"]: {"type": "choice",
                        "instructions": config["instructions"], "criteria": config["criteria"]}
                }, model=config["checkpoint"])
                choice = result.get("answers", {}).get(config["question_id"], {}).get("choice")
                if choice not in config["actions"]:
                    raise ValueError(f"Laya returned unsupported choice: {choice!r}")
                action = {"short": 0, "flat": 1, "long": 2}[choice]
                if not mask[action]:
                    raise ValueError("Laya selected an infeasible action")
                counts[choice] += 1
                counts["model_queries"] += 1
                if current_rss_mb() > args.max_memory_mb:
                    raise ValueError(f"RSS exceeded {args.max_memory_mb} MiB cap")
                return action, None, mask

            _, metrics = simulate_episode(feature_path, ledger_path, segment, choose_laya,
                                          fee_bps, slippage_bps, args.max_memory_mb,
                                          collect_experiences=False)
            laya_rows.append({"strategy": config["strategy_id"], "role": "validation",
                              "window_id": segment["id"], "month_utc": segment["monthly_segment"],
                              "action_counts": dict(sorted(counts.items())), "metrics": metrics})
            for name, target in (("always_short_v1", 0), ("always_long_v1", 2), ("noop_flat_v1", 1)):
                def choose_fixed(feature: dict[str, Any], position: int, wanted: int = target):
                    mask = feasible_action_mask(feature.get("mark_price"), position)
                    return (wanted if mask[wanted] else 1), None, mask
                _, fixed_metrics = simulate_episode(feature_path, ledger_path, segment, choose_fixed,
                                                    fee_bps, slippage_bps, args.max_memory_mb,
                                                    collect_experiences=False)
                controls[name].append({"strategy": name, "role": "validation",
                    "window_id": segment["id"], "month_utc": segment["monthly_segment"],
                    "metrics": fixed_metrics})
    checkpoint, _ = checkpoint_inventory(model_cache, config["checkpoint_revision"])
    strategies = {"laya_zero_shot": summarize(laya_rows),
                  **{name: summarize(rows) for name, rows in controls.items()}}
    return {
        "schema": REPORT_SCHEMA, "version": 1, "evaluation_role": "validation",
        "segmentation": "UTC calendar-month isolated episodes; start flat and liquidate at each month boundary. Monthly values are not additive to the continuous 90-day report.",
        "laya_artifact_manifest_sha256": base_manifest_sha,
        "laya_report_sha256": sha256_file(Path(args.laya_artifact) / "report.json"),
        "laya_strategy_config_sha256": config_sha, "checkpoint": checkpoint,
        "historical_input_manifest_sha256": sha256_file(inputs / "manifest.json"),
        "role_plan_sha256": sha256_file(role_plan_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
        "monthly_laya_results": laya_rows, "monthly_control_results": controls,
        "summary": {"strategies": strategies,
                    "monthly_laya_beats_always_short": [
                        laya_rows[i]["metrics"]["net_return_fraction"] > controls["always_short_v1"][i]["metrics"]["net_return_fraction"]
                        for i in range(len(laya_rows))]},
        "audit_code_sha256": sha256_file(SCRIPT),
        "limitations": [
            "This is a regime sensitivity analysis on the same already-inspected Validation data.",
            "Each UTC month begins flat and is liquidated at the end; boundary resets change the continuous policy path.",
            "It is not independent evidence and must not be used as OOS performance.",
            "Test and OOS are not read.",
        ],
    }


def write_artifact(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    output = output.absolute()
    validate_data_path(output.parent, "artifact parent")
    if output.exists() or output.is_symlink():
        raise ValueError("monthly audit output must be new and create-only")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = stage / "report.json"
        report_path.write_bytes(canonical_bytes(report))
        manifest = {"schema": ARTIFACT_SCHEMA, "version": 1,
                    "report_sha256": sha256_file(report_path),
                    "laya_report_sha256": report["laya_report_sha256"],
                    "audit_code_sha256": report["audit_code_sha256"]}
        (stage / "manifest.json").write_bytes(canonical_bytes(manifest))
        for path in stage.iterdir():
            with path.open("rb") as stream: os.fsync(stream.fileno())
            os.chmod(path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(fd)
        finally: os.close(fd)
        return manifest
    except BaseException:
        try: os.chmod(stage, 0o755); shutil.rmtree(stage, ignore_errors=True)
        except OSError: pass
        raise


def verify_artifact(args: argparse.Namespace) -> dict[str, Any]:
    artifact = validate_data_path(args.artifact, "monthly audit artifact")
    if artifact.is_symlink() or not artifact.is_dir() or {p.name for p in artifact.iterdir()} != {"manifest.json", "report.json"}:
        raise ValueError("monthly audit artifact has missing or unmanifested files")
    manifest_path, report_path = artifact / "manifest.json", artifact / "report.json"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, report_path)):
        raise ValueError("monthly audit artifact files must be regular files")
    manifest, report_bytes = json.loads(manifest_path.read_bytes()), report_path.read_bytes()
    report = json.loads(report_bytes)
    if (manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("laya_report_sha256") != report.get("laya_report_sha256")
            or manifest.get("audit_code_sha256") != report.get("audit_code_sha256")):
        raise ValueError("monthly audit manifest/report mismatch")
    if canonical_bytes(build_report(args)) != report_bytes:
        raise ValueError("monthly Laya/control replay differs from artifact")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        sub = commands.add_parser(name)
        for option in ("historical-inputs", "role-plan", "training", "laya-artifact",
                       "laya-config", "model-cache", "temp-root"):
            sub.add_argument(f"--{option}", type=Path, required=True)
        sub.add_argument("--max-memory-mb", type=int, default=MAX_EVALUATION_RSS_MB)
        sub.add_argument("--max-gpu-memory-mb", type=int, default=8192)
        sub.add_argument("--artifact" if name == "verify" else "--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_memory_mb <= MAX_EVALUATION_RSS_MB:
        raise SystemExit(f"--max-memory-mb must be in 1..{MAX_EVALUATION_RSS_MB}")
    try:
        if args.command == "create":
            report = build_report(args)
            manifest = write_artifact(args.output, report)
            for strategy, stats in report["summary"]["strategies"].items():
                print(f"{strategy}: monthly mean return={stats['mean_net_return_fraction']:.6%}, "
                      f"mean drawdown={stats['mean_max_drawdown_fraction']:.6%}, "
                      f"exposure={stats['mean_exposure_minutes']:.1f} min, "
                      f"fills={stats['mean_fills']:.2f}, fees=${stats['mean_fees_usdt']:.4f}, "
                      f"slippage=${stats['mean_slippage_usdt']:.4f}")
            print(f"monthly Laya beats always-short: {report['summary']['monthly_laya_beats_always_short']}")
            print(f"artifact: {args.output.resolve()}")
            print(f"report sha256: {manifest['report_sha256']}")
        else:
            manifest = verify_artifact(args)
            print("monthly Laya regime replay recomputed and verified")
            print(f"report sha256: {manifest['report_sha256']}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            torch.cuda.OutOfMemoryError) as error:
        raise SystemExit(f"Laya monthly regime audit rejected: {error}") from error


if __name__ == "__main__":
    main()
