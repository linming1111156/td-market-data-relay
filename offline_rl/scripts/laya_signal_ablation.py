#!/usr/bin/env python3
"""Measure whether zero-shot Laya decisions change when market features are neutralized."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from td_market_data_rl.contracts import feasible_action_mask
from td_market_data_rl.historical import verify_plan
from td_market_data_rl.learner import simulate_episode
from laya_policy_validation import (
    MAX_EVALUATION_RSS_MB,
    canonical_bytes,
    checkpoint_inventory,
    current_rss_mb,
    load_strategy_config,
    make_state,
    sha256_file,
    validate_data_path,
)
from laya_simple_baseline_audit import read_laya_artifact


SCRIPT = Path(__file__).resolve()
ARTIFACT_SCHEMA = "td_market_data_laya_signal_ablation_artifact"
REPORT_SCHEMA = "td_market_data_laya_market_feature_ablation"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = validate_data_path(args.historical_inputs, "historical inputs")
    role_plan = validate_data_path(args.role_plan, "role plan")
    training = validate_data_path(args.training, "training artifact")
    cache = validate_data_path(args.model_cache, "Laya model cache")
    temp_root = validate_data_path(args.temp_root, "temporary workspace")
    _, baseline_report, baseline_manifest_hash = read_laya_artifact(args.laya_artifact)
    config_path = args.laya_config.resolve(strict=True)
    config, config_hash = load_strategy_config(config_path)
    if (baseline_report.get("laya_strategy_config_sha256") != config_hash
            or baseline_report.get("role_plan_sha256") != sha256_file(role_plan)
            or baseline_report.get("historical_input_manifest_sha256") != sha256_file(inputs / "manifest.json")):
        raise ValueError("ablation inputs differ from the frozen Laya Validation artifact")
    manifest_path = training / "manifest.json"
    training_manifest = json.loads(manifest_path.read_bytes())
    if baseline_report.get("training_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("ablation training artifact differs from the frozen Laya comparison")
    fee_bps = training_manifest["accounting"]["fee_bps"]
    slippage_bps = training_manifest["accounting"]["slippage_bps"]

    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["TMPDIR"] = str(temp_root)
    if not torch.cuda.is_available():
        raise ValueError("Laya ablation requires the frozen CUDA setup")
    total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    if not 0 < args.max_gpu_memory_mb <= total_gpu_mb:
        raise ValueError("GPU memory cap must be positive and no larger than device memory")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_mb / total_gpu_mb, device=0)
    torch.set_num_threads(4)
    from laya import Router

    router = Router(device="cuda", max_loaded=int(config["max_loaded_checkpoints"]),
                    revisions={config["checkpoint"]: config["checkpoint_revision"]})
    plan = verify_plan(role_plan, inputs, verify_input_artifact=False)
    windows = plan["roles"].get("validation", [])
    if not windows:
        raise ValueError("role plan has no Validation windows")
    results = []
    for window in windows:
        feature_path = inputs / window["features"]["path"]
        ledger_path = inputs / window["ledger_inputs"]["path"]
        for path, metadata in ((feature_path, window["features"]), (ledger_path, window["ledger_inputs"])):
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size != metadata["size_bytes"]
                    or sha256_file(path) != metadata["sha256"]):
                raise ValueError(f"Validation shard hash mismatch: {path}")
        counts: Counter[str] = Counter()

        def choose(feature: dict[str, Any], position: int):
            mask = feasible_action_mask(feature.get("mark_price"), position)
            if not mask[0] and not mask[2]:
                counts["flat_forced_by_feasibility_mask"] += 1
                return 1, None, mask
            state = make_state(feature["features"], feature.get("mark_price"), position)
            for key in state["market_features"]:
                state["market_features"][key] = 0.0
            answer = router.predict(state, {
                config["question_id"]: {
                    "type": "choice", "instructions": config["instructions"],
                    "criteria": config["criteria"],
                }
            }, model=config["checkpoint"]).get("answers", {}).get(config["question_id"], {})
            choice = answer.get("choice")
            if choice not in config["actions"]:
                raise ValueError(f"Laya returned unsupported ablation choice: {choice!r}")
            action = {"short": 0, "flat": 1, "long": 2}[choice]
            if not mask[action]:
                raise ValueError("Laya ablation selected an infeasible action")
            counts[choice] += 1
            counts["model_queries"] += 1
            if current_rss_mb() > args.max_memory_mb:
                raise ValueError(f"RSS exceeded configured {args.max_memory_mb} MiB cap")
            return action, None, mask

        _, metrics = simulate_episode(feature_path, ledger_path, window, choose,
                                      fee_bps, slippage_bps, args.max_memory_mb,
                                      collect_experiences=False)
        results.append({"strategy": config["strategy_id"], "input_ablation": "market_features_zeroed",
                        "role": "validation", "window_id": window["id"],
                        "action_counts": dict(sorted(counts.items())), "metrics": metrics})
    checkpoint, _ = checkpoint_inventory(cache, config["checkpoint_revision"])
    baseline_result = {item["window_id"]: item for item in baseline_report["laya_results"]}
    comparisons = []
    for row in results:
        observed = baseline_result[row["window_id"]]
        comparisons.append({
            "window_id": row["window_id"],
            "observed_action_counts": observed["action_counts"],
            "neutralized_action_counts": row["action_counts"],
            "observed_net_return_fraction": observed["metrics"]["net_return_fraction"],
            "neutralized_net_return_fraction": row["metrics"]["net_return_fraction"],
            "net_return_delta_fraction": row["metrics"]["net_return_fraction"] - observed["metrics"]["net_return_fraction"],
        })
    return {
        "schema": REPORT_SCHEMA, "version": 1, "evaluation_role": "validation",
        "ablation": "All seven supplied market return, realized volatility, and signed-volume features set to zero at every decision. Mark price, current position, risk limits, prompt, checkpoint, and ledger are unchanged.",
        "laya_artifact_manifest_sha256": baseline_manifest_hash,
        "laya_report_sha256": sha256_file(Path(args.laya_artifact) / "report.json"),
        "laya_strategy_config_sha256": config_hash,
        "checkpoint": checkpoint,
        "historical_input_manifest_sha256": sha256_file(inputs / "manifest.json"),
        "role_plan_sha256": sha256_file(role_plan),
        "training_manifest_sha256": sha256_file(manifest_path),
        "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
        "neutralized_results": results,
        "observed_vs_neutralized": comparisons,
        "summary": {
            "observed_mean_net_return_fraction": mean(item["observed_net_return_fraction"] for item in comparisons),
            "neutralized_mean_net_return_fraction": mean(item["neutralized_net_return_fraction"] for item in comparisons),
            "mean_net_return_delta_fraction": mean(item["net_return_delta_fraction"] for item in comparisons),
            "observed_action_counts": baseline_report["laya_results"][0]["action_counts"],
            "neutralized_action_counts": results[0]["action_counts"],
        },
        "ablation_code_sha256": sha256_file(SCRIPT),
        "limitations": [
            "This is an input ablation on the already-inspected Validation window, not independent evidence.",
            "Zeroing every market feature is a coarse intervention; it tests dependence on their supplied values collectively, not the value of each feature separately.",
            "The model still receives mark price, current position, and stated risk/cost constraints.",
            "Test and OOS are not read.",
        ],
    }


def write_artifact(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    output = output.absolute()
    validate_data_path(output.parent, "artifact parent")
    if output.exists() or output.is_symlink():
        raise ValueError("ablation output must be new and create-only")
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = stage / "report.json"
        report_path.write_bytes(canonical_bytes(report))
        manifest = {"schema": ARTIFACT_SCHEMA, "version": 1,
                    "report_sha256": sha256_file(report_path),
                    "laya_report_sha256": report["laya_report_sha256"],
                    "ablation_code_sha256": report["ablation_code_sha256"]}
        (stage / "manifest.json").write_bytes(canonical_bytes(manifest))
        for path in stage.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        try:
            os.chmod(stage, 0o755)
            shutil.rmtree(stage, ignore_errors=True)
        except OSError:
            pass
        raise


def verify_artifact(args: argparse.Namespace) -> dict[str, Any]:
    artifact = validate_data_path(args.artifact, "ablation artifact")
    if artifact.is_symlink() or not artifact.is_dir() or {p.name for p in artifact.iterdir()} != {"manifest.json", "report.json"}:
        raise ValueError("ablation artifact has missing or unmanifested files")
    manifest_path, report_path = artifact / "manifest.json", artifact / "report.json"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, report_path)):
        raise ValueError("ablation artifact files must be regular files")
    manifest = json.loads(manifest_path.read_bytes())
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("laya_report_sha256") != report.get("laya_report_sha256")
            or manifest.get("ablation_code_sha256") != report.get("ablation_code_sha256")):
        raise ValueError("ablation artifact manifest/report hash mismatch")
    if canonical_bytes(build_report(args)) != report_bytes:
        raise ValueError("recomputed feature-ablation results differ from the artifact")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--historical-inputs", type=Path, required=True)
        sub.add_argument("--role-plan", type=Path, required=True)
        sub.add_argument("--training", type=Path, required=True)
        sub.add_argument("--laya-artifact", type=Path, required=True)
        sub.add_argument("--laya-config", type=Path, required=True)
        sub.add_argument("--model-cache", type=Path, required=True)
        sub.add_argument("--temp-root", type=Path, required=True)
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
            s = report["summary"]
            print(f"observed Laya: {s['observed_mean_net_return_fraction']:.6%}")
            print(f"market features zeroed: {s['neutralized_mean_net_return_fraction']:.6%}")
            print(f"return delta: {s['mean_net_return_delta_fraction']:.6%}")
            print(f"neutralized actions: {s['neutralized_action_counts']}")
            print(f"artifact: {args.output.resolve()}")
            print(f"report sha256: {manifest['report_sha256']}")
        else:
            manifest = verify_artifact(args)
            print("Laya feature ablation recomputed and verified")
            print(f"report sha256: {manifest['report_sha256']}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            torch.cuda.OutOfMemoryError) as error:
        raise SystemExit(f"Laya signal ablation rejected: {error}") from error


if __name__ == "__main__":
    main()
