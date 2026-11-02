#!/usr/bin/env python3
"""Compare deterministic Validation policies with a stronger flat-action prior."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import numpy as np

from td_market_data_rl.analysis.validation_matrix import _initialize_evaluation_worker
from td_market_data_rl.factors import registry_sha256, selected_factor_values
from td_market_data_rl.historical import sha256_file, verify_plan
from td_market_data_rl.learner import simulate_episode
from td_market_data_rl.policy import apply_action_mask, load_actor
from td_market_data_rl.contracts import feasible_action_mask
from td_market_data_rl.sweep import verify_sweep


SCHEMA = "td_market_data_usdm_rl_conservative_policy_validation"
FLAT_PRIOR_MULTIPLIERS = (1.0, 4.0, 9.0)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    import torch

    input_root = Path(task["historical_inputs"])
    training_path = Path(task["training_path"])
    model = json.loads((training_path / "model.json").read_bytes())
    manifest = json.loads((training_path / "manifest.json").read_bytes())
    factor_names = model["feature_names"]
    actor = load_actor(model, len(factor_names) + 1, device="cpu")
    means = model["normalizer"]["mean"]
    scales = model["normalizer"]["scale"]
    multiplier = float(task["flat_prior_multiplier"])
    if multiplier < 1 or not math.isfinite(multiplier):
        raise ValueError("flat-action prior multiplier must be finite and at least one")
    window = task["window"]
    action_counts = [0, 0, 0]

    def choose(feature: dict[str, Any], position: int) -> tuple[int, None]:
        _, values = selected_factor_values(feature["features"], model["factor_set_id"])
        vector = [(value - means[index]) / scales[index]
                  for index, value in enumerate(values)] + [position]
        with torch.inference_mode():
            logits = actor(torch.tensor(vector, dtype=torch.float32))
            # A multiplier m adds log(m) to the flat-action logit. At equal
            # logits this gives P(flat)=m/(m+2), without overriding strong signals.
            logits[1] += math.log(multiplier)
            logits = apply_action_mask(logits, feasible_action_mask(feature.get("mark_price"), position))
            action = int(torch.argmax(logits).item())
        action_counts[action] += 1
        return action, None

    features_path = input_root / window["features"]["path"]
    ledger_path = input_root / window["ledger_inputs"]["path"]
    _, metrics = simulate_episode(
        features_path, ledger_path, window, choose,
        manifest["accounting"]["fee_bps"], manifest["accounting"]["slippage_bps"],
        4096, collect_experiences=False)
    expected_minutes = window["minute_count"]
    if metrics["minutes"] != expected_minutes:
        raise ValueError("Validation replay row count differs from frozen role plan")
    return {"factor_set_id": manifest["factor_set_id"], "training_seed": manifest["seed"],
            "training_run": task["training_relative_path"],
            "training_manifest_sha256": task["training_manifest_sha256"],
            "model_sha256": manifest["model_sha256"],
            "flat_prior_multiplier": multiplier,
            "flat_probability_at_equal_logits": multiplier / (multiplier + 2.0),
            "inference": "deterministic_argmax",
            "actions_short_flat_long": action_counts,
            "window_id": window["id"], "metrics": metrics}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["flat_prior_multiplier"]), []).append(row)
    by_prior = {}
    by_factor_and_prior = {}
    for prior, entries in sorted(groups.items(), key=lambda item: float(item[0])):
        pnls = [item["metrics"]["net_pnl_usdt"] for item in entries]
        by_prior[prior] = {"observation_count": len(entries),
                           "mean_net_pnl_usdt": mean(pnls),
                           "positive_observations": sum(value > 0 for value in pnls),
                           "mean_fills": mean(item["metrics"]["fills"] for item in entries),
                           "mean_fees_usdt": mean(item["metrics"]["fees_usdt"] for item in entries),
                           "mean_slippage_usdt": mean(item["metrics"]["slippage_usdt"] for item in entries),
                           "beats_noop_by_mean": mean(pnls) > 0.0}
        suites = sorted({item["factor_set_id"] for item in entries})
        for suite in suites:
            suite_rows = [item for item in entries if item["factor_set_id"] == suite]
            key = f"{suite}|{prior}"
            values = [item["metrics"]["net_pnl_usdt"] for item in suite_rows]
            by_factor_and_prior[key] = {
                "factor_set_id": suite, "flat_prior_multiplier": float(prior),
                "observation_count": len(values), "mean_net_pnl_usdt": mean(values),
                "positive_observations": sum(value > 0 for value in values),
                "mean_fills": mean(item["metrics"]["fills"] for item in suite_rows),
                "mean_fees_usdt": mean(item["metrics"]["fees_usdt"] for item in suite_rows),
                "mean_slippage_usdt": mean(item["metrics"]["slippage_usdt"] for item in suite_rows),
            }
    return {"noop_net_pnl_usdt": 0.0, "summary_by_flat_prior": by_prior,
            "summary_by_factor_and_flat_prior": by_factor_and_prior}


def _build_report(source_sweep: Path, historical_inputs: Path,
                  role_plan_path: Path) -> dict[str, Any]:
    source = source_sweep.resolve(strict=True)
    input_root = historical_inputs.resolve(strict=True)
    role_plan = role_plan_path.resolve(strict=True)
    verify_sweep(SimpleNamespace(artifact=source, historical_inputs=input_root,
                                 role_plan=role_plan))
    plan = verify_plan(role_plan, input_root, verify_input_artifact=False)
    validation_windows = plan["roles"].get("validation", [])
    if not validation_windows:
        raise ValueError("frozen role plan has no Validation windows")
    source_report = json.loads((source / "report.json").read_bytes())
    tasks = []
    run_inventory = []
    for item in source_report["training_runs"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe training run path in source sweep")
        training_path = source / relative
        if training_path.is_symlink() or not training_path.is_dir():
            raise ValueError("source sweep training run is not a regular directory")
        manifest_path = training_path / "manifest.json"
        model_path = training_path / "model.json"
        manifest = json.loads(manifest_path.read_bytes())
        if (sha256_file(manifest_path) != item["manifest_sha256"]
                or sha256_file(model_path) != item["model_sha256"]
                or manifest.get("factor_set_id") != item["factor_set_id"]
                or manifest.get("seed") != item["seed"]):
            raise ValueError("source sweep training run hash or identity mismatch")
        run_inventory.append({"path": relative.as_posix(),
                              "manifest_sha256": sha256_file(manifest_path),
                              "model_sha256": sha256_file(model_path),
                              "factor_set_id": item["factor_set_id"], "seed": item["seed"]})
        for window in validation_windows:
            for path_key in ("features", "ledger_inputs"):
                entry = window[path_key]
                path = input_root / entry["path"]
                if (path.is_symlink() or not path.is_file()
                        or path.stat().st_size != entry["size_bytes"]
                        or sha256_file(path) != entry["sha256"]):
                    raise ValueError(f"Validation shard failed role-plan verification: {path}")
            for multiplier in FLAT_PRIOR_MULTIPLIERS:
                tasks.append({"historical_inputs": str(input_root),
                              "training_path": str(training_path),
                              "training_relative_path": relative.as_posix(),
                              "training_manifest_sha256": sha256_file(manifest_path),
                              "window": window,
                              "flat_prior_multiplier": multiplier})
    workers = min(4, len(tasks))
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_evaluation_worker) as pool:
        rows = list(pool.map(_evaluate_task, tasks))
    rows.sort(key=lambda item: (item["factor_set_id"], item["training_seed"],
                                item["window_id"], item["flat_prior_multiplier"]))
    return {"schema": SCHEMA, "version": 1,
            "evaluation_role": "validation",
            "inference_protocol": "deterministic_argmax_flat_logit_prior_v1",
            "flat_prior_multipliers": list(FLAT_PRIOR_MULTIPLIERS),
            "factor_registry_sha256": registry_sha256(),
            "historical_input_manifest_sha256": sha256_file(input_root / "manifest.json"),
            "role_plan_sha256": sha256_file(role_plan),
            "role_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
            "source_sweep": {"path": str(source),
                             "manifest_sha256": sha256_file(source / "manifest.json"),
                             "report_sha256": sha256_file(source / "report.json")},
            "validation_windows": [{"id": window["id"],
                                    "start_ns": window["start_ns"],
                                    "end_ns": window["end_ns"],
                                    "features_sha256": window["features"]["sha256"],
                                    "ledger_sha256": window["ledger_inputs"]["sha256"]}
                                   for window in validation_windows],
            "training_runs": run_inventory,
            "parallel_workers": workers,
            "cost_scenario": source_report["cost_scenario"],
            "noop_baseline_net_pnl_usdt": 0.0,
            "results": rows, **_summary(rows),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "accounting_code_sha256": sha256_file(Path(__file__).resolve().parents[1]
                                                   / "src/td_market_data_rl/learner.py"),
            "limitations": [
                "This is an inference-time prior diagnostic; the frozen models were trained without this prior.",
                "Each fixed Train-seed model is replayed deterministically on every Validation window.",
                "NOOP has zero PnL by definition under the frozen account protocol.",
                "This is historical research, not evidence of future profitability.",
                "Test and OOS windows are not read.",
            ]}


def create_artifact(args: Any) -> None:
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("artifact output must be a new create-only directory")
    resolved = output.resolve()
    data_root = Path("/mnt/data").resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[2]
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError("artifact output must be stored under /mnt/data")
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError("artifact output must be outside the source repository")
    if any(resolved == item or item in resolved.parents for item in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("system temporary directories are not allowed")
    report = _build_report(args.source_sweep, args.historical_inputs, args.role_plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = stage / "report.json"
        report_path.write_bytes(json.dumps(report, sort_keys=True, indent=2,
                                           allow_nan=False).encode() + b"\n")
        manifest = {"schema": f"{SCHEMA}_artifact", "version": 1,
                    "report_sha256": sha256_file(report_path),
                    "source_sweep_manifest_sha256": report["source_sweep"]["manifest_sha256"],
                    "historical_input_manifest_sha256": report["historical_input_manifest_sha256"],
                    "role_plan_sha256": report["role_plan_sha256"],
                    "script_sha256": report["script_sha256"]}
        (stage / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True,
                                                         indent=2).encode() + b"\n")
        for path in stage.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        try:
            for path in stage.iterdir():
                os.chmod(path, 0o644)
            os.chmod(stage, 0o755)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"Conservative Validation artifact created: {output}")
    print(f"models={len(report['training_runs'])} prior_levels={len(FLAT_PRIOR_MULTIPLIERS)} "
          f"results={len(report['results'])}; Test/OOS not read")


def verify_artifact(args: Any) -> None:
    root = args.artifact.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact must be a regular directory")
    manifest = json.loads((root / "manifest.json").read_bytes())
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != f"{SCHEMA}_artifact"
            or manifest.get("version") != 1
            or report.get("schema") != SCHEMA
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("script_sha256") != sha256_file(Path(__file__).resolve())):
        raise ValueError("artifact schema or hash mismatch")
    recomputed = _build_report(args.source_sweep, args.historical_inputs, args.role_plan)
    if canonical(recomputed) != canonical(report):
        raise ValueError("recomputed conservative Validation results differ")
    print("Conservative Validation artifact verified by recomputing every replay")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    verify = commands.add_parser("verify")
    for command in (create, verify):
        command.add_argument("--source-sweep", type=Path, required=True)
        command.add_argument("--historical-inputs", type=Path, required=True)
        command.add_argument("--role-plan", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            create_artifact(args)
        else:
            verify_artifact(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"conservative policy validation rejected: {error}") from error


if __name__ == "__main__":
    main()
