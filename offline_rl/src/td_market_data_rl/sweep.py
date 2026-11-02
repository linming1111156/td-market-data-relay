from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

from .evaluator import evaluate
from .factors import suites, registry_sha256
from .historical import verify_plan
from .learner import train


SWEEP_SCHEMA = "td_market_data_usdm_rl_validation_factor_sweep"
BASELINE_STRATEGIES = ["noop_flat_v1", "buy_and_hold_long_v1", "mark_momentum_15m_sign_v1"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def create_sweep(args: Any) -> dict[str, Any]:
    input_root = args.historical_inputs.resolve(strict=True)
    role_plan = args.role_plan.resolve(strict=True)
    plan = verify_plan(role_plan, input_root)
    if "validation" not in plan["roles"]:
        raise ValueError("factor sweep requires a frozen Validation role")
    factor_sets = args.factor_sets or sorted(suites())
    if not factor_sets or len(set(factor_sets)) != len(factor_sets):
        raise ValueError("factor sets must be nonempty and unique")
    unknown = sorted(set(factor_sets) - set(suites()))
    if unknown:
        raise ValueError(f"unknown factor sets: {unknown}")
    training_seeds = args.training_seeds
    evaluation_seeds = args.evaluation_seeds
    if not training_seeds or len(set(training_seeds)) != len(training_seeds):
        raise ValueError("training seeds must be nonempty and unique")
    if not evaluation_seeds or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("evaluation seeds must be nonempty and unique")
    output = args.output.absolute()
    data_root = Path("/mnt/data").resolve(strict=True)
    resolved = output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError("Validation sweep output must be new and create-only")
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError("Validation sweep output must be stored under /mnt/data")
    repo_root = Path(__file__).resolve().parents[3]
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError("Validation sweep output must be outside the source repository")
    if any(resolved == item or item in resolved.parents for item in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("Validation sweep output must not use system temporary storage")
    if not 1 <= args.updates <= 20:
        raise ValueError("updates must be between 1 and 20")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        runs_root = stage / "training_runs"
        runs_root.mkdir()
        rl_results = []
        training_inventory = []
        first_run = None
        for factor_set in factor_sets:
            for seed in training_seeds:
                run_id = f"{factor_set}-seed-{seed}"
                run_path = runs_root / run_id
                train_args = SimpleNamespace(
                    features=None, dataset=None, historical_inputs=input_root,
                    role_plan=role_plan, output=run_path, device=args.device,
                    updates=args.updates, seed=seed, factor_set=factor_set,
                    fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                    max_memory_mb=args.max_memory_mb,
                    max_gpu_memory_mb=args.max_gpu_memory_mb)
                run_manifest = train(train_args)
                if first_run is None:
                    first_run = run_path
                one_seed = [seed]
                report = evaluate({
                    "features": None, "dataset": None, "historical_inputs": input_root,
                    "role_plan": role_plan, "training": run_path,
                    "seeds": one_seed, "evaluation_role": "validation",
                    "strategies": ["rl_policy_v1"],
                })
                rl_results.append({
                    "factor_set_id": factor_set,
                    "training_seed": seed,
                    "evaluation_seed": seed,
                    "training_run": f"training_runs/{run_id}",
                    "training_manifest_sha256": sha256_file(run_path / "manifest.json"),
                    "evaluation_code_sha256": report["evaluation_code_sha256"],
                    "model_sha256": run_manifest["model_sha256"],
                    "metrics": report["results"][0]["metrics"],
                })
                training_inventory.append({
                    "path": f"training_runs/{run_id}",
                    "factor_set_id": factor_set,
                    "seed": seed,
                    "manifest_sha256": sha256_file(run_path / "manifest.json"),
                    "model_sha256": run_manifest["model_sha256"],
                    "trace_sha256": run_manifest["trace_sha256"],
                    "trainer_code_sha256": run_manifest["trainer_code_sha256"],
                })
        if first_run is None:
            raise ValueError("factor sweep produced no training runs")
        baseline_report = evaluate({
            "features": None, "dataset": None, "historical_inputs": input_root,
            "role_plan": role_plan, "training": first_run,
            "seeds": evaluation_seeds, "evaluation_role": "validation",
            "strategies": BASELINE_STRATEGIES,
        })
        summaries = {}
        for factor_set in factor_sets:
            rows = [item for item in rl_results if item["factor_set_id"] == factor_set]
            returns = [item["metrics"]["net_return_fraction"] for item in rows]
            drawdowns = [item["metrics"]["max_drawdown_fraction"] for item in rows]
            summaries[factor_set] = {
                "training_seed_count": len(rows),
                "mean_validation_net_return_fraction": mean(returns),
                "population_std_validation_net_return_fraction": pstdev(returns) if len(returns) > 1 else 0.0,
                "worst_validation_drawdown_fraction": min(drawdowns),
                "all_runs_below_noop": all(value < 0 for value in returns),
            }
        report = {
            "schema": SWEEP_SCHEMA,
            "version": 1,
            "evaluation_role": "validation",
            "historical_input_manifest_sha256": sha256_file(input_root / "manifest.json"),
            "role_plan_sha256": sha256_file(role_plan),
            "role_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
            "factor_registry_sha256": registry_sha256(),
            "sweep_code_sha256": sha256_file(Path(__file__).resolve()),
            "evaluation_code_sha256": baseline_report["evaluation_code_sha256"],
            "baseline_registry_sha256": baseline_report["baseline_registry_sha256"],
            "accounting": baseline_report["accounting"],
            "accounting_protocol_sha256": baseline_report["accounting_protocol_sha256"],
            "factor_sets": factor_sets,
            "training_seeds": training_seeds,
            "evaluation_seeds": evaluation_seeds,
            "updates": args.updates,
            "device_requested": args.device,
            "cost_scenario": {"fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps},
            "limits": {"max_memory_mb": args.max_memory_mb,
                       "max_gpu_memory_mb": args.max_gpu_memory_mb},
            "training_runs": training_inventory,
            "rl_validation_results": rl_results,
            "baseline_validation_results": baseline_report["results"],
            "summary_by_factor_set": summaries,
            "limitations": [
                "This is historical research, not evidence of future profitability.",
                "Each training seed is evaluated with the same numeric action-sampling seed.",
                "Execution uses the frozen next-minute aggregate-trade accounting model.",
                "Validation results are for selection only; Test and OOS are not read by this sweep.",
            ],
        }
        report_path = stage / "report.json"
        report_path.write_bytes(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        file_inventory = []
        for path in sorted(p for p in stage.rglob("*") if p.is_file()):
            file_inventory.append({"path": path.relative_to(stage).as_posix(),
                                   "size_bytes": path.stat().st_size,
                                   "sha256": sha256_file(path)})
        manifest = {"schema": "td_market_data_usdm_rl_validation_sweep_artifact",
                    "version": 1,
                    "report_sha256": sha256_file(report_path),
                    "file_inventory": file_inventory,
                    "historical_input_manifest_sha256": report["historical_input_manifest_sha256"],
                    "role_plan_sha256": report["role_plan_sha256"],
                    "factor_registry_sha256": registry_sha256()}
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        for path in stage.rglob("*"):
            if path.is_file():
                os.chmod(path, 0o444)
        for path in sorted((p for p in stage.rglob("*") if p.is_dir()),
                           key=lambda item: len(item.parts), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        return manifest
    except BaseException:
        try:
            for path in stage.rglob("*"):
                if path.is_file():
                    os.chmod(path, 0o644)
            for path in stage.rglob("*"):
                if path.is_dir():
                    os.chmod(path, 0o755)
            os.chmod(stage, 0o755)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_sweep(args: Any) -> dict[str, Any]:
    root = args.artifact.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Validation sweep artifact must be a regular directory")
    manifest = json.loads((root / "manifest.json").read_bytes())
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != "td_market_data_usdm_rl_validation_sweep_artifact"
            or manifest.get("version") != 1 or report.get("schema") != SWEEP_SCHEMA
            or report.get("evaluation_role") != "validation"
            or manifest.get("report_sha256") != sha256_file(report_path)):
        raise ValueError("unsupported or hash-mismatched Validation sweep artifact")
    expected_inventory = []
    for path in sorted(p for p in root.rglob("*")
                        if p.is_file() and p.relative_to(root) != Path("manifest.json")):
        expected_inventory.append({"path": path.relative_to(root).as_posix(),
                                  "size_bytes": path.stat().st_size,
                                  "sha256": sha256_file(path)})
    if expected_inventory != manifest.get("file_inventory"):
        raise ValueError("Validation sweep file inventory is missing, changed, or unmanifested")

    input_root = args.historical_inputs.resolve(strict=True)
    role_plan = args.role_plan.resolve(strict=True)
    plan = verify_plan(role_plan, input_root, verify_input_artifact=False)
    input_hash = sha256_file(input_root / "manifest.json")
    role_plan_hash = sha256_file(role_plan)
    if (report.get("historical_input_manifest_sha256") != input_hash
            or report.get("role_plan_sha256") != role_plan_hash
            or report.get("role_plan_fingerprint_sha256") != plan["plan_fingerprint_sha256"]
            or report.get("factor_registry_sha256") != registry_sha256()
            or report.get("sweep_code_sha256") != sha256_file(Path(__file__).resolve())
            or report.get("evaluation_code_sha256") is None
            or manifest.get("historical_input_manifest_sha256") != input_hash
            or manifest.get("role_plan_sha256") != role_plan_hash):
        raise ValueError("Validation sweep source binding differs from supplied inputs")

    from .cli import verify_run

    run_by_path = {}
    for item in report.get("training_runs", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe training run path in sweep report")
        run_path = root / relative
        verified = verify_run(SimpleNamespace(
            artifact=run_path, historical_inputs=input_root, role_plan=role_plan,
            dataset=None, features=None))
        if (verified.get("factor_set_id") not in report["factor_sets"]
                or verified.get("seed") not in report["training_seeds"]
                or verified.get("role_plan_sha256") != role_plan_hash
                or verified.get("model_sha256") != item.get("model_sha256")
                or verified.get("trace_sha256") != item.get("trace_sha256")
                or verified.get("trainer_code_sha256") != item.get("trainer_code_sha256")
                or sha256_file(run_path / "manifest.json") != item.get("manifest_sha256")
                or verified.get("factor_set_id") != item.get("factor_set_id")
                or verified.get("seed") != item.get("seed")):
            raise ValueError("training run does not match sweep inventory")
        run_by_path[relative.as_posix()] = run_path

    expected_combinations = {(suite, seed) for suite in report["factor_sets"]
                             for seed in report["training_seeds"]}
    actual_combinations = {(item["factor_set_id"], item["seed"])
                           for item in report.get("training_runs", [])}
    if actual_combinations != expected_combinations:
        raise ValueError("sweep is missing or duplicating factor/seed training runs")
    result_combinations = {(item.get("factor_set_id"), item.get("training_seed"))
                           for item in report.get("rl_validation_results", [])}
    if (result_combinations != expected_combinations
            or len(report.get("rl_validation_results", [])) != len(expected_combinations)
            or any(item.get("evaluation_seed") != item.get("training_seed")
                   for item in report["rl_validation_results"])):
        raise ValueError("sweep RL results do not cover every fixed training seed exactly once")
    for item in report.get("rl_validation_results", []):
        run_path = run_by_path.get(item["training_run"])
        if run_path is None:
            raise ValueError("RL result references an unknown training run")
        evaluation = evaluate({
            "features": None, "dataset": None, "historical_inputs": input_root,
            "role_plan": role_plan, "training": run_path,
            "seeds": [item["evaluation_seed"]], "evaluation_role": "validation",
            "strategies": ["rl_policy_v1"],
        })
        if (evaluation["evaluation_code_sha256"] != report.get("evaluation_code_sha256")
                or item.get("evaluation_code_sha256") != evaluation["evaluation_code_sha256"]):
            raise ValueError("Validation evaluator code hash differs from sweep report")
        recomputed = evaluation["results"][0]["metrics"]
        if canonical(recomputed) != canonical(item["metrics"]):
            raise ValueError("recomputed RL Validation metrics differ from sweep report")

    first_run = next(iter(run_by_path.values()))
    baseline_report = evaluate({
        "features": None, "dataset": None, "historical_inputs": input_root,
        "role_plan": role_plan, "training": first_run,
        "seeds": report["evaluation_seeds"], "evaluation_role": "validation",
        "strategies": BASELINE_STRATEGIES,
    })
    if (baseline_report["evaluation_code_sha256"] != report.get("evaluation_code_sha256")
            or baseline_report["baseline_registry_sha256"] != report.get("baseline_registry_sha256")
            or canonical(baseline_report["accounting"]) != canonical(report.get("accounting"))
            or canonical(baseline_report["results"]) != canonical(report.get("baseline_validation_results"))):
        raise ValueError("recomputed baseline Validation metrics differ from sweep report")

    summaries = {}
    for factor_set in report["factor_sets"]:
        rows = [item for item in report["rl_validation_results"]
                if item["factor_set_id"] == factor_set]
        returns = [item["metrics"]["net_return_fraction"] for item in rows]
        drawdowns = [item["metrics"]["max_drawdown_fraction"] for item in rows]
        summaries[factor_set] = {
            "training_seed_count": len(rows),
            "mean_validation_net_return_fraction": mean(returns),
            "population_std_validation_net_return_fraction": pstdev(returns) if len(returns) > 1 else 0.0,
            "worst_validation_drawdown_fraction": min(drawdowns),
            "all_runs_below_noop": all(value < 0 for value in returns),
        }
    if canonical(summaries) != canonical(report.get("summary_by_factor_set")):
        raise ValueError("factor-suite Validation summary mismatch")
    return manifest
