from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace
from typing import Any

from ..evaluator import evaluate
from ..factors import registry_sha256
from ..historical import verify_plan
from ..sweep import verify_sweep


MATRIX_SCHEMA = "td_market_data_usdm_rl_validation_seed_matrix"
MAX_EVALUATION_WORKERS = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _initialize_evaluation_worker() -> None:
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _evaluate_task(task: dict[str, Any]) -> dict[str, Any]:
    return evaluate({"features": None, "dataset": None,
                     "historical_inputs": task["historical_inputs"],
                     "role_plan": task["role_plan"], "training": task["training"],
                     "seeds": task["seeds"], "evaluation_role": "validation",
                     "strategies": ["rl_policy_v1"]})


def _summary(results: list[dict[str, Any]], baseline_results: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_returns = [item["metrics"]["net_return_fraction"] for item in baseline_results
                        if item["strategy"] == "noop_flat_v1"]
    noop_mean = mean(baseline_returns)
    by_suite: dict[str, list[float]] = {}
    for item in results:
        by_suite.setdefault(item["factor_set_id"], []).append(
            item["metrics"]["net_return_fraction"])
    summaries = {}
    for suite, returns in sorted(by_suite.items()):
        summaries[suite] = {
            "validation_observation_count": len(returns),
            "training_seed_count": len({item["training_seed"] for item in results
                                         if item["factor_set_id"] == suite}),
            "evaluation_seed_count": len({item["evaluation_seed"] for item in results
                                           if item["factor_set_id"] == suite}),
            "mean_validation_net_return_fraction": mean(returns),
            "population_std_validation_net_return_fraction": pstdev(returns) if len(returns) > 1 else 0.0,
            "worst_validation_drawdown_fraction": min(
                item["metrics"]["max_drawdown_fraction"] for item in results
                if item["factor_set_id"] == suite),
            "all_observations_below_noop": all(value < noop_mean for value in returns),
        }
    best_suite = max(summaries, key=lambda suite: summaries[suite]["mean_validation_net_return_fraction"])
    best_mean = summaries[best_suite]["mean_validation_net_return_fraction"]
    return {"noop_mean_validation_net_return_fraction": noop_mean,
            "summary_by_factor_set": summaries,
            "best_factor_set_by_mean_validation_return": best_suite,
            "best_factor_set_mean_validation_net_return_fraction": best_mean,
            "best_factor_set_beats_noop": best_mean > noop_mean}


def _run_matrix(source: Path, input_root: Path, role_plan: Path) -> dict[str, Any]:
    source_report = json.loads((source / "report.json").read_bytes())
    run_by_key = {(item["factor_set_id"], item["seed"]): item for item in source_report["training_runs"]}
    tasks = []
    task_metadata = []
    for factor_set in source_report["factor_sets"]:
        for training_seed in source_report["training_seeds"]:
            run_item = run_by_key[(factor_set, training_seed)]
            tasks.append({"historical_inputs": input_root, "role_plan": role_plan,
                          "training": source / run_item["path"],
                          "seeds": source_report["evaluation_seeds"]})
            task_metadata.append((factor_set, training_seed, run_item))
    results = []
    worker_count = min(MAX_EVALUATION_WORKERS, len(tasks))
    with ProcessPoolExecutor(max_workers=worker_count,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_evaluation_worker) as executor:
        reports = list(executor.map(_evaluate_task, tasks))
    for (factor_set, training_seed, run_item), report in zip(task_metadata, reports, strict=True):
        for evaluation_row in report["results"]:
                results.append({"factor_set_id": factor_set,
                                "training_seed": training_seed,
                                "evaluation_seed": evaluation_row["seed"],
                                "training_run": run_item["path"],
                                "training_manifest_sha256": run_item["manifest_sha256"],
                                "model_sha256": run_item["model_sha256"],
                                "evaluation_code_sha256": report["evaluation_code_sha256"],
                                "window_id": evaluation_row["window_id"],
                                "metrics": evaluation_row["metrics"]})
    summary = _summary(results, source_report["baseline_validation_results"])
    return {"schema": MATRIX_SCHEMA, "version": 1,
            "evaluation_role": "validation",
            "historical_input_manifest_sha256": source_report["historical_input_manifest_sha256"],
            "role_plan_sha256": source_report["role_plan_sha256"],
            "role_plan_fingerprint_sha256": source_report["role_plan_fingerprint_sha256"],
            "factor_registry_sha256": registry_sha256(),
            "evaluation_code_sha256": source_report["evaluation_code_sha256"],
            "seed_matrix_code_sha256": sha256_file(Path(__file__).resolve()),
            "source_sweep": {"path": str(source),
                             "manifest_sha256": sha256_file(source / "manifest.json"),
                             "report_sha256": sha256_file(source / "report.json"),
                             "training_run_count": len(source_report["training_runs"])},
            "factor_sets": source_report["factor_sets"],
            "training_seeds": source_report["training_seeds"],
            "evaluation_seeds": source_report["evaluation_seeds"],
            "parallel_evaluation_workers": worker_count,
            "updates": source_report["updates"],
            "cost_scenario": source_report["cost_scenario"],
            "accounting": source_report["accounting"],
            "baseline_registry_sha256": source_report["baseline_registry_sha256"],
            "baseline_validation_results": source_report["baseline_validation_results"],
            "validation_results": results,
            **summary,
            "limitations": [
                "Every frozen Train-seed model is evaluated with every listed Validation action-sampling seed.",
                "This is historical research, not evidence of future profitability.",
                "Test and OOS windows are not read by this matrix.",
                "Execution uses the frozen next-minute aggregate-trade accounting model.",
            ]}


def create_matrix(args: Any) -> dict[str, Any]:
    source = args.source_sweep.resolve(strict=True)
    input_root = args.historical_inputs.resolve(strict=True)
    role_plan = args.role_plan.resolve(strict=True)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("source Validation sweep must be a regular directory")
    verify_sweep(SimpleNamespace(artifact=source, historical_inputs=input_root, role_plan=role_plan))
    output = args.output.absolute()
    data_root = Path("/mnt/data").resolve(strict=True)
    resolved = output.resolve()
    repo_root = Path(__file__).resolve().parents[3]
    if output.exists() or output.is_symlink():
        raise ValueError("Validation matrix output must be new and create-only")
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError("Validation matrix output must be stored under /mnt/data")
    if resolved == repo_root or repo_root in resolved.parents:
        raise ValueError("Validation matrix output must be outside the source repository")
    if any(resolved == item or item in resolved.parents for item in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("Validation matrix output must not use system temporary storage")
    plan = verify_plan(role_plan, input_root, verify_input_artifact=False)
    if "validation" not in plan["roles"]:
        raise ValueError("Validation seed matrix requires a frozen Validation role")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report = _run_matrix(source, input_root, role_plan)
        report_path = stage / "report.json"
        report_path.write_bytes(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        inventory = [{"path": path.relative_to(stage).as_posix(),
                      "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                     for path in sorted(stage.rglob("*")) if path.is_file()]
        manifest = {"schema": "td_market_data_usdm_rl_validation_seed_matrix_artifact",
                    "version": 1, "report_sha256": sha256_file(report_path),
                    "file_inventory": inventory,
                    "source_sweep_manifest_sha256": report["source_sweep"]["manifest_sha256"],
                    "historical_input_manifest_sha256": report["historical_input_manifest_sha256"],
                    "role_plan_sha256": report["role_plan_sha256"]}
        (stage / "manifest.json").write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        for path in stage.rglob("*"):
            if path.is_file(): os.chmod(path, 0o444)
        for path in sorted((item for item in stage.rglob("*") if item.is_dir()),
                           key=lambda item: len(item.parts), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        return manifest
    except BaseException:
        try:
            for path in stage.rglob("*"):
                if path.is_file(): os.chmod(path, 0o644)
                elif path.is_dir(): os.chmod(path, 0o755)
            os.chmod(stage, 0o755)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_matrix(args: Any) -> dict[str, Any]:
    root = args.artifact.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Validation matrix artifact must be a regular directory")
    manifest = json.loads((root / "manifest.json").read_bytes())
    report_path = root / "report.json"
    report = json.loads(report_path.read_bytes())
    inventory = [{"path": path.relative_to(root).as_posix(),
                  "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                 for path in sorted(root.rglob("*"))
                 if path.is_file() and path.relative_to(root) != Path("manifest.json")]
    if (manifest.get("schema") != "td_market_data_usdm_rl_validation_seed_matrix_artifact"
            or manifest.get("version") != 1 or report.get("schema") != MATRIX_SCHEMA
            or report.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or inventory != manifest.get("file_inventory")):
        raise ValueError("Validation seed matrix artifact is incomplete or hash-mismatched")
    source = Path(report["source_sweep"]["path"]).resolve(strict=True)
    input_root = args.historical_inputs.resolve(strict=True)
    role_plan = args.role_plan.resolve(strict=True)
    verify_sweep(SimpleNamespace(artifact=source, historical_inputs=input_root, role_plan=role_plan))
    if (sha256_file(source / "manifest.json") != report["source_sweep"]["manifest_sha256"]
            or sha256_file(source / "report.json") != report["source_sweep"]["report_sha256"]
            or sha256_file(Path(__file__).resolve()) != report.get("seed_matrix_code_sha256")
            or report.get("historical_input_manifest_sha256") != sha256_file(input_root / "manifest.json")
            or report.get("role_plan_sha256") != sha256_file(role_plan)
            or report.get("factor_registry_sha256") != registry_sha256()):
        raise ValueError("Validation seed matrix source binding differs")
    plan = verify_plan(role_plan, input_root, verify_input_artifact=False)
    if report.get("role_plan_fingerprint_sha256") != plan["plan_fingerprint_sha256"]:
        raise ValueError("Validation seed matrix role-plan fingerprint differs")

    from ..cli import verify_run
    source_report = json.loads((source / "report.json").read_bytes())
    run_by_key = {(item["factor_set_id"], item["seed"]): item for item in source_report["training_runs"]}
    expected = {(suite, train_seed, eval_seed)
                for suite in report["factor_sets"]
                for train_seed in report["training_seeds"]
                for eval_seed in report["evaluation_seeds"]}
    expected_windows = {(suite, train_seed, eval_seed, window["id"])
                        for suite, train_seed, eval_seed in expected
                        for window in plan["roles"]["validation"]}
    actual = {(item["factor_set_id"], item["training_seed"], item["evaluation_seed"], item["window_id"])
              for item in report["validation_results"]}
    if actual != expected_windows or len(report["validation_results"]) != len(expected_windows):
        raise ValueError("Validation matrix does not cover each training/evaluation seed combination")
    evaluation_jobs = []
    for key, run_item in run_by_key.items():
        training = source / run_item["path"]
        verify_run(SimpleNamespace(artifact=training, historical_inputs=input_root,
                                   role_plan=role_plan, dataset=None, features=None))
        if sha256_file(training / "manifest.json") != run_item["manifest_sha256"]:
            raise ValueError("source sweep training manifest hash mismatch")
        evaluation_jobs.append((key, run_item, {
            "historical_inputs": input_root, "role_plan": role_plan,
            "training": training, "seeds": report["evaluation_seeds"]}))
    worker_count = min(MAX_EVALUATION_WORKERS, len(evaluation_jobs))
    if report.get("parallel_evaluation_workers") != worker_count:
        raise ValueError("Validation matrix parallel worker count differs from the bounded policy")
    with ProcessPoolExecutor(max_workers=worker_count,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_initialize_evaluation_worker) as executor:
        recomputed_reports = list(executor.map(_evaluate_task,
                                               [item[2] for item in evaluation_jobs]))
    evaluation_code_hashes = set()
    for (key, run_item, _task), evaluation in zip(evaluation_jobs, recomputed_reports, strict=True):
        evaluation_code_hashes.add(evaluation["evaluation_code_sha256"])
        by_seed_window = {(row["seed"], row["window_id"]): row for row in evaluation["results"]}
        for eval_seed in report["evaluation_seeds"]:
            matches = [item for item in report["validation_results"]
                       if (item["factor_set_id"], item["training_seed"], item["evaluation_seed"])
                       == (key[0], key[1], eval_seed)]
            if len(matches) != len(plan["roles"]["validation"]):
                raise ValueError("Validation matrix is missing a window result")
            for item in matches:
                recomputed = by_seed_window[(eval_seed, item["window_id"])]
                if (item["training_run"] != run_item["path"]
                        or item["training_manifest_sha256"] != run_item["manifest_sha256"]
                        or item["model_sha256"] != run_item["model_sha256"]
                        or item["window_id"] != recomputed["window_id"]
                        or canonical(item["metrics"]) != canonical(recomputed["metrics"])):
                    raise ValueError("recomputed Validation seed-matrix result differs")
    if evaluation_code_hashes != {report.get("evaluation_code_sha256")}:
        raise ValueError("Validation evaluator code fingerprint differs")
    summary = _summary(report["validation_results"], report["baseline_validation_results"])
    if canonical({key: report[key] for key in summary}) != canonical(summary):
        raise ValueError("Validation seed-matrix summary does not recompute")
    return manifest
