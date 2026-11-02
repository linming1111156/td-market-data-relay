from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import numpy as np

from .contracts import (ACTION_TARGETS_LOTS, MAX_TRAIN_MINUTES, MINUTE_NS,
                        RELAY_ROOT, read_role_plan, read_verified_features,
                        reject_system_temp, sha256_file)
from .learner import set_seed, simulate_episode
from .factors import selected_factor_values, suites, registry_sha256
from .historical import verify_plan as verify_historical_role_plan


BASELINE_PATH = Path(__file__).with_name("baseline_strategies.json")
FACTOR_REGISTRY_PATH = Path(__file__).with_name("factor_sets.json")
MAX_EVALUATION_SEEDS = 10
MAX_EVALUATION_MINUTES = 3 * MAX_TRAIN_MINUTES


def package_code_sha256() -> str:
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")) + [BASELINE_PATH, FACTOR_REGISTRY_PATH]:
        digest.update(source.name.encode() + b"\0")
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_inputs(features_path: Path | None, dataset_path: Path | None, role_plan_path: Path,
                training_path: Path, historical_root: Path | None = None):
    for path, label in ((role_plan_path, "role-plan input"), (training_path, "training input")):
        reject_system_temp(path, label)
    if historical_root is not None:
        if features_path is not None or dataset_path is not None:
            raise ValueError("historical evaluation cannot mix Relay and Vision inputs")
        reject_system_temp(historical_root, "historical input")
    elif features_path is None or dataset_path is None:
        raise ValueError("evaluation requires Relay inputs or --historical-inputs")
    else:
        for path, label in ((features_path, "feature input"), (dataset_path, "Dataset input")):
            reject_system_temp(path, label)
    if role_plan_path.is_symlink() or training_path.is_symlink():
        raise ValueError("role plan/training artifact must not be a symlink")
    if historical_root is None:
        feature_manifest, source_manifest = read_verified_features(features_path, dataset_path)
        feature_hash = sha256_file(features_path / "manifest.json")
        source_hash = sha256_file(dataset_path / "manifest.json")
        plan = read_role_plan(role_plan_path, feature_manifest, source_manifest, feature_hash, source_hash)
        historical_manifest = None
    else:
        plan = verify_historical_role_plan(role_plan_path, historical_root)
        historical_manifest = json.loads((historical_root / "manifest.json").read_bytes())
        feature_hash = sha256_file(historical_root / "manifest.json")
        source_hash = historical_manifest["source_manifests"]["bulk_audit_sha256"]
        feature_manifest = {"protocol_sha256": historical_manifest["adapter_configuration_sha256"]}
        source_manifest = {"source_capture_set_sha256": None}
    training_manifest_path = training_path / "manifest.json"
    model_path = training_path / "model.json"
    trace_path = training_path / "training_trace.jsonl.zst"
    if any(path.is_symlink() or not path.is_file() for path in (training_manifest_path, model_path, trace_path)):
        raise ValueError("training artifact is missing a regular manifest/model/trace file")
    if {path.name for path in training_path.iterdir()} != {"manifest.json", "model.json", "training_trace.jsonl.zst"}:
        raise ValueError("training artifact has unmanifested files")
    training = json.loads(training_manifest_path.read_bytes())
    model = json.loads(model_path.read_bytes())
    factor_set_id = training.get("factor_set_id", model.get("factor_set_id", "raw_baseline_v1"))
    factor_names = suites().get(factor_set_id)
    if (factor_names is None
            or model.get("factor_set_id", factor_set_id) != factor_set_id
            or training.get("factor_set_sha256", registry_sha256()) != registry_sha256()
            or model.get("factor_set_sha256", registry_sha256()) != registry_sha256()
            or model.get("feature_names") != factor_names):
        raise ValueError("trained model factor suite does not match the frozen registry")
    if (training.get("schema") != "td_market_data_usdm_rl_training"
            or training.get("version") != 1
            or training.get("role") != "train"
            or training.get("model_sha256") != sha256_file(model_path)
            or training.get("trace_sha256") != sha256_file(trace_path)
            or training.get("features_manifest_sha256") != feature_hash
            or training.get("source_dataset_manifest_sha256") != source_hash
            or training.get("role_plan_sha256") != sha256_file(role_plan_path)
            or training.get("feature_protocol_sha256") != feature_manifest.get("protocol_sha256")):
        raise ValueError("trained model is invalid or bound to different source/role-plan inputs")
    if historical_manifest is not None:
        expected_windows = [{"id": item["id"], "start_ns": item["start_ns"], "end_ns": item["end_ns"]}
                            for item in plan["roles"]["train"]]
        if (training.get("input_kind") != "audited_vision_historical"
                or training.get("historical_input_manifest_sha256") != feature_hash
                or training.get("historical_role_plan_fingerprint_sha256")
                != plan.get("plan_fingerprint_sha256")
                or training.get("train_windows") != expected_windows
                or model.get("normalizer", {}).get("rows") != plan["totals"]["train"]):
            raise ValueError("historical training artifact does not match Train-only role data")
        plan = {"windows": plan["roles"], "historical": True}
    if (model.get("schema") != "td_market_data_usdm_rl_linear_policy"
            or model.get("version") != 1
            or model.get("action_target_lots") != list(ACTION_TARGETS_LOTS)):
        raise ValueError("unsupported trained model contract")
    baseline = json.loads(BASELINE_PATH.read_bytes())
    if baseline.get("schema") != "td_market_data_usdm_rl_baseline_strategies" or baseline.get("version") != 1:
        raise ValueError("unsupported baseline policy registry")
    training["factor_set_id"] = factor_set_id
    training["factor_names"] = factor_names
    return feature_manifest, source_manifest, plan, training, model


def evaluate(inputs: dict[str, Any]) -> dict[str, Any]:
    historical_root = inputs.get("historical_inputs")
    if historical_root is not None:
        historical_root = historical_root.resolve(strict=True)
        feature_path = historical_root
        dataset_path = None
    else:
        feature_path = inputs["features"].resolve(strict=True)
        dataset_path = inputs["dataset"].resolve(strict=True)
    role_plan_path = inputs["role_plan"].resolve(strict=True)
    training_path = inputs["training"].resolve(strict=True)
    feature_manifest, source_manifest, plan, training, model = load_inputs(
        None if historical_root is not None else feature_path, dataset_path, role_plan_path,
        training_path, historical_root)
    evaluation_windows = []
    report_windows: dict[str, list[dict[str, int]]] = {}
    if plan.get("historical"):
        for role, windows in plan["windows"].items():
            if role == "train" or role != inputs.get("evaluation_role", "validation"):
                continue
            for window in windows:
                evaluation_windows.append((role, window,
                    feature_path / window["features"]["path"],
                    feature_path / window["ledger_inputs"]["path"]))
                report_windows.setdefault(role, []).append({
                    "start_ns": window["start_ns"], "end_ns": window["end_ns"]})
    else:
        for role, window in plan["windows"].items():
            if role == "train" or role != inputs.get("evaluation_role", "validation"):
                continue
            evaluation_windows.append((role, window, feature_path / "features.jsonl",
                                      feature_path / "ledger_inputs.jsonl"))
            report_windows[role] = [{"start_ns": window["start_ns"], "end_ns": window["end_ns"]}]
    if not evaluation_windows:
        raise ValueError("role plan has no held-out evaluation windows")
    total_minutes = sum((window["end_ns"] - window["start_ns"]) // MINUTE_NS
                        for _, window, _, _ in evaluation_windows)
    seeds = inputs["seeds"]
    if not seeds or len(seeds) > MAX_EVALUATION_SEEDS or len(set(seeds)) != len(seeds):
        raise ValueError(f"provide 1..{MAX_EVALUATION_SEEDS} unique evaluation seeds")
    if total_minutes > MAX_EVALUATION_MINUTES:
        raise ValueError("held-out evaluation windows exceed the three-year total limit")
    fee_bps = training["accounting"]["fee_bps"]
    slippage_bps = training["accounting"]["slippage_bps"]
    factor_names = training["factor_names"]
    actor = torch.nn.Linear(len(factor_names) + 1, 3, bias=True, device="cpu")
    with torch.no_grad():
        actor.weight.copy_(torch.tensor(model["weights"], dtype=torch.float32))
        actor.bias.copy_(torch.tensor(model["bias"], dtype=torch.float32))
    normalizer = model["normalizer"]
    means, scales = normalizer["mean"], normalizer["scale"]
    if (normalizer.get("fit_role") != "train" or len(means) != len(factor_names)
            or len(scales) != len(factor_names)
            or any(not isinstance(value, (int, float)) or not np.isfinite(value) for value in means + scales)
            or any(value <= 0 for value in scales)):
        raise ValueError("trained normalizer does not match the selected Train-only factor suite")
    if (len(model.get("weights", [])) != 3
            or any(len(row) != len(factor_names) + 1 or any(not np.isfinite(value) for value in row) for row in model["weights"])
            or len(model.get("bias", [])) != 3
            or any(not np.isfinite(value) for value in model["bias"])):
        raise ValueError("trained model parameters have invalid shape or non-finite values")
    strategy_registry = json.loads(BASELINE_PATH.read_bytes())
    strategies = [entry["id"] for entry in strategy_registry["strategies"]]
    if strategies != ["noop_flat_v1", "buy_and_hold_long_v1", "mark_momentum_15m_sign_v1"]:
        raise ValueError("baseline registry contains an unsupported or reordered strategy")
    results = []
    for role, window, input_feature_path, input_ledger_path in evaluation_windows:
        for seed in seeds:
            for strategy in ["rl_policy_v1", *strategies]:
                set_seed(seed)
                rng = torch.Generator(device="cpu").manual_seed(seed)

                def select_action(feature: dict[str, Any], position: int,
                                  selected: str = strategy) -> tuple[int, list[float] | None]:
                    if selected == "noop_flat_v1":
                        return 1, None
                    if selected == "buy_and_hold_long_v1":
                        return 2, None
                    if selected == "mark_momentum_15m_sign_v1":
                        momentum = feature["features"]["mark_log_return_15m"]
                        return (2 if momentum > 0 else 0 if momentum < 0 else 1), None
                    _, values = selected_factor_values(feature["features"], training["factor_set_id"])
                    vector = [(value - means[index]) / scales[index]
                              for index, value in enumerate(values)] + [position]
                    with torch.inference_mode():
                        probabilities = torch.softmax(actor(torch.tensor(vector, dtype=torch.float32)), dim=0)
                        action = int(torch.multinomial(probabilities, 1, generator=rng).item())
                    return action, None

                experiences, stats = simulate_episode(
                    input_feature_path, input_ledger_path,
                    window, select_action, fee_bps, slippage_bps, 4096,
                    collect_experiences=False)
                results.append({"role": role, "seed": seed, "strategy": strategy,
                                "window_id": window.get("id", role),
                                "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
                                "metrics": stats})
    return {"schema": "td_market_data_usdm_rl_baseline_report", "version": 1,
            "factor_set_id": training["factor_set_id"],
            "factor_set_sha256": registry_sha256(),
            "source_dataset_manifest_sha256": training["source_dataset_manifest_sha256"],
            "features_manifest_sha256": sha256_file(feature_path / "manifest.json"),
            "role_plan_sha256": sha256_file(role_plan_path),
            "training_manifest_sha256": sha256_file(training_path / "manifest.json"),
            "trained_model_sha256": training["model_sha256"],
            "baseline_registry_sha256": sha256_file(BASELINE_PATH),
            "evaluation_code_sha256": package_code_sha256(),
            "evaluation_role": inputs.get("evaluation_role", "validation"),
            "evaluation_windows": report_windows, "seeds": seeds,
            "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
            "accounting_protocol_sha256": training["feature_protocol_sha256"],
            "accounting": {"initial_equity_usdt": 10_000.0, "lot_size_btc": 0.001,
                           "minimum_notional_usdt": 50.0, "entry_notional_cap_usdt": 100.0,
                           "maximum_absolute_position_btc": 0.001,
                           "participation_cap_fraction": 0.01,
                           "maintenance_margin_fraction": 0.05,
                           "liquidation_surcharge_bps": 25},
            "policy_cadence_minutes": 15,
            "device": "cpu", "results": results,
            "runtime_versions": {"torch": torch.__version__, "numpy": np.__version__,
                                 "cuda_build": torch.version.cuda},
            "limitations": ["Execution uses next-minute aggregate-trade VWAP, not order-book queue simulation.",
                            "Only selected source captures and fixed windows are represented.",
                            "The frozen momentum baseline is descriptive and was not tuned on held-out windows."]}


def create_report(args: Any) -> dict[str, Any]:
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("evaluation output must be a new create-only directory")
    reject_system_temp(output, "evaluation output")
    if output.resolve() == RELAY_ROOT or RELAY_ROOT in output.resolve().parents:
        raise ValueError("evaluation report must be stored outside the source repository")
    inputs = {"features": args.features, "dataset": args.dataset,
              "historical_inputs": args.historical_inputs, "role_plan": args.role_plan,
              "training": args.training, "seeds": args.seeds,
              "evaluation_role": args.evaluation_role}
    report = evaluate(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = stage / "report.json"
        report_path.write_bytes(json.dumps(report, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        manifest = {"schema": "td_market_data_usdm_rl_evaluation_artifact", "version": 1,
                    "report_sha256": sha256_file(report_path),
                    "source_dataset_manifest_sha256": report["source_dataset_manifest_sha256"],
                    "features_manifest_sha256": report["features_manifest_sha256"],
                    "role_plan_sha256": report["role_plan_sha256"],
                    "training_manifest_sha256": report["training_manifest_sha256"],
                    "factor_set_id": report["factor_set_id"],
                    "factor_set_sha256": report["factor_set_sha256"],
                    "baseline_registry_sha256": report["baseline_registry_sha256"],
                    "evaluation_code_sha256": report["evaluation_code_sha256"]}
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
        for path in stage.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.chmod(report_path, 0o444)
        os.chmod(manifest_path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return manifest
    except BaseException:
        try:
            os.chmod(stage, 0o755)
            for path in stage.iterdir():
                os.chmod(path, 0o644)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise


def verify_report(args: Any) -> dict[str, Any]:
    if args.artifact.is_symlink():
        raise ValueError("evaluation artifact must not be a symlink")
    artifact = args.artifact.resolve(strict=True)
    reject_system_temp(artifact, "evaluation artifact")
    if not artifact.is_dir():
        raise ValueError("evaluation artifact must be a directory")
    entries = list(artifact.iterdir())
    if ({path.name for path in entries} != {"manifest.json", "report.json"}
            or any(path.is_symlink() or not path.is_file() for path in entries)):
        raise ValueError("evaluation artifact has missing or unmanifested files")
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    report_path = artifact / "report.json"
    if (manifest.get("schema") != "td_market_data_usdm_rl_evaluation_artifact"
            or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)):
        raise ValueError("evaluation artifact manifest/report hash mismatch")
    report = json.loads(report_path.read_bytes())
    for field in ("source_dataset_manifest_sha256", "features_manifest_sha256",
                  "role_plan_sha256", "training_manifest_sha256", "baseline_registry_sha256",
                  "evaluation_code_sha256", "factor_set_id", "factor_set_sha256"):
        if manifest.get(field) != report.get(field):
            raise ValueError(f"evaluation artifact manifest mismatch: {field}")
    inputs = {"features": args.features, "dataset": args.dataset,
              "historical_inputs": args.historical_inputs, "role_plan": args.role_plan,
              "training": args.training, "seeds": report["seeds"],
              "evaluation_role": report["evaluation_role"]}
    recomputed = evaluate(inputs)
    if sha256_file(report_path) != hashlib.sha256(
            json.dumps(recomputed, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n").hexdigest():
        raise ValueError("recomputed evaluation report differs from the published report")
    return manifest
