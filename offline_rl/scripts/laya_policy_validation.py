#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from td_market_data_rl.contracts import (
    ACTION_TARGETS_LOTS,
    LOT_BTC,
    MINUTE_NS,
    feasible_action_mask,
    reject_system_temp,
)
from td_market_data_rl.evaluator import evaluate
from td_market_data_rl.historical import verify_plan
from td_market_data_rl.learner import simulate_episode


SCRIPT = Path(__file__).resolve()
RELAY_ROOT = SCRIPT.parents[2]
DEFAULT_CONFIG = RELAY_ROOT / "offline_rl/src/td_market_data_rl/laya_strategy.json"
DEFAULT_CACHE = Path("/mnt/data/td-market-data-local/model-cache/laya")
MAX_EVALUATION_RSS_MB = 12_288
LAYER_SCHEMA = "td_market_data_laya_policy_validation"
ARTIFACT_SCHEMA = "td_market_data_laya_policy_validation_artifact"
RL_STRATEGIES = ["rl_policy_v1", "rl_policy_argmax_v1", "noop_flat_v1"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def current_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def validate_data_path(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    data_root = Path("/mnt/data").resolve(strict=True)
    if resolved != data_root and data_root not in resolved.parents:
        raise ValueError(f"{label} must be stored under /mnt/data")
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError(f"{label} must not use /tmp")
    return resolved


def load_strategy_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Laya strategy config must be a regular local file")
    raw = path.read_bytes()
    config = json.loads(raw)
    if (config.get("schema") != "td_market_data_laya_zero_shot_policy"
            or config.get("version") != 1
            or config.get("checkpoint") != "typed-decisions"
            or config.get("question_id") != "target_position"
            or config.get("actions") != {"short": -1, "flat": 0, "long": 1}
            or set(config.get("criteria", {})) != {"short", "flat", "long"}
            or not config.get("instructions")):
        raise ValueError("unsupported or incomplete local Laya strategy config")
    revision = config.get("checkpoint_revision")
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Laya checkpoint revision must be a pinned lowercase Git SHA")
    return config, hashlib.sha256(raw).hexdigest()


def checkpoint_inventory(cache_root: Path, revision: str) -> tuple[dict[str, Any], Path]:
    hf_cache = cache_root / "hub"
    snapshot = hf_cache / "models--convaiinnovations--laya" / "snapshots" / revision / "typed-decisions"
    if not snapshot.is_dir():
        raise ValueError(f"pinned Laya checkpoint is not cached at {snapshot}")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if hf_cache not in resolved.parents:
                    raise ValueError("Laya checkpoint file resolves outside its Hugging Face cache")
            files.append({"path": path.relative_to(snapshot).as_posix(),
                          "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    if not files or not any(item["path"].endswith("model.safetensors") for item in files):
        raise ValueError("pinned Laya checkpoint has no model.safetensors file")
    return {"repo_id": "convaiinnovations/laya", "subfolder": "typed-decisions",
            "revision": revision, "files": files}, snapshot


def make_state(features: dict[str, Any], mark_price: float | None, position_lots: int) -> dict[str, Any]:
    def numeric(name: str, scale: float = 1.0) -> float | None:
        value = features.get(name)
        if value is None:
            return None
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"invalid Laya market feature: {name}")
        return round(float(value) * scale, 6)

    return {
        "market": "BTCUSDT Binance USD-M perpetual",
        "decision_interval_minutes": 15,
        "market_features": {
            "mark_return_1m_bps": numeric("mark_log_return_1m", 10_000),
            "mark_return_5m_bps": numeric("mark_log_return_5m", 10_000),
            "mark_return_15m_bps": numeric("mark_log_return_15m", 10_000),
            "realized_volatility_15m_bps": numeric("mark_realized_volatility_15m", 10_000),
            "realized_volatility_60m_bps": numeric("mark_realized_volatility_60m", 10_000),
            "signed_volume_imbalance_1m": numeric("aggtrade_signed_volume_imbalance_1m"),
            "signed_volume_imbalance_5m": numeric("aggtrade_signed_volume_imbalance_5m"),
        },
        "mark_price_usdt": round(float(mark_price), 4) if mark_price is not None else None,
        "portfolio": {"current_position_lots": int(position_lots),
                      "current_position_btc": round(int(position_lots) * LOT_BTC, 6)},
        "risk_constraints": {
            "maximum_absolute_position_btc": LOT_BTC,
            "entry_order_btc": LOT_BTC,
            "minimum_notional_usdt": 50,
            "maximum_entry_notional_usdt": 100,
            "taker_fee_bps_each_side": 10,
            "adverse_slippage_bps_each_side": 5,
            "funding_is_applied_from_the_historical_settlement_ledger": True,
        },
        "available_targets": ["short", "flat", "long"],
    }


def build_report(args: Any) -> dict[str, Any]:
    input_root = validate_data_path(args.historical_inputs, "historical inputs")
    plan_path = validate_data_path(args.role_plan, "role plan")
    training_path = validate_data_path(args.training, "RL training artifact")
    config_path = args.laya_config.resolve(strict=True)
    config, config_hash = load_strategy_config(config_path)
    cache_root = args.model_cache.absolute()
    if cache_root.exists():
        cache_root = validate_data_path(cache_root, "Laya model cache")
    else:
        data_root = Path("/mnt/data").resolve(strict=True)
        if data_root not in cache_root.resolve().parents:
            raise ValueError("Laya model cache must be under /mnt/data")
        cache_root.mkdir(parents=True, exist_ok=True)
    reject_system_temp(input_root, "historical inputs")
    reject_system_temp(plan_path, "role plan")
    reject_system_temp(training_path, "RL training artifact")

    # Keep downloaded weights out of the default home cache and /tmp.
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["TMPDIR"] = str(args.temp_root.resolve(strict=True))
    reject_system_temp(args.temp_root.resolve(strict=True), "temporary workspace")
    validate_data_path(args.temp_root, "temporary workspace")

    if not torch.cuda.is_available():
        raise ValueError("Laya validation is configured for CUDA, but CUDA is unavailable")
    total_gpu_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    if args.max_gpu_memory_mb <= 0 or args.max_gpu_memory_mb > total_gpu_mb:
        raise ValueError("GPU memory cap must be positive and no larger than device memory")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_mb / total_gpu_mb, device=0)
    torch.set_num_threads(4)

    from laya import Router

    router = Router(
        device="cuda",
        max_loaded=int(config["max_loaded_checkpoints"]),
        revisions={"typed-decisions": config["checkpoint_revision"]},
    )
    plan = verify_plan(plan_path, input_root, verify_input_artifact=False)
    validation_windows = plan["roles"].get("validation", [])
    if not validation_windows:
        raise ValueError("role plan has no Validation windows")

    rl_report = evaluate({
        "features": None,
        "dataset": None,
        "historical_inputs": input_root,
        "role_plan": plan_path,
        "training": training_path,
        "seeds": [101, 202, 303],
        "evaluation_role": "validation",
        "strategies": RL_STRATEGIES,
    })
    training_manifest = json.loads((training_path / "manifest.json").read_bytes())
    fee_bps = training_manifest["accounting"]["fee_bps"]
    slippage_bps = training_manifest["accounting"]["slippage_bps"]
    if fee_bps != 10 or slippage_bps != 5:
        raise ValueError("this frozen Laya comparison expects the RL model's 10+5 bps cost protocol")

    laya_results = []
    for window in validation_windows:
        feature_path = input_root / window["features"]["path"]
        ledger_path = input_root / window["ledger_inputs"]["path"]
        for source_path, metadata in ((feature_path, window["features"]),
                                      (ledger_path, window["ledger_inputs"])):
            if (source_path.is_symlink() or not source_path.is_file()
                    or source_path.stat().st_size != metadata["size_bytes"]
                    or sha256_file(source_path) != metadata["sha256"]):
                raise ValueError(f"Validation shard hash mismatch: {source_path}")
        counts: Counter[str] = Counter()

        def choose_laya(feature: dict[str, Any], position_lots: int):
            mask = feasible_action_mask(feature.get("mark_price"), position_lots)
            # Respect the exact same action feasibility mask as the RL policy.
            if not mask[0] and not mask[2]:
                counts["flat_forced_by_feasibility_mask"] += 1
                return 1, None, mask
            state = make_state(feature["features"], feature.get("mark_price"), position_lots)
            result = router.predict(state, {
                config["question_id"]: {
                    "type": "choice",
                    "instructions": config["instructions"],
                    "criteria": config["criteria"],
                }
            }, model=config["checkpoint"])
            answer = result.get("answers", {}).get(config["question_id"], {})
            choice = answer.get("choice")
            if choice not in config["actions"]:
                raise ValueError(f"Laya returned an unsupported target choice: {choice!r}")
            action = {"short": 0, "flat": 1, "long": 2}[choice]
            if not mask[action]:
                raise ValueError("Laya selected an action excluded by the shared feasibility mask")
            counts[choice] += 1
            counts["model_queries"] += 1
            if current_rss_mb() > args.max_memory_mb:
                raise ValueError(f"RSS exceeded configured {args.max_memory_mb} MiB cap")
            return action, None, mask

        _, metrics = simulate_episode(
            feature_path, ledger_path, window, choose_laya,
            fee_bps, slippage_bps, args.max_memory_mb,
            collect_experiences=False,
        )
        laya_results.append({
            "strategy": config["strategy_id"],
            "role": "validation",
            "window_id": window["id"],
            "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
            "action_counts": dict(sorted(counts.items())),
            "metrics": metrics,
        })

    checkpoint, snapshot = checkpoint_inventory(cache_root, config["checkpoint_revision"])
    if current_rss_mb() > args.max_memory_mb:
        raise ValueError(f"RSS exceeded configured {args.max_memory_mb} MiB cap")
    rl_summary: dict[str, dict[str, Any]] = {}
    for strategy in RL_STRATEGIES:
        rows = [row for row in rl_report["results"] if row["strategy"] == strategy]
        rl_summary[strategy] = {
            "observations": len(rows),
            "mean_net_return_fraction": mean(row["metrics"]["net_return_fraction"] for row in rows),
            "mean_gross_price_pnl_usdt": mean(row["metrics"]["gross_price_pnl_usdt"] for row in rows),
            "mean_fees_usdt": mean(row["metrics"]["fees_usdt"] for row in rows),
            "mean_slippage_usdt": mean(row["metrics"]["slippage_usdt"] for row in rows),
            "mean_fills": mean(row["metrics"]["fills"] for row in rows),
            "mean_traded_notional_usdt": mean(row["metrics"]["traded_notional_usdt"] for row in rows),
        }
    laya_mean_return = mean(row["metrics"]["net_return_fraction"] for row in laya_results)
    report = {
        "schema": LAYER_SCHEMA,
        "version": 1,
        "evaluation_role": "validation",
        "strategy_type": "zero_shot_pretrained_typed_decision_model",
        "strategy_id": config["strategy_id"],
        "laya_strategy_config_sha256": config_hash,
        "laya_package_version": __import__("laya").__version__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "checkpoint": checkpoint,
        "checkpoint_local_snapshot": str(snapshot),
        "checkpoint_file_manifest_sha256": hashlib.sha256(canonical_bytes(checkpoint["files"])).hexdigest(),
        "historical_input_manifest_sha256": sha256_file(input_root / "manifest.json"),
        "role_plan_sha256": sha256_file(plan_path),
        "role_plan_fingerprint_sha256": plan["plan_fingerprint_sha256"],
        "training_manifest_sha256": sha256_file(training_path / "manifest.json"),
        "trained_model_sha256": training_manifest["model_sha256"],
        "rl_evaluation_code_sha256": rl_report["evaluation_code_sha256"],
        "laya_validation_code_sha256": sha256_file(SCRIPT),
        "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
        "accounting_protocol_sha256": rl_report["accounting_protocol_sha256"],
        "evaluation_seeds_for_rl": [101, 202, 303],
        "laya_results": laya_results,
        "rl_results": rl_report["results"],
        "summary": {
            "laya_mean_net_return_fraction": laya_mean_return,
            "laya_mean_gross_price_pnl_usdt": mean(row["metrics"]["gross_price_pnl_usdt"] for row in laya_results),
            "rl_by_strategy": rl_summary,
            "laya_beats_rl_sampled_mean": laya_mean_return > rl_summary["rl_policy_v1"]["mean_net_return_fraction"],
            "laya_beats_rl_argmax_mean": laya_mean_return > rl_summary["rl_policy_argmax_v1"]["mean_net_return_fraction"],
            "laya_beats_noop": laya_mean_return > rl_summary["noop_flat_v1"]["mean_net_return_fraction"],
        },
        "resource_limits": {"max_rss_mb": args.max_memory_mb,
                            "max_gpu_memory_mb": args.max_gpu_memory_mb,
                            "max_loaded_laya_checkpoints": int(config["max_loaded_checkpoints"])},
        "limitations": [
            "Laya is a general pretrained typed-decision model, not a finance-trained predictor or RL trading agent.",
            "The local prompt is a zero-shot research strategy and is kept out of Git; its hash is recorded.",
            "Laya confidence/probability outputs are not used; this checkpoint emitted a calibration warning during the smoke call.",
            "Validation has already been used for model selection, so this comparison is exploratory and not independent evidence.",
            "This report does not read Test or OOS and does not establish live trading performance.",
            "Execution uses the same frozen next-minute aggregate-trade accounting approximation as the RL comparison.",
        ],
    }
    return report


def write_artifact(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Laya artifact output must be new and create-only")
    reject_system_temp(output, "Laya artifact output")
    validate_data_path(output.parent, "Laya artifact parent")
    if output.resolve() == RELAY_ROOT or RELAY_ROOT in output.resolve().parents:
        raise ValueError("Laya validation artifact must be outside the source repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        report_path = stage / "report.json"
        report_path.write_bytes(canonical_bytes(report))
        manifest = {
            "schema": ARTIFACT_SCHEMA,
            "version": 1,
            "report_sha256": sha256_file(report_path),
            "historical_input_manifest_sha256": report["historical_input_manifest_sha256"],
            "role_plan_sha256": report["role_plan_sha256"],
            "training_manifest_sha256": report["training_manifest_sha256"],
            "laya_strategy_config_sha256": report["laya_strategy_config_sha256"],
            "checkpoint_file_manifest_sha256": report["checkpoint_file_manifest_sha256"],
            "laya_validation_code_sha256": report["laya_validation_code_sha256"],
        }
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(canonical_bytes(manifest))
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


def verify_artifact(args: Any) -> dict[str, Any]:
    artifact = args.artifact.resolve(strict=True)
    if artifact.is_symlink() or not artifact.is_dir():
        raise ValueError("Laya artifact must be a regular directory")
    files = list(artifact.iterdir())
    if ({path.name for path in files} != {"manifest.json", "report.json"}
            or any(path.is_symlink() or not path.is_file() for path in files)):
        raise ValueError("Laya artifact has missing or unmanifested files")
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    report_path = artifact / "report.json"
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)):
        raise ValueError("Laya artifact manifest/report hash mismatch")
    for field in ("historical_input_manifest_sha256", "role_plan_sha256",
                  "training_manifest_sha256", "laya_strategy_config_sha256",
                  "checkpoint_file_manifest_sha256", "laya_validation_code_sha256"):
        if manifest.get(field) != report.get(field):
            raise ValueError(f"Laya artifact manifest mismatch: {field}")
    recomputed = build_report(args)
    if canonical_bytes(recomputed) != report_path.read_bytes():
        raise ValueError("recomputed Laya/RL Validation report differs from artifact")
    return manifest


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--historical-inputs", type=Path, required=True)
    parser.add_argument("--role-plan", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--laya-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--temp-root", type=Path,
                        default=Path("/mnt/data/td-market-data-local/research-inputs/.tmp"))
    parser.add_argument("--max-memory-mb", type=int, default=MAX_EVALUATION_RSS_MB)
    parser.add_argument("--max-gpu-memory-mb", type=int, default=8192)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a frozen zero-shot Laya policy with the same frozen RL model on Validation.")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    add_common_paths(create)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    add_common_paths(verify)
    verify.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if args.max_memory_mb <= 0 or args.max_memory_mb > MAX_EVALUATION_RSS_MB:
        raise SystemExit(f"--max-memory-mb must be in 1..{MAX_EVALUATION_RSS_MB}")
    try:
        if args.command == "create":
            report = build_report(args)
            manifest = write_artifact(args.output, report)
            print(f"Laya Validation comparison created: strategies=4 windows={len(report['laya_results'])}")
            print(f"Laya mean return: {report['summary']['laya_mean_net_return_fraction']:.6%}")
            print(f"RL sampled mean: {report['summary']['rl_by_strategy']['rl_policy_v1']['mean_net_return_fraction']:.6%}")
            print(f"RL argmax mean: {report['summary']['rl_by_strategy']['rl_policy_argmax_v1']['mean_net_return_fraction']:.6%}")
            print(f"NOOP: {report['summary']['rl_by_strategy']['noop_flat_v1']['mean_net_return_fraction']:.6%}")
            print(f"artifact: {args.output.resolve()}")
            print(f"report sha256: {manifest['report_sha256']}")
        else:
            manifest = verify_artifact(args)
            print("Laya Validation comparison verified by recomputing every policy result")
            print(f"report sha256: {manifest['report_sha256']}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            torch.cuda.OutOfMemoryError) as error:
        raise SystemExit(f"Laya policy validation rejected: {error}") from error


if __name__ == "__main__":
    main()
