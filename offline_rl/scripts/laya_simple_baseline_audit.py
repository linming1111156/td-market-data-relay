#!/usr/bin/env python3
"""Compare a verified Laya Validation report with fixed-position controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any

from td_market_data_rl.contracts import MINUTE_NS, feasible_action_mask, reject_system_temp
from td_market_data_rl.historical import verify_plan
from td_market_data_rl.learner import simulate_episode


SCRIPT = Path(__file__).resolve()
ARTIFACT_SCHEMA = "td_market_data_laya_simple_baseline_audit_artifact"
REPORT_SCHEMA = "td_market_data_laya_simple_baseline_audit"
RL_STRATEGIES = ("rl_policy_v1", "rl_policy_argmax_v1", "noop_flat_v1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def data_path(path: Path, label: str, *, must_exist: bool = True) -> Path:
    resolved = path.resolve(strict=must_exist)
    root = Path("/mnt/data").resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must be under /mnt/data")
    if resolved == Path("/tmp") or Path("/tmp") in resolved.parents:
        raise ValueError(f"{label} must not use /tmp")
    return resolved


def read_laya_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    if path.is_symlink():
        raise ValueError("Laya artifact must not be a symlink")
    root = data_path(path, "Laya artifact")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Laya artifact must be a regular directory")
    if {item.name for item in root.iterdir()} != {"manifest.json", "report.json"}:
        raise ValueError("Laya artifact has missing or unmanifested files")
    manifest_path, report_path = root / "manifest.json", root / "report.json"
    if any(p.is_symlink() or not p.is_file() for p in (manifest_path, report_path)):
        raise ValueError("Laya artifact files must be regular files")
    manifest = json.loads(manifest_path.read_bytes())
    report = json.loads(report_path.read_bytes())
    if (manifest.get("schema") != "td_market_data_laya_policy_validation_artifact"
            or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or report.get("schema") != "td_market_data_laya_policy_validation"
            or report.get("evaluation_role") != "validation"):
        raise ValueError("Laya artifact manifest is invalid or is not Validation-only")
    return manifest, report, sha256_file(manifest_path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty strategy result set")
    keys = ("net_return_fraction", "net_pnl_usdt", "gross_price_pnl_usdt",
            "funding_pnl_usdt", "fees_usdt", "slippage_usdt", "fills",
            "traded_notional_usdt", "exposure_minutes", "max_drawdown_fraction")
    return {
        "observations": len(rows),
        **{f"mean_{key}": mean(float(row["metrics"][key]) for row in rows)
           for key in keys},
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = data_path(args.historical_inputs, "historical inputs")
    role_plan = data_path(args.role_plan, "role plan")
    training = data_path(args.training, "training artifact")
    _, laya_report, laya_manifest_sha = read_laya_artifact(args.laya_artifact)
    plan = verify_plan(role_plan, inputs, verify_input_artifact=False)
    windows = plan["roles"].get("validation", [])
    if not windows:
        raise ValueError("role plan has no Validation windows")
    training_manifest_path = training / "manifest.json"
    if training_manifest_path.is_symlink() or not training_manifest_path.is_file():
        raise ValueError("RL training manifest is missing or not a regular file")
    training_manifest = json.loads(training_manifest_path.read_bytes())
    training_hash = sha256_file(training_manifest_path)
    if (laya_report.get("role_plan_sha256") != sha256_file(role_plan)
            or laya_report.get("historical_input_manifest_sha256") != sha256_file(inputs / "manifest.json")
            or laya_report.get("training_manifest_sha256") != training_hash):
        raise ValueError("Laya report and requested inputs are not bound to the same frozen data/model")
    fee_bps = training_manifest["accounting"]["fee_bps"]
    slippage_bps = training_manifest["accounting"]["slippage_bps"]
    if (laya_report.get("cost_scenario") != {"fee_bps": fee_bps, "slippage_bps": slippage_bps}
            or fee_bps != 10 or slippage_bps != 5):
        raise ValueError("baseline audit expects the Laya report's frozen 10+5 bps cost case")

    controls: dict[str, list[dict[str, Any]]] = {
        "always_short_v1": [], "always_long_v1": [], "noop_flat_v1": []}
    for window in windows:
        feature_path = inputs / window["features"]["path"]
        ledger_path = inputs / window["ledger_inputs"]["path"]
        for path, metadata in ((feature_path, window["features"]), (ledger_path, window["ledger_inputs"])):
            if (path.is_symlink() or not path.is_file()
                    or path.stat().st_size != metadata["size_bytes"]
                    or sha256_file(path) != metadata["sha256"]):
                raise ValueError(f"Validation shard hash mismatch: {path}")
        for strategy, target in (("always_short_v1", 0), ("always_long_v1", 2), ("noop_flat_v1", 1)):
            def choose(_feature: dict[str, Any], position: int, action: int = target):
                mask = feasible_action_mask(_feature.get("mark_price"), position)
                selected = action if mask[action] else 1
                return selected, None, mask

            _, metrics = simulate_episode(
                feature_path, ledger_path, window, choose,
                fee_bps, slippage_bps, args.max_memory_mb,
                collect_experiences=False,
            )
            if metrics["minutes"] != (window["end_ns"] - window["start_ns"]) // MINUTE_NS:
                raise ValueError(f"Validation window row count differs from plan: {window['id']}")
            controls[strategy].append({
                "strategy": strategy, "role": "validation", "window_id": window["id"],
                "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
                "metrics": metrics,
            })

    rl_results = [row for row in laya_report["rl_results"]
                  if row.get("strategy") in RL_STRATEGIES and row.get("role") == "validation"]
    rl_by_strategy = {strategy: summarize([row for row in rl_results if row["strategy"] == strategy])
                      for strategy in RL_STRATEGIES}
    if any(row["observations"] != len(windows) * 3 for row in rl_by_strategy.values()):
        raise ValueError("Laya artifact does not contain the expected three-seed RL/NOOP comparison")
    laya_rows = laya_report["laya_results"]
    if len(laya_rows) != len(windows):
        raise ValueError("Laya report does not contain exactly one deterministic result per Validation window")
    laya_summary = summarize(laya_rows)
    control_summaries = {strategy: summarize(rows) for strategy, rows in controls.items()}
    strategies = {"laya_zero_shot": laya_summary, **control_summaries, **rl_by_strategy}
    laya_return = laya_summary["mean_net_return_fraction"]
    return {
        "schema": REPORT_SCHEMA, "version": 1, "evaluation_role": "validation",
        "laya_artifact_manifest_sha256": laya_manifest_sha,
        "laya_report_sha256": sha256_file(data_path(args.laya_artifact, "Laya artifact") / "report.json"),
        "historical_input_manifest_sha256": sha256_file(inputs / "manifest.json"),
        "role_plan_sha256": sha256_file(role_plan),
        "training_manifest_sha256": training_hash,
        "cost_scenario": {"fee_bps": fee_bps, "slippage_bps": slippage_bps},
        "window_ids": [window["id"] for window in windows],
        "laya_results": laya_rows,
        "fixed_control_results": controls,
        "rl_results": rl_results,
        "summary": {
            "strategies": strategies,
            "laya_beats_always_short": laya_return > control_summaries["always_short_v1"]["mean_net_return_fraction"],
            "laya_beats_always_long": laya_return > control_summaries["always_long_v1"]["mean_net_return_fraction"],
            "laya_beats_noop": laya_return > control_summaries["noop_flat_v1"]["mean_net_return_fraction"],
            "laya_beats_rl_sampled": laya_return > rl_by_strategy["rl_policy_v1"]["mean_net_return_fraction"],
            "laya_beats_rl_argmax": laya_return > rl_by_strategy["rl_policy_argmax_v1"]["mean_net_return_fraction"],
        },
        "audit_code_sha256": sha256_file(SCRIPT),
        "limitations": [
            "This is the already-inspected Validation window and is exploratory, not independent evidence.",
            "A positive result against RL does not show incremental value over a simple fixed-position policy.",
            "Always-short and always-long controls obey the same action feasibility mask, fill ledger, fees, slippage, funding, and terminal close rules.",
            "Test and OOS are not read by this audit.",
        ],
    }


def write_artifact(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    output = data_path(output, "artifact output", must_exist=False)
    if output.exists() or output.is_symlink():
        raise ValueError("audit artifact output must be new and create-only")
    if not output.parent.is_dir():
        raise ValueError("audit artifact parent directory must already exist")
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
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return manifest
    except BaseException:
        try:
            os.chmod(stage, 0o755)
            shutil.rmtree(stage, ignore_errors=True)
        except OSError:
            pass
        raise


def verify_artifact(args: argparse.Namespace) -> dict[str, Any]:
    artifact = data_path(args.artifact, "audit artifact")
    if artifact.is_symlink() or not artifact.is_dir() or {p.name for p in artifact.iterdir()} != {"manifest.json", "report.json"}:
        raise ValueError("audit artifact has missing or unmanifested files")
    manifest_path, report_path = artifact / "manifest.json", artifact / "report.json"
    if any(path.is_symlink() or not path.is_file() for path in (manifest_path, report_path)):
        raise ValueError("audit artifact files must be regular files")
    manifest = json.loads(manifest_path.read_bytes())
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if (manifest.get("schema") != ARTIFACT_SCHEMA or manifest.get("version") != 1
            or manifest.get("report_sha256") != sha256_file(report_path)
            or manifest.get("laya_report_sha256") != report.get("laya_report_sha256")
            or manifest.get("audit_code_sha256") != report.get("audit_code_sha256")):
        raise ValueError("audit artifact manifest/report hash mismatch")
    if canonical_bytes(build_report(args)) != report_bytes:
        raise ValueError("recomputed fixed-baseline audit differs from artifact")
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
        sub.add_argument("--max-memory-mb", type=int, default=4096)
        if name == "create":
            sub.add_argument("--output", type=Path, required=True)
        else:
            sub.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.max_memory_mb <= 12_288:
        raise SystemExit("--max-memory-mb must be in 1..12288")
    try:
        if args.command == "create":
            report = build_report(args)
            manifest = write_artifact(args.output, report)
            for strategy, stats in report["summary"]["strategies"].items():
                print(f"{strategy}: return={stats['mean_net_return_fraction']:.6%} "
                      f"pnl={stats['mean_net_pnl_usdt']:.4f} "
                      f"fills={stats['mean_fills']:.2f} "
                      f"notional={stats['mean_traded_notional_usdt']:.2f}")
            print(f"Laya beats always short: {report['summary']['laya_beats_always_short']}")
            print(f"artifact: {args.output.resolve()}")
            print(f"report sha256: {manifest['report_sha256']}")
        else:
            manifest = verify_artifact(args)
            print("fixed-control Validation audit recomputed and verified")
            print(f"report sha256: {manifest['report_sha256']}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(f"Laya fixed-baseline audit rejected: {error}") from error


if __name__ == "__main__":
    main()
