from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .contracts import MAX_TRAIN_MINUTES, MINUTE_NS, RELAY_ROOT

try:
    from verify_vision_historical_rl_inputs import verify as verify_historical_inputs
except ImportError as error:  # pragma: no cover - packaging/source layout guard
    raise RuntimeError("historical input verifier is unavailable from the Relay repository") from error


PLAN_SCHEMA = "td_market_data_usdm_rl_historical_role_plan"
PLAN_PROTOCOL = "audited_reliable_half_open_role_windows_v1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_output(path: Path) -> Path:
    output = path.absolute()
    resolved = output.resolve()
    data_root = Path("/mnt/data").resolve(strict=True)
    if output.exists() or output.is_symlink():
        raise ValueError("historical role-plan output must be new and create-only")
    if resolved == data_root or data_root not in resolved.parents:
        raise ValueError("historical role plan must be stored under /mnt/data")
    if any(resolved == root or root in resolved.parents for root in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError("historical role plan must not be stored in system temporary storage")
    if resolved == RELAY_ROOT or RELAY_ROOT in resolved.parents:
        raise ValueError("historical role plan must be outside the source repository")
    return output


def build_payload(input_root: Path, verify_artifact: bool = True) -> dict[str, Any]:
    input_root = input_root.resolve(strict=True)
    if verify_artifact:
        verify_historical_inputs(input_root)
    input_manifest = json.loads((input_root / "manifest.json").read_bytes())
    manifest_path = input_root / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    grouped: dict[str, list[dict[str, Any]]] = {role: [] for role in ("train", "validation", "test", "oos")}
    ordered = []
    for window in input_manifest["windows"]:
        item = {
            "id": window["id"],
            "start_ns": window["start_ms"] * 1_000_000,
            "end_ns": window["end_ms"] * 1_000_000,
            "minute_count": window["row_count"],
            "features": {**window["features"], "path": f"windows/{window['id']}/{window['features']['path']}"},
            "ledger_inputs": {**window["ledger_inputs"],
                              "path": f"windows/{window['id']}/{window['ledger_inputs']['path']}"},
        }
        grouped[window["role"]].append(item)
        ordered.append((item["start_ns"], item["end_ns"], window["role"], item["id"]))
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("historical inputs must contain Train and Validation roles")
    if sum(item["minute_count"] for item in grouped["train"]) > MAX_TRAIN_MINUTES:
        raise ValueError("historical Train role exceeds the one-year cap")
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("historical reliable windows overlap or are out of chronological order")
    payload = {
        "schema": PLAN_SCHEMA,
        "version": 1,
        "protocol": PLAN_PROTOCOL,
        "input_manifest_sha256": manifest_sha,
        "adapter_code_sha256": input_manifest["adapter"]["sha256"],
        "role_plan_code_sha256": sha256_file(Path(__file__).resolve()),
        "feature_names": input_manifest["feature_names"],
        "time_basis": input_manifest["window_time_basis"],
        "funding_time_basis": input_manifest["funding_time_basis"],
        "roles": grouped,
        "totals": {role: sum(item["minute_count"] for item in grouped[role])
                   for role in grouped},
    }
    payload["plan_fingerprint_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def verify_plan(plan_path: Path, input_root: Path,
                verify_input_artifact: bool = True) -> dict[str, Any]:
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("historical role plan must be a regular file")
    input_root = input_root.resolve(strict=True)
    if verify_input_artifact:
        verify_historical_inputs(input_root)
    manifest = json.loads((input_root / "manifest.json").read_bytes())
    plan = json.loads(plan_path.read_bytes())
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or plan.get("version") != 1:
        raise ValueError("unsupported historical role plan")
    fingerprint = plan.get("plan_fingerprint_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_fingerprint_sha256"}
    if hashlib.sha256(canonical(body)).hexdigest() != fingerprint:
        raise ValueError("historical role-plan fingerprint mismatch")
    expected = build_payload(input_root, verify_artifact=False)
    if plan != expected:
        raise ValueError("historical role plan differs from its bound input manifest")
    if plan.get("input_manifest_sha256") != sha256_file(input_root / "manifest.json"):
        raise ValueError("historical role plan references another input artifact")
    if sum(plan["totals"][role] for role in plan["totals"]) != manifest["totals"]["reliable_rows"]:
        raise ValueError("historical role-plan totals do not cover reliable rows")
    return plan


def training_windows(input_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    root = input_root.resolve(strict=True)
    result = []
    for item in plan["roles"]["train"]:
        feature_path = root / item["features"]["path"]
        ledger_path = root / item["ledger_inputs"]["path"]
        for path, entry in ((feature_path, item["features"]), (ledger_path, item["ledger_inputs"])):
            if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size_bytes"]
                    or sha256_file(path) != entry["sha256"]):
                raise ValueError(f"historical Train shard failed role-plan verification: {path}")
        result.append({"id": item["id"], "start_ns": item["start_ns"],
                       "end_ns": item["end_ns"], "feature_path": feature_path,
                       "ledger_path": ledger_path})
    return result


def create_role_plan(args: Any) -> dict[str, Any]:
    input_root = args.historical_inputs.resolve(strict=True)
    payload = build_payload(input_root)
    output = reject_output(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    if json.loads(output.read_bytes()) != payload:
        raise ValueError("written historical role plan differs from its frozen payload")
    return payload
