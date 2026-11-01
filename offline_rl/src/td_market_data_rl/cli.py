from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import zstandard

from .contracts import (FEATURE_NAMES, MAX_TRAIN_MINUTES, MINUTE_NS, RELAY_ROOT,
                        read_role_plan, read_verified_features, reject_system_temp, sha256_file)
from .learner import train


def parse_time(value: str) -> int:
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use an ISO-8601 UTC timestamp, e.g. 2026-01-01T00:00:00Z") from error
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    if moment.utcoffset().total_seconds() != 0:
        raise argparse.ArgumentTypeError("timestamps must be UTC")
    ns = int(moment.timestamp()) * 1_000_000_000
    if ns % MINUTE_NS:
        raise argparse.ArgumentTypeError("role boundaries must align to a UTC minute")
    return ns


def role_plan(args: Any) -> dict[str, Any]:
    features, dataset = args.features.resolve(strict=True), args.dataset.resolve(strict=True)
    reject_system_temp(features, "feature input")
    reject_system_temp(dataset, "Dataset input")
    feature_manifest, source_manifest = read_verified_features(features, dataset)
    windows = {"train": {"start_ns": args.train_start, "end_ns": args.train_end},
               "validation": {"start_ns": args.validation_start, "end_ns": args.validation_end}}
    for role in ("test", "oos"):
        start, end = getattr(args, f"{role}_start"), getattr(args, f"{role}_end")
        if (start is None) != (end is None):
            raise ValueError(f"{role} needs both start and end boundaries")
        if start is not None:
            windows[role] = {"start_ns": start, "end_ns": end}
    path = args.output.absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("role-plan output must be new; role assignments are immutable")
    reject_system_temp(path, "role-plan output")
    if path.resolve() == RELAY_ROOT or RELAY_ROOT in path.resolve().parents:
        raise ValueError("role plans must be stored outside the source repository")
    windows_for_validation = {"windows": windows, "protocol": "chronological_half_open_no_random_split_v1",
                              "schema": "td_market_data_usdm_rl_role_plan", "version": 1,
                              "features_manifest_sha256": sha256_file(features / "manifest.json"),
                              "source_dataset_manifest_sha256": sha256_file(dataset / "manifest.json"),
                              "feature_window": feature_manifest["window"],
                              "source_capture_set_sha256": source_manifest.get("source_capture_set_sha256")}
    read_role_plan_from_object(windows_for_validation, feature_manifest, source_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(windows_for_validation, sort_keys=True, indent=2).encode() + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return windows_for_validation


def read_role_plan_from_object(plan: dict[str, Any], feature_manifest: dict[str, Any],
                               source_manifest: dict[str, Any]) -> None:
    windows = plan["windows"]
    ordered = [windows[key] for key in ("train", "validation", "test", "oos") if key in windows]
    for window in ordered:
        start, end = window["start_ns"], window["end_ns"]
        if start >= end or start % MINUTE_NS or end % MINUTE_NS:
            raise ValueError("role windows must be nonempty and minute aligned")
        if start < feature_manifest["window"]["start_ns"] or end > feature_manifest["window"]["end_ns"]:
            raise ValueError("role windows must stay inside the fixed feature Dataset")
    if any(right["start_ns"] < left["end_ns"] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("role windows must be chronological and non-overlapping")
    if (ordered[0]["start_ns"] != feature_manifest["window"]["start_ns"]
            or ordered[-1]["end_ns"] != feature_manifest["window"]["end_ns"]
            or any(left["end_ns"] != right["start_ns"] for left, right in zip(ordered, ordered[1:]))):
        raise ValueError("role windows must cover the fixed Dataset; do not discard gaps silently")
    if (windows["train"]["end_ns"] - windows["train"]["start_ns"]) // MINUTE_NS > MAX_TRAIN_MINUTES:
        raise ValueError("Train role exceeds the one-year cap")


def verify_run(args: Any) -> dict[str, Any]:
    if args.artifact.is_symlink():
        raise ValueError("run artifact must not be a symlink")
    artifact = args.artifact.resolve(strict=True)
    reject_system_temp(artifact, "training artifact")
    if not artifact.is_dir():
        raise ValueError("run artifact must be a regular directory")
    expected = {"manifest.json", "model.json", "training_trace.jsonl.zst"}
    entries = list(artifact.iterdir())
    if ({item.name for item in entries} != expected
            or any(item.is_symlink() or not item.is_file() for item in entries)):
        raise ValueError("run artifact has missing or unmanifested files")
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    model = json.loads((artifact / "model.json").read_bytes())
    if manifest.get("schema") != "td_market_data_usdm_rl_training" or manifest.get("version") != 1:
        raise ValueError("unsupported run manifest")
    if (sha256_file(artifact / "model.json") != manifest.get("model_sha256")
            or sha256_file(artifact / "training_trace.jsonl.zst") != manifest.get("trace_sha256")):
        raise ValueError("run artifact file hash mismatch")
    if model.get("feature_names") != list(FEATURE_NAMES) or len(model.get("weights", [])) != 3:
        raise ValueError("model feature/action shape invalid")
    rows = model["weights"]
    if (any(len(row) != 8 or any(not math.isfinite(value) for value in row) for row in rows)
            or len(model.get("bias", [])) != 3
            or any(not math.isfinite(value) for value in model["bias"])):
        raise ValueError("model parameters are malformed")
    count = 0
    trace_digest = hashlib.sha256()
    with (artifact / "training_trace.jsonl.zst").open("rb") as compressed:
        reader = zstandard.ZstdDecompressor().stream_reader(compressed)
        buffer = b""
        while chunk := reader.read(1024 * 1024):
            buffer += chunk
            if len(buffer) > 2 * 1024 * 1024 and b"\n" not in buffer:
                raise ValueError("training trace row exceeds the 2 MiB verifier limit")
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if not line:
                    continue
                json.loads(line)
                trace_digest.update(line + b"\n")
                count += 1
        if buffer:
            raise ValueError("training trace has a truncated final JSONL record")
    if (count != manifest.get("updates")
            or trace_digest.hexdigest() != manifest.get("trace_uncompressed_lines_sha256")):
        raise ValueError("trace update count differs from manifest")
    if bool(args.dataset) != bool(args.features):
        raise ValueError("verify-run requires both --dataset and --features, or neither")
    if args.dataset and args.features:
        reject_system_temp(args.features.resolve(strict=True), "feature input")
        reject_system_temp(args.dataset.resolve(strict=True), "Dataset input")
        feature_manifest, source_manifest = read_verified_features(args.features, args.dataset)
        if (manifest["features_manifest_sha256"] != sha256_file(args.features / "manifest.json")
                or manifest["source_dataset_manifest_sha256"] != sha256_file(args.dataset / "manifest.json")
                or manifest["source_capture_set_sha256"] != source_manifest.get("source_capture_set_sha256")):
            raise ValueError("artifact source hashes differ from the provided inputs")
    return manifest


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="td-market-data-rl")
    commands = parser.add_subparsers(dest="command", required=True)
    roles = commands.add_parser("role-plan", help="freeze chronological Train/Validation(/Test/OOS) windows")
    roles.add_argument("--features", type=Path, required=True)
    roles.add_argument("--dataset", type=Path, required=True)
    for role in ("train", "validation", "test", "oos"):
        roles.add_argument(f"--{role}-start", type=parse_time, required=role in ("train", "validation"))
        roles.add_argument(f"--{role}-end", type=parse_time, required=role in ("train", "validation"))
    roles.add_argument("--output", type=Path, required=True)
    roles.set_defaults(func=role_plan)
    learner = commands.add_parser("train", help="train the linear categorical policy on Train only")
    learner.add_argument("--features", type=Path, required=True)
    learner.add_argument("--dataset", type=Path, required=True)
    learner.add_argument("--role-plan", type=Path, required=True)
    learner.add_argument("--output", type=Path, required=True)
    learner.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    learner.add_argument("--updates", type=int, default=1)
    learner.add_argument("--seed", type=int, default=101)
    learner.add_argument("--fee-bps", type=int, default=10)
    learner.add_argument("--slippage-bps", type=int, default=5)
    learner.add_argument("--max-memory-mb", type=int, default=4096)
    learner.add_argument("--max-gpu-memory-mb", type=int, default=2048)
    learner.set_defaults(func=train)
    verifier = commands.add_parser("verify-run", help="verify immutable run hashes and model shape")
    verifier.add_argument("--artifact", type=Path, required=True)
    verifier.add_argument("--features", type=Path)
    verifier.add_argument("--dataset", type=Path)
    verifier.set_defaults(func=verify_run)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        result = args.func(args)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError,
            torch.cuda.OutOfMemoryError) as error:
        raise SystemExit(f"offline USD-M RL rejected: {error}") from error
    if args.command == "role-plan":
        print(f"immutable role plan created: {args.output.resolve()}")
    elif args.command == "train":
        print(f"offline RL training complete: device={result['selected_device']} updates={result['updates']} transitions={result['transitions']}")
        print(f"artifact: {args.output.resolve()}")
    else:
        print(f"training artifact valid: updates={result['updates']} transitions={result['transitions']}")
