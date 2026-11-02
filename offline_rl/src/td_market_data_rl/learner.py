from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import resource
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard

from .contracts import (ACTION_TARGETS_LOTS, LOT_BTC, RELAY_ROOT,
                        MAX_TRAIN_MINUTES, MAX_UPDATES, MINUTE_NS,
                        canonical, read_role_plan, read_verified_features, reject_system_temp,
                        sha256_file)
from .factors import selected_factor_values, suites, registry_sha256
from .historical import training_windows, verify_plan as verify_historical_role_plan


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        if path.name.endswith(".zst"):
            reader = zstandard.ZstdDecompressor().stream_reader(stream)
            source = io.BufferedReader(reader, buffer_size=1024 * 1024)
        else:
            source = stream
        for index, raw in enumerate(source, 1):
            if not raw.endswith(b"\n") or len(raw) > 1024 * 1024:
                raise ValueError(f"{path.name} row {index} is truncated or oversized")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} row {index} is not an object")
            yield row


def trainer_code_sha256() -> str:
    digest = hashlib.sha256()
    rust_root = RELAY_ROOT / "bulk_engine"
    sources = (list(Path(__file__).parent.glob("*.py"))
               + [rust_root / "src" / "main.rs",
                  rust_root / "src" / "training_replay.rs",
                  rust_root / "src" / "replay_cache.rs",
                  rust_root / "Cargo.toml", rust_root / "Cargo.lock"])
    for source_file in sorted(sources):
        digest.update(source_file.name.encode() + b"\0")
        with source_file.open("rb") as source_stream:
            for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def prepare_historical_cache(train_sources: list[dict[str, Any]], stage: Path) -> tuple[Path, str]:
    """Parse immutable zstd JSONL once, in bounded parallel chunks, outside /tmp."""
    engine = RELAY_ROOT / "bulk_engine" / "target" / "release" / "td-market-data-bulk-engine"
    if not engine.is_file():
        raise ValueError("Rust replay engine is missing; build it with cargo build --release")
    engine_hash = sha256_file(engine)
    sources = []
    for item in train_sources:
        feature_path, ledger_path = Path(item["feature_path"]), Path(item["ledger_path"])
        sources.append({"id": item["id"], "start_ns": item["start_ns"],
                        "end_ns": item["end_ns"],
                        "minute_count": (item["end_ns"] - item["start_ns"]) // MINUTE_NS,
                        "feature_path": str(feature_path), "ledger_path": str(ledger_path),
                        "feature_sha256": sha256_file(feature_path),
                        "ledger_sha256": sha256_file(ledger_path)})
    key_payload = {"schema": "td_market_data_historical_replay_cache_key", "version": 1,
                   "engine_sha256": engine_hash, "trainer_code_sha256": trainer_code_sha256(),
                   "sources": sources}
    cache_key = hashlib.sha256(canonical(key_payload)).hexdigest()
    cache_root = Path(train_sources[0]["feature_path"]).parents[3] / f".td-market-data-rust-cache-{cache_key[:16]}"
    cache_manifest_path = cache_root / "cache-manifest.json"
    if cache_manifest_path.is_file():
        manifest = json.loads(cache_manifest_path.read_bytes())
        if (manifest.get("schema") != "td_market_data_historical_replay_cache"
                or manifest.get("version") != 1
                or manifest.get("cache_key_sha256") != cache_key
                or manifest.get("engine_sha256") != engine_hash):
            raise ValueError("existing Rust replay cache manifest does not match current inputs")
        for window in manifest.get("windows", []):
            if Path(window.get("path", "")).name != window.get("path"):
                raise ValueError("existing Rust replay cache contains an unsafe file path")
            cache_path = cache_root / window["path"]
            if not cache_path.is_file() or sha256_file(cache_path) != window["sha256"]:
                raise ValueError("existing Rust replay cache file is missing or corrupt")
        if len(manifest.get("windows", [])) != len(sources):
            raise ValueError("existing Rust replay cache window inventory differs")
        return cache_root, sha256_file(cache_manifest_path)

    # Use the dataset's parent filesystem for bounded staging, never system /tmp.
    cache_parent = cache_root.parent
    cache_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_root.name}.staging-", dir=cache_parent))
    request_path, report_path = stage / "cache-prepare-request.json", stage / "cache-prepare-report.json"
    request = {"schema": "td_market_data_training_prepare_request", "version": 1,
               "windows": [{key: item[key] for key in
                            ("id", "start_ns", "end_ns", "minute_count", "feature_path", "ledger_path")}
                           for item in sources],
               "output_directory": str(staging)}
    request_path.write_bytes(canonical(request) + b"\n")
    try:
        subprocess.run([str(engine), "training-prepare", str(request_path), str(report_path)],
                       check=True, capture_output=True, text=True)
        report = json.loads(report_path.read_bytes())
        if (report.get("schema") != "td_market_data_training_prepare_result"
                or report.get("version") != 1
                or len(report.get("windows", [])) != len(sources)):
            raise ValueError("Rust cache preparer returned an invalid report")
        windows = []
        by_id = {item["id"]: item for item in sources}
        for row in report["windows"]:
            source = by_id.get(row.get("window_id"))
            if source is None or row.get("rows") != source["minute_count"]:
                raise ValueError("Rust cache row count differs from the audited Train window")
            path = staging / row["path"]
            windows.append({"id": row["window_id"], "path": row["path"],
                            "rows": row["rows"], "sha256": sha256_file(path)})
        manifest = {"schema": "td_market_data_historical_replay_cache", "version": 1,
                    "cache_key_sha256": cache_key, "engine_sha256": engine_hash,
                    "trainer_code_sha256": key_payload["trainer_code_sha256"],
                    "source_windows": sources, "chunk_rows": report["chunk_rows"],
                    "workers_per_window": report["workers_per_window"], "windows": windows}
        (staging / "cache-manifest.json").write_bytes(canonical(manifest) + b"\n")
        if cache_root.exists():
            raise ValueError("Rust replay cache appeared concurrently; inspect before removing either cache")
        staging.rename(cache_root)
    except subprocess.CalledProcessError as error:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f"Rust historical cache preparation failed: {error.stderr.strip()}") from error
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        request_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
    return cache_root, sha256_file(cache_root / "cache-manifest.json")


def replay_historical_rust(stage: Path, update: int, train_sources: list[dict[str, Any]],
                           cache_root: Path,
                           factor_names: list[str], means: list[float], scales: list[float],
                           actor: torch.nn.Linear, seed: int, fee_bps: int,
                           slippage_bps: int, memory_limit_mb: int
                           ) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Run ordered, per-window accounting in the Rust engine.

    Audited Train windows are separate episodes and can be replayed concurrently;
    time inside each episode remains ordered because actions change position.
    Only compact decision experiences return to PyTorch for GPU gradient updates.
    """
    engine = RELAY_ROOT / "bulk_engine" / "target" / "release" / "td-market-data-bulk-engine"
    if not engine.is_file():
        raise ValueError(f"Rust replay engine is missing; build it with cargo build --release --manifest-path {RELAY_ROOT / 'bulk_engine' / 'Cargo.toml'}")
    work = stage / f"replay-{update:03d}"
    work.mkdir()
    request_path = work / "request.json"
    report_path = work / "report.json"
    engine_output = work / "experiences"
    parameter_weights = actor.weight.detach().cpu().tolist()
    parameter_bias = actor.bias.detach().cpu().tolist()
    request = {
        "schema": "td_market_data_training_replay_request", "version": 1,
        "windows": [{"id": item["id"],
                     "minute_count": (item["end_ns"] - item["start_ns"]) // MINUTE_NS,
                     "cache_path": str(cache_root / f"window-{index:02}.tdcache")}
                    for index, item in enumerate(train_sources)],
        "factor_names": factor_names, "means": means, "scales": scales,
        "weights": parameter_weights, "bias": parameter_bias,
        "seed": int(seed + update), "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "memory_limit_mb": memory_limit_mb,
        "output_directory": str(engine_output),
    }
    request_path.write_bytes(canonical(request) + b"\n")
    try:
        subprocess.run([str(engine), "training-replay", str(request_path), str(report_path)],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Rust historical replay failed: {error.stderr.strip()}") from error
    report = json.loads(report_path.read_bytes())
    if (report.get("schema") != "td_market_data_training_replay_result"
            or report.get("version") != 1
            or len(report.get("windows", [])) != len(train_sources)):
        raise ValueError("Rust replay engine returned an invalid report")
    by_id = {item["id"]: item for item in train_sources}
    episode_experiences = []
    window_stats = []
    dtype = np.dtype([("state", "<f4", (len(factor_names) + 1,)),
                      ("action", "u1"), ("reward", "<f4")], align=False)
    expected_record_size = (len(factor_names) + 1) * 4 + 5
    for row in report["windows"]:
        source = by_id.get(row.get("window_id"))
        if source is None:
            raise ValueError("Rust replay returned an unknown Train window")
        path = engine_output / row["experiences_path"]
        with path.open("rb") as stream:
            header = stream.read(20)
            if len(header) != 20 or header[:8] != b"TDRP1\0\0\0":
                raise ValueError("Rust replay experience header is invalid")
            width = int.from_bytes(header[8:12], "little")
            count = int.from_bytes(header[12:20], "little")
            if width != len(factor_names) + 1 or count != row["experience_count"]:
                raise ValueError("Rust replay experience dimensions do not match report")
            records = np.fromfile(stream, dtype=dtype, count=count)
            if records.size != count or stream.read(1):
                raise ValueError("Rust replay experience file is truncated or oversized")
            if dtype.itemsize != expected_record_size:
                raise ValueError("host experience record layout is not packed as specified")
        episode_experiences.append((records["state"], records["action"].astype(np.int64),
                                    records["reward"]))
        window_stats.append({"window_id": source["id"], **row["stats"]})
    shutil.rmtree(work)
    return episode_experiences, window_stats


def fit_normalizer(path: Path, start: int, end: int) -> tuple[list[float], list[float], int]:
    _, means, scales, count = fit_factor_normalizer(path, start, end, "raw_baseline_v1")
    return means, scales, count


def fit_factor_normalizer(path: Path, start: int, end: int,
                          factor_set_id: str) -> tuple[list[str], list[float], list[float], int]:
    return fit_factor_normalizer_paths([(path, start, end)], factor_set_id)


def fit_factor_normalizer_paths(paths: list[tuple[Path, int, int]],
                                factor_set_id: str) -> tuple[list[str], list[float], list[float], int]:
    factor_names = suites().get(factor_set_id)
    if factor_names is None:
        raise ValueError(f"unknown factor set: {factor_set_id}")
    means = np.zeros(len(factor_names), dtype=np.float64)
    m2 = np.zeros(len(factor_names), dtype=np.float64)
    count = 0
    for path, start, end in paths:
        for row in jsonl(path):
            timestamp = row["decision_at_ns"]
            if timestamp < start or timestamp >= end:
                continue
            if not row["feature_ready"]:
                continue
            _, factor_values = selected_factor_values(row["features"], factor_set_id)
            vector = np.array(factor_values, dtype=np.float64)
            count += 1
            delta = vector - means
            means += delta / count
            m2 += delta * (vector - means)
    if count < 2:
        raise ValueError("Train window needs at least two complete feature rows to fit normalization")
    scale = np.sqrt(m2 / count)
    scale[scale < 1e-12] = 1.0
    return factor_names, means.tolist(), scale.tolist(), count


def rows_in_window(feature_path: Path, ledger_path: Path, start: int, end: int) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for feature, ledger in zip(jsonl(feature_path), jsonl(ledger_path), strict=True):
        if feature["decision_at_ns"] < start:
            continue
        if feature["decision_at_ns"] >= end:
            break
        yield feature, ledger


def select_device(requested: str, max_gpu_memory_mb: int) -> tuple[torch.device, str | None]:
    if requested == "cpu":
        return torch.device("cpu"), None
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but PyTorch reports no usable CUDA device")
        device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(device).total_memory
        cap = max_gpu_memory_mb * 1024 * 1024
        if cap <= 0 or cap > total:
            raise ValueError("GPU memory cap must be positive and no larger than device memory")
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        return device, None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        total = torch.cuda.get_device_properties(device).total_memory
        cap = max_gpu_memory_mb * 1024 * 1024
        if 0 < cap <= total:
            try:
                torch.cuda.set_per_process_memory_fraction(cap / total, device)
                return device, None
            except RuntimeError as error:
                return torch.device("cpu"), f"CUDA memory limit unavailable: {error}"
    return torch.device("cpu"), "CUDA unavailable; using CPU"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


def apply_fill(position_lots: int, target_lots: int, ledger: dict[str, Any], fee_bps: float,
               slippage_bps: float) -> tuple[int, float, float, int]:
    """Return position, transaction costs, effective fill price and filled lots."""
    delta = target_lots - position_lots
    if delta == 0:
        return position_lots, 0.0, 0.0, 0
    price = ledger["execution_vwap"]
    volume = ledger["execution_base_volume"]
    if price is None or volume <= 0:
        return position_lots, 0.0, 0.0, 0
    cap_lots = math.floor(volume * 0.01 / LOT_BTC)
    max_lots = abs(delta)
    if target_lots == 0 or position_lots * target_lots < 0:
        max_lots = abs(position_lots)  # a reversal is closed first, then reconsidered next decision
    requested_lots = min(abs(delta), cap_lots, max_lots)
    if requested_lots == 0:
        return position_lots, 0.0, 0.0, 0
    reducing = (target_lots == 0 or position_lots * target_lots < 0
                or (position_lots * target_lots > 0 and abs(target_lots) < abs(position_lots)))
    if not reducing:
        requested_lots = min(requested_lots, math.floor(100.0 / (price * LOT_BTC)))
        if requested_lots * price * LOT_BTC < 50.0:
            return position_lots, 0.0, 0.0, 0
    signed_fill_lots = min(requested_lots, abs(delta)) * (1 if delta > 0 else -1)
    raw_notional = abs(signed_fill_lots * LOT_BTC * price)
    effective_price = price * (1 + math.copysign(slippage_bps / 10_000, signed_fill_lots))
    fee = raw_notional * fee_bps / 10_000
    slippage = abs(effective_price - price) * abs(signed_fill_lots * LOT_BTC)
    after = position_lots + signed_fill_lots
    # Protocol never permits crossing the desired target or exceeding one lot.
    if abs(after) > 1 or (delta > 0 and after > target_lots) or (delta < 0 and after < target_lots):
        raise ValueError("fill exceeded the frozen position/target bound")
    return after, fee + slippage, effective_price, signed_fill_lots


def simulate_episode(feature_path: Path, ledger_path: Path, window: dict[str, int],
                     select_action: Any, fee_bps: float, slippage_bps: float,
                     memory_limit_mb: int, collect_experiences: bool = False
                     ) -> tuple[list[tuple[list[float], int, float]], dict[str, Any]]:
    start, end = window["start_ns"], window["end_ns"]
    position = 0
    previous_mark: float | None = None
    equity = 10_000.0
    max_equity = equity
    min_drawdown = 0.0
    pending_cost = 0.0
    pending_reward = 0.0
    experiences: list[tuple[list[float], int, float]] = []
    current_state: list[float] | None = None
    current_action = 1
    decisions = fills = 0
    total_fees = total_slippage = funding_pnl = gross_pnl = 0.0
    traded_btc = traded_notional = 0.0
    ready_minutes = valid_mark_minutes = exposure_minutes = 0
    minute_count = 0
    for feature, ledger in rows_in_window(feature_path, ledger_path, start, end):
        minute_count += 1
        if minute_count % 4096 == 0:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if rss_mb > memory_limit_mb:
                raise ValueError(f"process RSS exceeded configured {memory_limit_mb} MiB limit")
        price = feature.get("mark_price")
        if price is not None:
            valid_mark_minutes += 1
        if feature["feature_ready"]:
            ready_minutes += 1
        if position:
            exposure_minutes += 1
        if price is not None and previous_mark is not None and position:
            pnl = position * LOT_BTC * (price - previous_mark)
            equity += pnl
            pending_reward += pnl
            gross_pnl += pnl
        previous_mark = price
        for settlement in ledger["funding_settlements"]:
            cashflow = -position * LOT_BTC * settlement["mark_price"] * settlement["funding_rate"]
            equity += cashflow
            pending_reward += cashflow
            funding_pnl += cashflow
        max_equity = max(max_equity, equity)
        if max_equity > 0:
            min_drawdown = min(min_drawdown, (equity - max_equity) / max_equity)
        if not feature["decision_due"]:
            continue
        if collect_experiences and current_state is not None:
            reward = pending_reward - pending_cost
            experiences.append((current_state, current_action, reward / 10_000.0))
        pending_reward = 0.0
        pending_cost = 0.0
        if not feature["feature_ready"]:
            current_state = None
            continue
        action, state = select_action(feature, position)
        if not isinstance(action, int) or isinstance(action, bool) or action not in (0, 1, 2):
            raise ValueError("policy returned an invalid categorical action")
        target_lots = ACTION_TARGETS_LOTS[action]
        position, cost, _, filled = apply_fill(position, target_lots, ledger, fee_bps, slippage_bps)
        if filled:
            fills += 1
            traded_btc += abs(filled) * LOT_BTC
            traded_notional += abs(filled) * LOT_BTC * ledger["execution_vwap"]
            fees = abs(filled * LOT_BTC * (ledger["execution_vwap"] or 0)) * fee_bps / 10_000
            slippage = cost - fees
            equity -= cost
            pending_cost += cost
            total_fees += fees
            total_slippage += slippage
        # Margin guard is adverse liquidation at the explicit maintenance margin.
        if price is not None and position and abs(position) * LOT_BTC * price * 0.05 >= equity:
            liquidation_fee = abs(position) * LOT_BTC * price * 0.0025
            equity -= liquidation_fee
            pending_cost += liquidation_fee
            total_fees += liquidation_fee
            position = 0
        max_equity = max(max_equity, equity)
        if max_equity > 0:
            min_drawdown = min(min_drawdown, (equity - max_equity) / max_equity)
        current_state, current_action = state, action
        decisions += 1
    if collect_experiences and current_state is not None:
        experiences.append((current_state, current_action,
                            (pending_reward - pending_cost) / 10_000.0))
    stats = {"minutes": minute_count, "decisions": decisions, "fills": fills,
             "gross_price_pnl_usdt": gross_pnl, "funding_pnl_usdt": funding_pnl,
             "fees_usdt": total_fees, "slippage_usdt": total_slippage,
             "ending_equity_usdt": equity, "max_drawdown_fraction": min_drawdown,
             "net_pnl_usdt": equity - 10_000.0,
             "net_return_fraction": (equity - 10_000.0) / 10_000.0,
             "ending_position_lots": position, "closed_at_end": position == 0,
             "traded_btc": traded_btc, "traded_notional_usdt": traded_notional,
             "exposure_minutes": exposure_minutes, "mark_coverage_fraction": valid_mark_minutes / minute_count if minute_count else 0.0,
             "feature_ready_fraction": ready_minutes / minute_count if minute_count else 0.0,
             "mean_policy_reward_fraction": float(np.mean([item[2] for item in experiences])) if experiences else 0.0}
    return experiences, stats


def train(args: Any) -> dict[str, Any]:
    if not 1 <= args.updates <= MAX_UPDATES:
        raise ValueError(f"updates must be 1..{MAX_UPDATES}")
    if not (0 < args.max_memory_mb <= 8192 and 0 < args.max_gpu_memory_mb <= 8192):
        raise ValueError("memory caps must be between 1 and 8192 MiB")
    if args.fee_bps not in (5, 10, 15, 20) or args.slippage_bps not in (1, 5, 10):
        raise ValueError("costs must be selected from the frozen protocol scenarios")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("training output must be a new create-only directory")
    reject_system_temp(output, "training output")
    if output.resolve() == RELAY_ROOT or RELAY_ROOT in output.resolve().parents:
        raise ValueError("training artifacts must be stored outside the source repository")
    if args.role_plan.is_symlink():
        raise ValueError("role plan must not be a symlink")
    plan_path = args.role_plan.resolve(strict=True)
    historical_root = getattr(args, "historical_inputs", None)
    historical_manifest = None
    if historical_root is not None:
        if args.features is not None or args.dataset is not None:
            raise ValueError("historical training cannot mix Relay capture inputs with Vision history")
        historical_root = historical_root.resolve(strict=True)
        reject_system_temp(historical_root, "historical input")
        plan = verify_historical_role_plan(plan_path, historical_root, verify_input_artifact=False)
        historical_manifest = json.loads((historical_root / "manifest.json").read_bytes())
        train_sources = training_windows(historical_root, plan)
        train_window = {"windows": [{"id": item["id"], "start_ns": item["start_ns"],
                                     "end_ns": item["end_ns"]} for item in train_sources]}
        feature_manifest_hash = sha256_file(historical_root / "manifest.json")
        source_manifest_hash = historical_manifest["source_manifests"]["bulk_audit_sha256"]
        source_manifest = {"source_capture_set_sha256": None}
        feature_protocol_sha256 = historical_manifest["adapter_configuration_sha256"]
        role_plan_hash = sha256_file(plan_path)
        normalizer_paths = [(item["feature_path"], item["start_ns"], item["end_ns"])
                            for item in train_sources]
        train_minutes = sum((item["end_ns"] - item["start_ns"]) // MINUTE_NS
                            for item in train_sources)
        input_kind = "audited_vision_historical"
    else:
        if args.features is None or args.dataset is None:
            raise ValueError("training requires either --historical-inputs or both --features and --dataset")
        feature_path, dataset_path = args.features.resolve(strict=True), args.dataset.resolve(strict=True)
        reject_system_temp(feature_path, "feature input")
        reject_system_temp(dataset_path, "Dataset input")
        reject_system_temp(plan_path, "role-plan input")
        feature_manifest, source_manifest = read_verified_features(feature_path, dataset_path)
        feature_manifest_hash = sha256_file(feature_path / "manifest.json")
        source_manifest_hash = sha256_file(dataset_path / "manifest.json")
        plan = read_role_plan(plan_path, feature_manifest, source_manifest,
                              feature_manifest_hash, source_manifest_hash)
        train_window = plan["windows"]["train"]
        train_sources = [{"id": "train", "start_ns": train_window["start_ns"],
                          "end_ns": train_window["end_ns"],
                          "feature_path": feature_path / "features.jsonl",
                          "ledger_path": feature_path / "ledger_inputs.jsonl"}]
        normalizer_paths = [(feature_path / "features.jsonl", train_window["start_ns"], train_window["end_ns"])]
        train_minutes = (train_window["end_ns"] - train_window["start_ns"]) // MINUTE_NS
        feature_protocol_sha256 = feature_manifest["protocol_sha256"]
        role_plan_hash = sha256_file(plan_path)
        input_kind = "relay_capture"
    if train_minutes > MAX_TRAIN_MINUTES:
        raise ValueError("Train role exceeds the one-year hard cap")
    factor_names, means, scales, normalizer_rows = fit_factor_normalizer_paths(
        normalizer_paths, args.factor_set)
    device, fallback = select_device(args.device, args.max_gpu_memory_mb)
    set_seed(args.seed)
    actor = torch.nn.Linear(len(factor_names) + 1, len(ACTION_TARGETS_LOTS), bias=True, device=device)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    historical_training = historical_manifest is not None
    learning_rate = 30.0 if historical_training else 0.1
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate)
    entropy_bonus = 0.0
    baseline = 0.0
    baseline_initialized = False
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        replay_cache_root = None
        replay_cache_manifest_hash = None
        if historical_training:
            replay_cache_root, replay_cache_manifest_hash = prepare_historical_cache(train_sources, stage)
        trace_path = stage / "training_trace.jsonl.zst"
        trace_digest = hashlib.sha256()
        episode_stats = []
        transitions = 0
        with trace_path.open("xb") as raw_trace:
            with zstandard.ZstdCompressor(level=3).stream_writer(raw_trace, closefd=False) as compressed:
                for update in range(args.updates):
                    if historical_training:
                        episode_experiences, window_stats = replay_historical_rust(
                            stage, update, train_sources, replay_cache_root,
                            factor_names, means, scales,
                            actor, args.seed, args.fee_bps, args.slippage_bps,
                            args.max_memory_mb)
                    else:
                        episode_experiences, window_stats = [], []
                        rng = torch.Generator(device=device).manual_seed(args.seed + update)

                        def select_action(feature: dict[str, Any], position: int) -> tuple[int, list[float]]:
                            _, values = selected_factor_values(feature["features"], args.factor_set)
                            base = [(value - means[index]) / scales[index]
                                    for index, value in enumerate(values)]
                            state = base + [position]
                            tensor = torch.tensor(state, dtype=torch.float32, device=device)
                            probabilities = torch.softmax(actor(tensor), dim=0)
                            return int(torch.multinomial(probabilities, 1, generator=rng).item()), state

                        for source_window in train_sources:
                            experiences, stats = simulate_episode(
                                source_window["feature_path"], source_window["ledger_path"],
                                {"start_ns": source_window["start_ns"], "end_ns": source_window["end_ns"]},
                                select_action, args.fee_bps, args.slippage_bps,
                                args.max_memory_mb, collect_experiences=True)
                            if not experiences:
                                raise ValueError(f"Train window {source_window['id']} contains no policy decisions")
                            episode_experiences.append((
                                np.asarray([item[0] for item in experiences], dtype=np.float32),
                                np.asarray([item[1] for item in experiences], dtype=np.int64),
                                np.asarray([item[2] for item in experiences], dtype=np.float32)))
                            window_stats.append({"window_id": source_window["id"], **stats})
                    if not episode_experiences or sum(len(item[1]) for item in episode_experiences) == 0:
                        raise ValueError("Train window contains no ready policy decisions")
                    state_array = np.concatenate([item[0] for item in episode_experiences])
                    action_array = np.concatenate([item[1] for item in episode_experiences])
                    reward_array = np.concatenate([item[2] for item in episode_experiences])
                    states = torch.as_tensor(state_array, dtype=torch.float32, device=device)
                    actions = torch.as_tensor(action_array, dtype=torch.int64, device=device)
                    rewards = torch.as_tensor(reward_array, dtype=torch.float32, device=device)
                    returns = torch.empty_like(rewards)
                    offset = 0
                    for episode in episode_experiences:
                        running_return = torch.zeros((), dtype=torch.float32, device=device)
                        for index in range(len(episode[2]) - 1, -1, -1):
                            running_return = rewards[offset + index] + 0.99 * running_return
                            returns[offset + index] = running_return
                        offset += len(episode[2])
                    logits = actor(states)
                    all_log_probs = torch.log_softmax(logits, dim=1)
                    log_probs = all_log_probs.gather(1, actions[:, None]).squeeze(1)
                    if historical_training:
                        # Returns are fractional equity changes, often around 1e-5.
                        # Centering and scaling within each Train update keeps the
                        # REINFORCE gradient numerically useful without looking at
                        # Validation/Test/OOS outcomes.
                        advantages = returns - returns.mean()
                        advantage_scale = advantages.std(unbiased=False)
                        if float(advantage_scale.item()) > 1e-8:
                            advantages = advantages / advantage_scale
                        else:
                            advantages = torch.zeros_like(advantages)
                    else:
                        advantages = returns - baseline
                        advantage_scale = advantages.std(unbiased=False)
                    probabilities = all_log_probs.exp()
                    entropy = -(probabilities * all_log_probs).sum(dim=1).mean()
                    action_counts = [int((actions == index).sum().item())
                                     for index in range(len(ACTION_TARGETS_LOTS))]
                    action_return_means = [
                        float(returns[actions == index].mean().item())
                        if action_counts[index] else 0.0
                        for index in range(len(ACTION_TARGETS_LOTS))]
                    policy_loss = -(log_probs * advantages.detach()).mean()
                    loss = policy_loss - entropy_bonus * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    optimizer.step()
                    mean_return = float(returns.mean().item())
                    baseline = mean_return if not baseline_initialized else 0.9 * baseline + 0.1 * mean_return
                    baseline_initialized = True
                    if len(window_stats) == 1:
                        stats = {key: value for key, value in window_stats[0].items()
                                 if key != "window_id"}
                    else:
                        stats = {"minutes": sum(row["minutes"] for row in window_stats),
                                 "decisions": sum(row["decisions"] for row in window_stats),
                                 "fills": sum(row["fills"] for row in window_stats),
                                 "net_pnl_usdt": sum(row["net_pnl_usdt"] for row in window_stats),
                                 "ending_equity_usdt_per_window_mean": sum(row["ending_equity_usdt"] for row in window_stats) / len(window_stats),
                                 "max_drawdown_fraction_worst_window": min(row["max_drawdown_fraction"] for row in window_stats),
                                 "gross_price_pnl_usdt": sum(row["gross_price_pnl_usdt"] for row in window_stats),
                                 "funding_pnl_usdt": sum(row["funding_pnl_usdt"] for row in window_stats),
                                 "fees_usdt": sum(row["fees_usdt"] for row in window_stats),
                                 "slippage_usdt": sum(row["slippage_usdt"] for row in window_stats),
                                 "traded_btc": sum(row["traded_btc"] for row in window_stats),
                                 "traded_notional_usdt": sum(row["traded_notional_usdt"] for row in window_stats),
                                 "window_stats": window_stats}
                    stats.update({"update": update + 1, "loss": float(loss.item()),
                                  "return_mean": mean_return, "baseline": baseline,
                                  "advantage_std": float(advantage_scale.item()),
                                  "policy_loss": float(policy_loss.item()),
                                  "gradient_norm_before_clip": float(gradient_norm.item()),
                                  "entropy": float(entropy.item()),
                                  "action_counts_short_flat_long": action_counts,
                                  "discounted_return_mean_short_flat_long": action_return_means})
                    episode_stats.append(stats)
                    transitions += len(action_array)
                    line = canonical(stats) + b"\n"
                    compressed.write(line)
                    trace_digest.update(line)
        model = {"schema": "td_market_data_usdm_rl_linear_policy", "version": 1,
                 "feature_names": factor_names, "factor_set_id": args.factor_set,
                 "factor_set_sha256": registry_sha256(),
                 "position_state": "current_position_lots in [-1,0,1]",
                 "action_target_lots": list(ACTION_TARGETS_LOTS), "weights": actor.weight.detach().cpu().tolist(),
                 "bias": actor.bias.detach().cpu().tolist(), "normalizer": {"fit_role": "train",
                 "rows": normalizer_rows, "mean": means, "scale": scales}}
        model_path = stage / "model.json"
        model_path.write_bytes(json.dumps(model, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        run = {"schema": "td_market_data_usdm_rl_training", "version": 1,
               "input_kind": input_kind,
               "source_dataset_manifest_sha256": source_manifest_hash,
               "features_manifest_sha256": feature_manifest_hash,
               "role_plan_sha256": role_plan_hash,
               "historical_input_manifest_sha256": feature_manifest_hash if historical_manifest else None,
               "historical_role_plan_fingerprint_sha256": plan.get("plan_fingerprint_sha256") if historical_manifest else None,
               "train_windows": [{"id": item["id"], "start_ns": item["start_ns"],
                                  "end_ns": item["end_ns"]} for item in train_sources],
               "source_capture_set_sha256": source_manifest.get("source_capture_set_sha256"),
               "feature_protocol_sha256": feature_protocol_sha256,
               "trainer_code_sha256": trainer_code_sha256(),
               "role": "train", "train_window": train_window,
               "factor_set_id": args.factor_set, "factor_set_sha256": registry_sha256(),
               "algorithm": {"name": ("linear_categorical_reinforce_train_standardized_v2"
                                        if historical_training else "linear_categorical_reinforce"),
                             "gamma": 0.99,
                             "baseline": ("per-update return mean"
                                          if historical_training
                                          else "scalar exponential moving average alpha=0.1"),
                             "optimizer": "SGD", "learning_rate": learning_rate,
                             "gradient_norm_clip": 10.0, "entropy_bonus": entropy_bonus,
                             "historical_replay_engine": ("Rust ordered account replay over parallel chunk cache v1"
                                                           if historical_training else "Python ordered replay"),
                             "historical_replay_cache_manifest_sha256": replay_cache_manifest_hash,
                             "historical_action_sampler": ("xorshift64star-v1"
                                                           if historical_training else "torch.multinomial"),
                             "historical_advantage_normalization":
                                 "per-update centered and population-standardized; epsilon=1e-8"
                                 if historical_training else "disabled"},
               "accounting": {"fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps,
                              "max_position_lots": 1, "entry_notional_cap_usdt": 100.0,
                              "participation_fraction": 0.01, "initial_equity_usdt": 10000.0,
                              "maintenance_margin_fraction": 0.05,
                              "liquidation_surcharge_bps": 25},
               "seed": args.seed, "requested_device": args.device,
               "selected_device": str(device), "device_fallback_reason": fallback,
               "updates": args.updates, "transitions": transitions,
               "limits": {"max_train_minutes": MAX_TRAIN_MINUTES, "max_updates": MAX_UPDATES,
                          "max_memory_mb": args.max_memory_mb, "max_gpu_memory_mb": args.max_gpu_memory_mb},
               "runtime_versions": {"torch": torch.__version__, "numpy": np.__version__,
                                    "cuda": torch.version.cuda,
                                    "rust_replay_engine_sha256": (
                                        sha256_file(RELAY_ROOT / "bulk_engine" / "target" / "release" / "td-market-data-bulk-engine")
                                        if historical_training else None)},
               "episode_stats": episode_stats,
               "model_sha256": sha256_file(model_path),
               "trace_sha256": sha256_file(trace_path),
               "trace_uncompressed_lines_sha256": trace_digest.hexdigest()}
        manifest_path = stage / "manifest.json"
        manifest_path.write_bytes(json.dumps(run, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")
        for path in stage.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.chmod(model_path, 0o444)
        os.chmod(trace_path, 0o444)
        os.chmod(manifest_path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
        parent_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return run
    except BaseException:
        try:
            os.chmod(stage, 0o755)
            for item in stage.iterdir():
                os.chmod(item, 0o644)
        except OSError:
            pass
        shutil.rmtree(stage, ignore_errors=True)
        raise
