from __future__ import annotations

import hashlib
import json
import math
import os
import random
import resource
import shutil
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


def jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            if not raw.endswith(b"\n") or len(raw) > 1024 * 1024:
                raise ValueError(f"{path.name} row {index} is truncated or oversized")
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name} row {index} is not an object")
            yield row


def fit_normalizer(path: Path, start: int, end: int) -> tuple[list[float], list[float], int]:
    _, means, scales, count = fit_factor_normalizer(path, start, end, "raw_baseline_v1")
    return means, scales, count


def fit_factor_normalizer(path: Path, start: int, end: int,
                          factor_set_id: str) -> tuple[list[str], list[float], list[float], int]:
    factor_names = suites().get(factor_set_id)
    if factor_names is None:
        raise ValueError(f"unknown factor set: {factor_set_id}")
    means = np.zeros(len(factor_names), dtype=np.float64)
    m2 = np.zeros(len(factor_names), dtype=np.float64)
    count = 0
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
    with feature_path.open("rb") as features, ledger_path.open("rb") as ledger:
        for feature_raw, ledger_raw in zip(features, ledger, strict=True):
            feature = json.loads(feature_raw)
            if feature["decision_at_ns"] < start:
                continue
            if feature["decision_at_ns"] >= end:
                break
            yield feature, json.loads(ledger_raw)


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
    feature_path, dataset_path = args.features.resolve(strict=True), args.dataset.resolve(strict=True)
    reject_system_temp(feature_path, "feature input")
    reject_system_temp(dataset_path, "Dataset input")
    reject_system_temp(args.role_plan.resolve(strict=True), "role-plan input")
    if output.resolve() == RELAY_ROOT or RELAY_ROOT in output.resolve().parents:
        raise ValueError("training artifacts must be stored outside the source repository")
    feature_manifest, source_manifest = read_verified_features(feature_path, dataset_path)
    feature_manifest_hash = sha256_file(feature_path / "manifest.json")
    source_manifest_hash = sha256_file(dataset_path / "manifest.json")
    if args.role_plan.is_symlink():
        raise ValueError("role plan must not be a symlink")
    plan_path = args.role_plan.resolve(strict=True)
    plan = read_role_plan(plan_path, feature_manifest, source_manifest,
                          feature_manifest_hash, source_manifest_hash)
    train_window = plan["windows"]["train"]
    train_minutes = (train_window["end_ns"] - train_window["start_ns"]) // MINUTE_NS
    if train_minutes > MAX_TRAIN_MINUTES:
        raise ValueError("Train role exceeds the one-year hard cap")
    factor_names, means, scales, normalizer_rows = fit_factor_normalizer(
        feature_path / "features.jsonl", train_window["start_ns"], train_window["end_ns"], args.factor_set)
    device, fallback = select_device(args.device, args.max_gpu_memory_mb)
    set_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    actor = torch.nn.Linear(len(factor_names) + 1, len(ACTION_TARGETS_LOTS), bias=True, device=device)
    torch.nn.init.zeros_(actor.weight)
    torch.nn.init.zeros_(actor.bias)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    baseline = 0.0
    baseline_initialized = False
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        trace_path = stage / "training_trace.jsonl.zst"
        trace_digest = hashlib.sha256()
        episode_stats = []
        transitions = 0
        with trace_path.open("xb") as raw_trace:
            with zstandard.ZstdCompressor(level=3).stream_writer(raw_trace, closefd=False) as compressed:
                for update in range(args.updates):
                    def select_action(feature: dict[str, Any], position: int) -> tuple[int, list[float]]:
                        _, values = selected_factor_values(feature["features"], args.factor_set)
                        base = [(value - means[index]) / scales[index]
                                for index, value in enumerate(values)]
                        state = base + [position]
                        tensor = torch.tensor(state, dtype=torch.float32, device=device)
                        probabilities = torch.softmax(actor(tensor), dim=0)
                        return int(torch.multinomial(probabilities, 1, generator=rng).item()), state

                    experiences, stats = simulate_episode(
                        feature_path / "features.jsonl", feature_path / "ledger_inputs.jsonl",
                        train_window, select_action, args.fee_bps, args.slippage_bps,
                        args.max_memory_mb, collect_experiences=True)
                    if not experiences:
                        raise ValueError("Train window contains no ready policy decisions")
                    states = torch.tensor([item[0] for item in experiences], dtype=torch.float32, device=device)
                    actions = torch.tensor([item[1] for item in experiences], dtype=torch.int64, device=device)
                    rewards = torch.tensor([item[2] for item in experiences], dtype=torch.float32, device=device)
                    returns = torch.empty_like(rewards)
                    running_return = torch.zeros((), dtype=torch.float32, device=device)
                    for index in range(len(experiences) - 1, -1, -1):
                        running_return = rewards[index] + 0.99 * running_return
                        returns[index] = running_return
                    log_probs = torch.log_softmax(actor(states), dim=1).gather(1, actions[:, None]).squeeze(1)
                    advantages = returns - baseline
                    loss = -(log_probs * advantages.detach()).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
                    optimizer.step()
                    mean_return = float(returns.mean().item())
                    baseline = mean_return if not baseline_initialized else 0.9 * baseline + 0.1 * mean_return
                    baseline_initialized = True
                    stats.update({"update": update + 1, "loss": float(loss.item()),
                                  "return_mean": mean_return, "baseline": baseline})
                    episode_stats.append(stats)
                    transitions += len(experiences)
                    line = canonical(stats) + b"\n"
                    compressed.write(line)
                    trace_digest.update(line)
        code_digest = hashlib.sha256()
        for source_file in sorted(Path(__file__).parent.glob("*.py")):
            code_digest.update(source_file.name.encode() + b"\0")
            with source_file.open("rb") as source_stream:
                for chunk in iter(lambda: source_stream.read(1024 * 1024), b""):
                    code_digest.update(chunk)
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
               "source_dataset_manifest_sha256": source_manifest_hash,
               "features_manifest_sha256": feature_manifest_hash,
               "role_plan_sha256": sha256_file(plan_path),
               "source_capture_set_sha256": source_manifest.get("source_capture_set_sha256"),
               "feature_protocol_sha256": feature_manifest["protocol_sha256"],
               "trainer_code_sha256": code_digest.hexdigest(),
               "role": "train", "train_window": train_window,
               "factor_set_id": args.factor_set, "factor_set_sha256": registry_sha256(),
               "algorithm": {"name": "linear_categorical_reinforce", "gamma": 0.99,
                             "baseline": "scalar exponential moving average alpha=0.1",
                             "optimizer": "SGD", "learning_rate": 0.1,
                             "gradient_norm_clip": 10.0, "entropy_bonus": 0.0},
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
                                    "cuda": torch.version.cuda},
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
