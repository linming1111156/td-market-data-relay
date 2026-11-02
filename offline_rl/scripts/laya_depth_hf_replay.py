#!/usr/bin/env python3
"""Historical, point-in-time Laya order-book pilot using the Rust depth ledger."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL = Path("/mnt/data/td-market-data-local")
DEFAULT_DEPTH = DEFAULT_LOCAL / "datasets/usdm-depth-python-20261101T1100Z-exact"
DEFAULT_MARK = DEFAULT_LOCAL / "datasets/usdm-mark-20261101T1100Z-exact"
DEFAULT_FUNDING = DEFAULT_LOCAL / "datasets/usdm-funding-20261101T1100Z-exact"
DEFAULT_CONFIG = DEFAULT_LOCAL / "datasets/depth-account-config-20261101T1100Z.json"
DEFAULT_POLICY = DEFAULT_LOCAL / "research-inputs/laya-depth-highfreq-1s-v1.json"
DEFAULT_MODEL_CACHE = DEFAULT_LOCAL / "model-cache/laya"
DEFAULT_TEMP = DEFAULT_LOCAL / "research-inputs/.tmp"
REPLAY = DEFAULT_LOCAL / "bin/trading-futures-depth-account-replay"
DEPTH_VERIFY = DEFAULT_LOCAL / "bin/trading-usdm-depth-dataset-verify"
NS = 1_000_000_000
LOT_BTC = 0.001
MAX_ORDER_NOTIONAL_USDT = 100.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def checked_path(path: Path, label: str, *, file: bool | None = None) -> Path:
    path = path.resolve(strict=True)
    root = Path("/mnt/data").resolve(strict=True)
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must be under /mnt/data")
    if path == Path("/tmp") or Path("/tmp") in path.parents:
        raise ValueError(f"{label} must not use /tmp")
    if file is True and (not path.is_file() or path.is_symlink()):
        raise ValueError(f"{label} must be a regular file: {path}")
    if file is False and (not path.is_dir() or path.is_symlink()):
        raise ValueError(f"{label} must be a regular directory: {path}")
    return path


def top_book(bids: dict[float, float], asks: dict[float, float], n: int = 25):
    bid = sorted(((p, q) for p, q in bids.items() if q > 0), reverse=True)[:n]
    ask = sorted((p, q) for p, q in asks.items() if q > 0)[:n]
    if not bid or not ask:
        raise ValueError("order book lost one side")
    return bid, ask


def imbalance(bids: list[tuple[float, float]], asks: list[tuple[float, float]], n: int) -> float:
    b = sum(q for _, q in bids[:n])
    a = sum(q for _, q in asks[:n])
    return (b - a) / (b + a) if b + a else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--depth", type=Path, default=DEFAULT_DEPTH)
    ap.add_argument("--mark", type=Path, default=DEFAULT_MARK)
    ap.add_argument("--funding", type=Path, default=DEFAULT_FUNDING)
    ap.add_argument("--account-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    ap.add_argument("--model-cache", type=Path, default=DEFAULT_MODEL_CACHE)
    ap.add_argument("--temp-root", type=Path, default=DEFAULT_TEMP)
    ap.add_argument("--output", type=Path, default=DEFAULT_LOCAL / "reports/laya-depth-hf-1s-20261101T1100Z-pilot")
    ap.add_argument("--imbalance-threshold", type=float, default=0.25)
    ap.add_argument("--max-rss-mb", type=int, default=12_288)
    ap.add_argument("--max-gpu-memory-mb", type=int, default=12_288)
    args = ap.parse_args()

    depth = checked_path(args.depth, "depth dataset", file=False)
    mark = checked_path(args.mark, "mark dataset", file=False)
    funding = checked_path(args.funding, "funding dataset", file=False)
    account_config = checked_path(args.account_config, "account config", file=True)
    policy_path = checked_path(args.policy, "local policy", file=True)
    model_cache = checked_path(args.model_cache, "model cache", file=False)
    temp_root = args.temp_root.resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_root = checked_path(temp_root, "temporary workspace", file=False)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists (create-only): {output}")
    output = checked_path(output.parent, "report parent", file=False) / output.name
    if ROOT == output or ROOT in output.parents:
        raise ValueError("report must be outside the source repository")
    if not 0 < args.imbalance_threshold < 1:
        raise ValueError("imbalance threshold must be between 0 and 1")

    policy = json.loads(policy_path.read_text())
    if (policy.get("schema") != "td_market_data_laya_depth_policy" or policy.get("version") != 1
            or policy.get("question_id") != "target_position"
            or policy.get("actions") != {"short": -1, "flat": 0, "long": 1}
            or set(policy.get("criteria", {})) != {"short", "flat", "long"}):
        raise ValueError("unsupported local Laya policy")
    revision = policy["checkpoint_revision"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("checkpoint revision must be a pinned lowercase Git SHA")

    env = os.environ.copy()
    env.update({"HF_HOME": str(model_cache), "HF_HUB_CACHE": str(model_cache / "hub"),
                "TMPDIR": str(temp_root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    os.environ.update({k: env[k] for k in ("HF_HOME", "HF_HUB_CACHE", "TMPDIR", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})

    subprocess.run([str(DEPTH_VERIFY), "--dataset", str(depth)], check=True)
    depth_manifest = json.loads((depth / "manifest.json").read_text())
    mark_manifest = json.loads((mark / "manifest.json").read_text())
    funding_manifest = json.loads((funding / "manifest.json").read_text())
    start_ns, end_ns = depth_manifest["window"]["start_ns"], depth_manifest["window"]["end_ns"]
    for name, manifest in (("mark", mark_manifest), ("funding", funding_manifest)):
        if manifest.get("window") != depth_manifest["window"]:
            raise ValueError(f"{name} dataset does not cover the exact depth window")
        if manifest.get("schema") is None:
            raise ValueError(f"{name} manifest missing schema")
    if not depth_manifest.get("replay_usable") or depth_manifest.get("health_issue_count") != 0:
        raise ValueError("depth dataset is not clean/replay usable")
    if int(depth_manifest["event_count"]) > 100_000:
        raise ValueError("depth event count exceeds this pilot's bounded memory design")

    snapshot = json.loads((depth / "snapshot.json").read_text())
    bids = {float(x["price"]): float(x["quantity"]) for x in snapshot["bids"] if float(x["quantity"]) > 0}
    asks = {float(x["price"]): float(x["quantity"]) for x in snapshot["asks"] if float(x["quantity"]) > 0}
    previous_id = int(snapshot["last_update_id"])
    events: list[dict[str, Any]] = []
    with (depth / "events.jsonl").open() as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            if int(row["prev_final_update_id"]) != previous_id:
                raise ValueError(f"depth update sequence mismatch at event line {line_no}")
            previous_id = int(row["final_update_id"])
            events.append(row)
    if len(events) != int(depth_manifest["event_count"]):
        raise ValueError("event count does not match verified manifest")

    base_config = json.loads(account_config.read_text())
    execution = base_config["execution"]
    if (float(execution["max_abs_position_btc"]) < LOT_BTC
            or float(execution["initial_position_btc"]) != 0.0):
        raise ValueError("pilot expects flat start and at least 0.001 BTC position capacity")
    # Reserve the stated 100 ms inference budget, on top of 250 ms execution latency.
    replay_config = json.loads(json.dumps(base_config))
    replay_config["execution"]["latency_ns"] = 350_000_000
    if float(replay_config["execution"]["taker_fee_bps"]) != 5.0:
        raise ValueError("pilot assumes the frozen 5 bps taker fee config")

    import torch
    from laya import Router

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this pilot policy is pinned to CUDA")
    total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    if args.max_gpu_memory_mb <= 0 or args.max_gpu_memory_mb > total_gpu:
        raise ValueError("GPU memory cap must be positive and no greater than available device memory")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_mb / total_gpu, device=0)
    torch.set_num_threads(4)
    router = Router(device="cuda", max_loaded=1, revisions={policy["checkpoint"]: revision})

    # Build only one-second point-in-time states. The order book is advanced with
    # events available by each decision timestamp; no later event enters a query.
    cadence_ns = int(policy["decision_cadence_seconds"] * NS)
    final_time = end_ns - replay_config["execution"]["latency_ns"] - 1
    decision_times = list(range(start_ns + cadence_ns, final_time + 1, cadence_ns))
    if not decision_times or decision_times[-1] != final_time:
        decision_times.append(final_time)
    mid_history: list[tuple[int, float]] = []
    features: list[dict[str, Any]] = []
    event_index = 0
    skipped_stale_decisions = 0
    for decision_ns in decision_times:
        while event_index < len(events) and int(events[event_index]["receive_ts_ns"]) <= decision_ns:
            for level in events[event_index]["levels"]:
                side = bids if level["side"] == "BID" else asks
                price, quantity = float(level["price"]), float(level["quantity"])
                if quantity <= 0:
                    side.pop(price, None)
                else:
                    side[price] = quantity
            event_index += 1
        bid, ask = top_book(bids, asks)
        best_bid, bid_qty = bid[0]
        best_ask, ask_qty = ask[0]
        mid = (best_bid + best_ask) / 2
        microprice = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
        past_mids = [x[1] for x in mid_history]
        ts_history = [x[0] for x in mid_history]
        def return_bps(seconds: int) -> float | None:
            ix = bisect.bisect_right(ts_history, decision_ns - seconds * NS) - 1
            return ((mid / past_mids[ix]) - 1) * 10_000 if ix >= 0 else None
        age_ns = decision_ns - int(events[event_index - 1]["receive_ts_ns"]) if event_index else decision_ns - int(snapshot["received_at_ns"])
        # The simulator checks age again at fill time, after the full model +
        # execution latency. Preserve that same bound at both decision/fill.
        max_decision_age_ns = int(execution["max_book_age_ns"]) - replay_config["execution"]["latency_ns"]
        if age_ns < 0 or age_ns > max_decision_age_ns:
            skipped_stale_decisions += 1
            continue
        features.append({
            "decision_time_ns": decision_ns,
            "source_update_id": int(events[event_index - 1]["final_update_id"]) if event_index else int(snapshot["last_update_id"]),
            "book_age_ns": age_ns, "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
            "spread_bps": (best_ask - best_bid) / mid * 10_000,
            "microprice_edge_bps": (microprice / mid - 1) * 10_000,
            "imbalance_top1": imbalance(bid, ask, 1), "imbalance_top5": imbalance(bid, ask, 5),
            "imbalance_top10": imbalance(bid, ask, 10), "imbalance_top25": imbalance(bid, ask, 25),
            "mid_return_1s_bps": return_bps(1), "mid_return_5s_bps": return_bps(5),
        })
        mid_history.append((decision_ns, mid))

    if len(features) < 100:
        raise ValueError("pilot window contains too few one-second decisions")
    first = features[0]
    warm_state = {
        "market": "BTCUSDT Binance USD-M perpetual", "horizon_seconds": 1,
        "order_book": {k: first[k] for k in ("best_bid", "best_ask", "spread_bps", "microprice_edge_bps", "imbalance_top1", "imbalance_top5", "imbalance_top10", "imbalance_top25")},
        "past_market_changes": {"mid_return_1s_bps": first["mid_return_1s_bps"], "mid_return_5s_bps": first["mid_return_5s_bps"]},
        "execution_costs": {"taker_fee_bps_each_side": 5, "simulated_latency_ms": 350, "max_adverse_bps": execution["max_adverse_bps"]},
        "position_limit_btc": LOT_BTC, "available_choices": ["short", "flat", "long"],
    }
    question = {policy["question_id"]: {"type": "choice", "instructions": policy["instructions"], "criteria": policy["criteria"]}}
    router.predict(warm_state, question, model=policy["checkpoint"])

    def infer_state(feature: dict[str, Any], position: float) -> dict[str, Any]:
        return {
            "market": "BTCUSDT Binance USD-M perpetual", "horizon_seconds": 1,
            "order_book": {k: round(feature[k], 8) for k in ("best_bid", "best_ask", "spread_bps", "microprice_edge_bps", "imbalance_top1", "imbalance_top5", "imbalance_top10", "imbalance_top25")},
            "past_market_changes": {"mid_return_1s_bps": feature["mid_return_1s_bps"], "mid_return_5s_bps": feature["mid_return_5s_bps"]},
            "position": {"current_btc": position, "position_limit_btc": LOT_BTC},
            "execution_costs": {"taker_fee_bps_each_side": 5, "simulated_latency_ms": 350, "max_adverse_bps": execution["max_adverse_bps"], "max_book_age_ms": execution["max_book_age_ns"] / 1e6},
            "available_choices": ["short", "flat", "long"],
        }

    times_ms: list[float] = []
    laya_actions: list[int] = []
    imbalance_actions: list[int] = []
    laya_orders: list[dict[str, Any]] = []
    imbalance_orders: list[dict[str, Any]] = []
    virtual_positions = {"laya": 0, "imbalance": 0}
    order_counts = {"laya": 0, "imbalance": 0}
    terminal_liquidations = {"laya": 0, "imbalance": 0}
    for feature in features:
        t0 = time.perf_counter_ns()
        response = router.predict(infer_state(feature, virtual_positions["laya"] * LOT_BTC), question, model=policy["checkpoint"])
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        times_ms.append(elapsed_ms)
        current_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        if current_rss_mb > args.max_rss_mb:
            raise RuntimeError(f"RSS {current_rss_mb:.1f} MiB exceeds the configured {args.max_rss_mb} MiB cap")
        answer = response.get("answers", {}).get(policy["question_id"], {}).get("choice")
        if answer not in policy["actions"]:
            raise ValueError(f"Laya returned unsupported choice {answer!r}")
        l_action = int(policy["actions"][answer])
        # Locked, untuned depth-only comparator: top-10 quantity imbalance.
        imb = feature["imbalance_top10"]
        b_action = 1 if imb >= args.imbalance_threshold else (-1 if imb <= -args.imbalance_threshold else 0)
        laya_actions.append(l_action)
        imbalance_actions.append(b_action)
        for name, action, target_list in (("laya", l_action, laya_orders), ("imbalance", b_action, imbalance_orders)):
            current = virtual_positions[name]
            # Force both strategies to end flat on the last fresh decision so
            # neither report hides a mark-to-market open position.
            if feature is features[-1]:
                if current:
                    terminal_liquidations[name] += 1
                action = 0
            # Reverse through flat; never submit an opposite-side $156 order.
            desired = action
            if current and desired == -current:
                desired = 0
            target_btc = desired * LOT_BTC
            delta = abs(target_btc - current * LOT_BTC)
            if delta and delta * feature["mid"] > MAX_ORDER_NOTIONAL_USDT:
                raise ValueError("generated single order exceeds the frozen $100 notional limit")
            if desired != current:
                oid = f"{name}-{order_counts[name]:05d}"
                target_list.append({"order_id": oid, "decision_time_ns": int(feature["decision_time_ns"]), "target_position_btc": target_btc})
                order_counts[name] += 1
                virtual_positions[name] = desired

    elapsed_sorted = sorted(times_ms)
    p99 = elapsed_sorted[min(len(elapsed_sorted) - 1, math.ceil(len(elapsed_sorted) * .99) - 1)]
    if p99 > float(policy["inference_latency_budget_ms"]):
        raise RuntimeError(f"Laya p99 inference latency {p99:.2f}ms exceeds its 100ms per-decision budget")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    stage.mkdir(mode=0o700)
    try:
        (stage / "laya-orders.json").write_text(json.dumps(laya_orders, separators=(",", ":")))
        (stage / "imbalance-orders.json").write_text(json.dumps(imbalance_orders, separators=(",", ":")))
        (stage / "replay-config.json").write_text(json.dumps(replay_config, indent=2) + "\n")
        run_specs = (("laya", "laya-orders.json", "laya-replay.json"),
                     ("book_imbalance_top10", "imbalance-orders.json", "imbalance-replay.json"))
        summaries: dict[str, Any] = {}
        for name, orders_name, report_name in run_specs:
            cmd = [str(REPLAY), "--depth", str(depth), "--mark", str(mark), "--funding", str(funding),
                   "--config", str(stage / "replay-config.json"), "--orders", str(stage / orders_name),
                   "--output", str(stage / report_name)]
            subprocess.run(cmd, check=True)
            subprocess.run([str(REPLAY), "--depth", str(depth), "--mark", str(mark), "--funding", str(funding),
                            "--config", str(stage / "replay-config.json"), "--orders", str(stage / orders_name),
                            "--verify-report", str(stage / report_name)], check=True)
            replay_report = json.loads((stage / report_name).read_text())
            candidate = replay_report["candidate"]
            account = candidate["account"]["final_snapshot"]
            execution_report = candidate["execution"]
            summaries[name] = {
                "orders_requested": len(execution_report["orders"]),
                "order_status_counts": dict(Counter(o["status"] for o in execution_report["orders"])),
                "fills": len(execution_report["fills"]),
                "turnover_usdt": execution_report["total_turnover_usdt"],
                "fees_usdt": account.get("trading_fees_paid_usdt"),
                "final_equity_usdt": account.get("net_equity_usdt"),
                "net_pnl_usdt": account.get("net_equity_usdt", 0) - float(replay_config["initial_cash_usdt"]),
                "delta_vs_noop_usdt": replay_report["candidate_delta_vs_noop_usdt"],
                "delta_vs_buy_and_hold_usdt": replay_report["candidate_delta_vs_buy_and_hold_usdt"],
                "final_position": candidate["account"]["final_snapshot"].get("position"),
                "replay_comparison_fingerprint_sha256": replay_report["comparison_fingerprint_sha256"],
            }

        def counts(actions: list[int]) -> dict[str, int]:
            return {"short": actions.count(-1), "flat": actions.count(0), "long": actions.count(1)}
        report = {
            "schema": "td_market_data_laya_depth_hf_pilot", "version": 1,
            "claim": "historical exploratory replay only; not live-trading evidence",
            "policy_config_sha256": sha256_file(policy_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "laya": {"package_version": __import__("laya").__version__, "checkpoint": policy["checkpoint"],
                     "checkpoint_revision": revision, "device": torch.cuda.get_device_name(0),
                     "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                     "warmup_queries": 1, "measured_queries": len(features), "action_counts": counts(laya_actions),
                     "inference_latency_ms": {"median": statistics.median(times_ms), "p95": elapsed_sorted[math.ceil(.95 * len(times_ms)) - 1], "p99": p99, "max": max(times_ms)},
                     "inference_budget_ms": policy["inference_latency_budget_ms"]},
            "baseline": {"name": "top10_quantity_imbalance", "threshold": args.imbalance_threshold,
                         "action_counts": counts(imbalance_actions)},
            "terminal_liquidation_orders": terminal_liquidations,
            "data": {"depth_dataset": str(depth), "depth_manifest_sha256": sha256_file(depth / "manifest.json"),
                     "mark_manifest_sha256": sha256_file(mark / "manifest.json"),
                     "funding_manifest_sha256": sha256_file(funding / "manifest.json"),
                     "window_start_ns": start_ns, "window_end_ns": end_ns,
                     "duration_seconds": (end_ns - start_ns) / NS, "verified_depth_events": len(events),
                     "health_issues": depth_manifest["health_issue_count"], "decisions": len(features),
                     "skipped_stale_decisions_over_max_book_age": skipped_stale_decisions},
            "execution_assumptions": {"decision_cadence_seconds": 1, "model_latency_budget_ms": 100,
                                      "execution_latency_ms": 250, "total_simulated_latency_ms": 350,
                                      "taker_fee_bps_each_side": replay_config["execution"]["taker_fee_bps"],
                                      "max_abs_position_btc": replay_config["execution"]["max_abs_position_btc"],
                                      "target_position_btc": LOT_BTC, "max_single_order_notional_usdt": MAX_ORDER_NOTIONAL_USDT,
                                      "reversal_routes_through_flat": True},
            "results": summaries,
            "resources": {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
                          "gpu_memory_reserved_mb": torch.cuda.max_memory_reserved(0) / (1024 * 1024)},
            "limitations": [
                "Only one 12m36s BTCUSDT depth window is evaluated; this cannot establish strategy reliability or generalization.",
                "Laya is a general typed-decision model, not a market microstructure model; the policy prompt is a zero-shot research hypothesis.",
                "The book-imbalance baseline is intentionally simple and its threshold is frozen before observing the result.",
                "The Rust replay uses recorded depth, a 350ms total decision-to-execution latency, 5 bps taker fees, and the configured adverse-price guard.",
                "The same-window NOOP and buy-and-hold comparisons are controls, not independent out-of-sample evidence.",
                "No exchange account, API key, or live order path is used.",
            ],
        }
        (stage / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        hashes = {p.name: sha256_file(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage / "manifest.json").write_text(json.dumps({"schema": "td_market_data_laya_depth_hf_pilot_artifact", "version": 1, "files": hashes}, sort_keys=True, indent=2) + "\n")
        for path in stage.iterdir():
            os.chmod(path, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
    except BaseException:
        if stage.exists():
            os.chmod(stage, 0o755)
            for path in stage.iterdir():
                os.chmod(path, 0o644)
            import shutil
            shutil.rmtree(stage)
        raise
    print(f"Laya depth pilot report: {output / 'report.json'}")
    print(json.dumps(report["results"], indent=2, sort_keys=True))
    print("Laya latency ms:", json.dumps(report["laya"]["inference_latency_ms"], sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Laya depth pilot failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
