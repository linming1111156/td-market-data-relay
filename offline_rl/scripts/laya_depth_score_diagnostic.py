#!/usr/bin/env python3
"""Score-distribution diagnostic for the frozen zero-shot Laya depth policy."""
from __future__ import annotations

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

from laya_depth_hf_replay import (
    DEFAULT_LOCAL,
    DEFAULT_MODEL_CACHE,
    DEFAULT_POLICY,
    DEFAULT_TEMP,
    DEPTH_VERIFY,
    NS,
    checked_path,
    imbalance,
    sha256_file,
    top_book,
)

ROOT = Path(__file__).resolve().parents[2]
WINDOWS = (
    ("20261101T0919Z", DEFAULT_LOCAL / "datasets/usdm-depth-20261101T0919Z-final"),
    ("20261101T1019Z", DEFAULT_LOCAL / "datasets/usdm-depth-20261101T1019Z-final"),
    ("20261101T1100Z", DEFAULT_LOCAL / "datasets/usdm-depth-python-20261101T1100Z-exact"),
)
MAX_DECISION_BOOK_AGE_NS = 650_000_000  # 350 ms total decision-to-fill budget; 1 s book-age cap.
HORIZONS_SECONDS = (1, 5)


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1)]


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        average_rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            result[order[k]] = average_rank
        i = j
    return result


def corr(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = statistics.mean(x), statistics.mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else None


def load_window(window_id: str, path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = checked_path(path, f"{window_id} depth dataset", file=False)
    subprocess.run([str(DEPTH_VERIFY), "--dataset", str(path)], check=True, capture_output=True, text=True)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (not manifest.get("replay_usable") or manifest.get("health_issue_count") != 0
            or manifest.get("receive_time_unknown_event_count") != 0):
        raise ValueError(f"{window_id} is not a clean replay-usable depth window")
    start_ns, end_ns = manifest["window"]["start_ns"], manifest["window"]["end_ns"]
    snapshot = json.loads((path / "snapshot.json").read_text())
    bids = {float(x["price"]): float(x["quantity"]) for x in snapshot["bids"] if float(x["quantity"]) > 0}
    asks = {float(x["price"]): float(x["quantity"]) for x in snapshot["asks"] if float(x["quantity"]) > 0}
    previous_id = int(snapshot["last_update_id"])
    events: list[dict[str, Any]] = []
    with (path / "events.jsonl").open() as stream:
        for line_no, line in enumerate(stream, 1):
            event = json.loads(line)
            # The verified datasets start with a Binance bootstrap event whose
            # update range overlaps the snapshot. Rust's dataset verifier
            # validates that overlap; subsequent events must chain exactly.
            if line_no > 1 and int(event["prev_final_update_id"]) != previous_id:
                raise ValueError(f"{window_id}: update chain mismatch at line {line_no}")
            previous_id = int(event["final_update_id"])
            events.append(event)
    if len(events) != manifest["event_count"]:
        raise ValueError(f"{window_id}: event count mismatch")

    cadence = NS
    times = list(range(start_ns + cadence, end_ns, cadence))
    grid: list[dict[str, Any]] = []
    event_index = 0
    for t in times:
        while event_index < len(events) and int(events[event_index]["receive_ts_ns"]) <= t:
            for level in events[event_index]["levels"]:
                side = bids if level["side"] == "BID" else asks
                px, qty = float(level["price"]), float(level["quantity"])
                if qty <= 0:
                    side.pop(px, None)
                else:
                    side[px] = qty
            event_index += 1
        bid, ask = top_book(bids, asks)
        best_bid, bid_qty = bid[0]
        best_ask, ask_qty = ask[0]
        mid = (best_bid + best_ask) / 2
        micro = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
        available_ns = int(events[event_index - 1]["receive_ts_ns"]) if event_index else int(snapshot["received_at_ns"])
        age_ns = t - available_ns
        row: dict[str, Any] = {
            "timestamp_ns": t, "mid": mid, "age_ns": age_ns,
            "fresh": 0 <= age_ns <= MAX_DECISION_BOOK_AGE_NS,
            "features": {
                "best_bid": best_bid, "best_ask": best_ask,
                "spread_bps": (best_ask - best_bid) / mid * 10_000,
                "microprice_edge_bps": (micro / mid - 1) * 10_000,
                "imbalance_top1": imbalance(bid, ask, 1),
                "imbalance_top5": imbalance(bid, ask, 5),
                "imbalance_top10": imbalance(bid, ask, 10),
                "imbalance_top25": imbalance(bid, ask, 25),
            },
        }
        grid.append(row)
    grid_times = [r["timestamp_ns"] for r in grid]
    score_rows: list[dict[str, Any]] = []
    for i, row in enumerate(grid):
        if not row["fresh"]:
            continue
        f = row["features"]
        prior = {}
        for sec in (1, 5):
            j = bisect.bisect_right(grid_times, row["timestamp_ns"] - sec * NS) - 1
            prior[f"mid_return_{sec}s_bps"] = ((row["mid"] / grid[j]["mid"]) - 1) * 10_000 if j >= 0 else None
        state = {
            "market": "BTCUSDT Binance USD-M perpetual", "horizon_seconds": 1,
            "order_book": {k: round(v, 8) for k, v in f.items()},
            "past_market_changes": prior,
            "position": {"current_btc": 0.0, "position_limit_btc": 0.001},
            "execution_costs": {"taker_fee_bps_each_side": 5, "simulated_latency_ms": 350,
                                "max_adverse_bps": 20, "max_book_age_ms": 1000},
            "available_choices": ["short", "flat", "long"],
        }
        future: dict[str, float | None] = {}
        for sec in HORIZONS_SECONDS:
            j = bisect.bisect_left(grid_times, row["timestamp_ns"] + sec * NS)
            if j < len(grid) and grid[j]["fresh"] and grid_times[j] - row["timestamp_ns"] - sec * NS <= 100_000_000:
                future[str(sec)] = (grid[j]["mid"] / row["mid"] - 1) * 10_000
            else:
                future[str(sec)] = None
        score_rows.append({"timestamp_ns": row["timestamp_ns"], "state": state,
                           "features": f, "future_mid_return_bps": future})
    metadata = {
        "window_id": window_id, "dataset": str(path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_fingerprint_sha256": manifest["source_fingerprint_sha256"],
        "window_start_ns": start_ns, "window_end_ns": end_ns,
        "duration_seconds": (end_ns - start_ns) / NS,
        "event_count": len(events), "health_issue_count": manifest["health_issue_count"],
        "grid_decisions": len(grid), "fresh_decisions": len(score_rows),
        "skipped_stale_decisions": len(grid) - len(score_rows),
    }
    return score_rows, metadata


def summarize(window_id: str, rows: list[dict[str, Any]], metadata: dict[str, Any], latency: list[float]) -> dict[str, Any]:
    probs = [r["probabilities"] for r in rows]
    actions = Counter(r["choice"] for r in rows)
    names = ("short", "flat", "long")
    per_option = {name: {
        "mean": statistics.mean(p[name] for p in probs),
        "stddev": statistics.pstdev(p[name] for p in probs),
        "min": min(p[name] for p in probs), "max": max(p[name] for p in probs),
    } for name in names}
    directional = [p["long"] - p["short"] for p in probs]
    future_metrics: dict[str, Any] = {}
    for sec in HORIZONS_SECONDS:
        pairs = [(directional[i], r["future_mid_return_bps"].get(str(sec)))
                 for i, r in enumerate(rows) if r["future_mid_return_bps"].get(str(sec)) is not None]
        x, y = [a for a, _ in pairs], [float(b) for _, b in pairs]
        rank_corr = corr(rank(x), rank(y)) if len(x) >= 3 else None
        aligned = [(1 if a > 0 else -1 if a < 0 else 0, b) for a, b in pairs if a != 0 and b != 0]
        nonzero = [(rows[i]["choice"], float(r["future_mid_return_bps"][str(sec)]))
                   for i, r in enumerate(rows)
                   if r["future_mid_return_bps"].get(str(sec)) not in (None, 0)]
        argmax_directional = [(choice, value) for choice, value in nonzero if choice in ("short", "long")]
        always_short_hit = sum(value < 0 for _, value in nonzero) / len(nonzero) if nonzero else None
        argmax_hit = (sum((choice == "long" and value > 0) or (choice == "short" and value < 0)
                          for choice, value in argmax_directional) / len(argmax_directional)
                      if argmax_directional else None)
        future_metrics[str(sec)] = {
            "n": len(pairs), "pearson_direction_score_vs_future_mid_return": corr(x, y),
            "spearman_direction_score_vs_future_mid_return": rank_corr,
            "directional_hit_rate_when_model_score_nonzero": (sum(1 for a, b in aligned if a * b > 0) / len(aligned)) if aligned else None,
            "nonzero_return_n": len(nonzero),
            "always_short_hit_rate_on_nonzero_returns": always_short_hit,
            "laya_argmax_hit_rate_on_nonzero_returns": argmax_hit,
            "laya_minus_always_short_hit_rate": (argmax_hit - always_short_hit) if argmax_hit is not None and always_short_hit is not None else None,
            "mean_future_mid_return_bps": statistics.mean(y) if y else None,
            "mean_future_return_top_decile_score_bps": None,
            "mean_future_return_bottom_decile_score_bps": None,
        }
        if len(pairs) >= 10:
            order = sorted(range(len(pairs)), key=lambda ix: x[ix])
            count = max(1, len(order) // 10)
            future_metrics[str(sec)]["mean_future_return_bottom_decile_score_bps"] = statistics.mean(y[ix] for ix in order[:count])
            future_metrics[str(sec)]["mean_future_return_top_decile_score_bps"] = statistics.mean(y[ix] for ix in order[-count:])
    vectors = {(p["short"], p["flat"], p["long"]) for p in probs}
    margins = [max(p.values()) - sorted(p.values())[-2] for p in probs]
    return {
        **metadata,
        "choice_counts": {name: actions[name] for name in names},
        "argmax_choice_share": {name: actions[name] / len(rows) for name in names},
        "unique_probability_vectors_rounded_by_laya": len(vectors),
        "option_score_summary": per_option,
        "direction_score_long_minus_short": {"mean": statistics.mean(directional), "stddev": statistics.pstdev(directional),
                                               "min": min(directional), "max": max(directional)},
        "argmax_margin": {"mean": statistics.mean(margins), "median": statistics.median(margins)},
        "future_return_diagnostics": future_metrics,
        "inference_latency_ms": {"median": statistics.median(latency), "p95": percentile(latency, .95),
                                 "p99": percentile(latency, .99), "max": max(latency)},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_LOCAL / "reports/laya-depth-score-diagnostic-20261101T0919-1100-v1")
    ap.add_argument("--max-rss-mb", type=int, default=12_288)
    ap.add_argument("--max-gpu-memory-mb", type=int, default=12_288)
    args = ap.parse_args()
    policy_path = checked_path(DEFAULT_POLICY, "local Laya policy", file=True)
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw)
    if (policy.get("schema") != "td_market_data_laya_depth_policy" or policy.get("version") != 1
            or policy.get("question_id") != "target_position"
            or policy.get("actions") != {"short": -1, "flat": 0, "long": 1}
            or set(policy.get("criteria", {})) != {"short", "flat", "long"}):
        raise ValueError("unsupported local policy")
    model_cache = checked_path(DEFAULT_MODEL_CACHE, "Laya cache", file=False)
    temp_root = DEFAULT_TEMP.resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_root = checked_path(temp_root, "temporary workspace", file=False)
    os.environ.update({"HF_HOME": str(model_cache), "HF_HUB_CACHE": str(model_cache / "hub"),
                       "TMPDIR": str(temp_root), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})

    loaded: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    for window_id, path in WINDOWS:
        rows, metadata = load_window(window_id, path)
        loaded.append((window_id, rows, metadata))
    import torch
    from laya import Router
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen diagnostic")
    total_gpu = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    if not 0 < args.max_gpu_memory_mb <= total_gpu:
        raise ValueError("GPU memory limit exceeds device memory")
    torch.cuda.set_per_process_memory_fraction(args.max_gpu_memory_mb / total_gpu, device=0)
    torch.set_num_threads(4)
    router = Router(device="cuda", max_loaded=1, revisions={policy["checkpoint"]: policy["checkpoint_revision"]})
    questions = {policy["question_id"]: {"type": "choice", "instructions": policy["instructions"], "criteria": policy["criteria"]}}
    # Warm once on the first chronological state; do not include warm-up in the sample.
    router.predict(loaded[0][1][0]["state"], questions, model=policy["checkpoint"])

    all_scores: list[dict[str, Any]] = []
    window_reports: list[dict[str, Any]] = []
    total_latencies: list[float] = []
    for window_id, inputs, metadata in loaded:
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        for item in inputs:
            started = time.perf_counter_ns()
            result = router.predict(item["state"], questions, model=policy["checkpoint"])
            latency_ms = (time.perf_counter_ns() - started) / 1e6
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 > args.max_rss_mb:
                raise RuntimeError("RSS exceeded configured cap")
            answer = result.get("answers", {}).get(policy["question_id"], {})
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != {"short", "flat", "long"}:
                raise ValueError(f"Laya did not return all three option scores: {answer!r}")
            probabilities = {k: float(probabilities[k]) for k in ("short", "flat", "long")}
            if any(not math.isfinite(v) or v < 0 for v in probabilities.values()) or abs(sum(probabilities.values()) - 1) > .002:
                raise ValueError(f"invalid Laya score distribution: {probabilities}")
            choice = answer.get("choice")
            if choice not in probabilities:
                raise ValueError(f"invalid Laya choice: {choice!r}")
            row = {"window_id": window_id, "timestamp_ns": item["timestamp_ns"],
                   "probabilities": probabilities, "choice": choice,
                   "direction_score_long_minus_short": probabilities["long"] - probabilities["short"],
                   "future_mid_return_bps": item["future_mid_return_bps"],
                   "features": item["features"], "inference_latency_ms": latency_ms}
            rows.append(row)
            latencies.append(latency_ms)
            total_latencies.append(latency_ms)
            all_scores.append(row)
        window_reports.append(summarize(window_id, rows, metadata, latencies))

    if percentile(total_latencies, .99) > 100:
        raise RuntimeError(f"inference p99 {percentile(total_latencies, .99):.2f} ms exceeds the frozen 100 ms budget")
    all_report = summarize("all_windows", all_scores, {
        "window_count": len(loaded), "fresh_decisions": len(all_scores),
        "window_ids": [x[0] for x in loaded],
    }, total_latencies)
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output exists (create-only): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.staging-{os.getpid()}"
    stage.mkdir(mode=0o700)
    try:
        scores_path = stage / "scores.jsonl"
        with scores_path.open("w") as stream:
            for row in all_scores:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        report = {
            "schema": "td_market_data_laya_depth_score_diagnostic", "version": 1,
            "policy_config_sha256": hashlib.sha256(policy_raw).hexdigest(),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": policy["checkpoint"], "checkpoint_revision": policy["checkpoint_revision"],
            "laya_version": __import__("laya").__version__, "torch_version": torch.__version__,
            "device": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "warmup_queries": 1, "measured_queries": len(all_scores),
            "windows": window_reports, "combined": all_report,
            "resources": {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          "gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(0) / (1024 * 1024),
                          "gpu_memory_reserved_mb": torch.cuda.max_memory_reserved(0) / (1024 * 1024)},
            "interpretation_limits": [
                "Choice probabilities are model scores; do not treat them as calibrated trading probabilities.",
                "Forward mid returns are gross diagnostic labels, not executable PnL; they exclude fees and book-walk execution.",
                "Three short windows on one date are a diagnostic sample, not independent out-of-sample validation.",
                "The policy prompt, checkpoint and input schema are frozen; no prompt or threshold tuning is performed.",
            ],
        }
        (stage / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
        files = {p.name: sha256_file(p) for p in sorted(stage.iterdir()) if p.is_file()}
        (stage / "manifest.json").write_text(json.dumps({"schema": "td_market_data_laya_depth_score_diagnostic_artifact", "version": 1, "files": files}, sort_keys=True, indent=2) + "\n")
        for p in stage.iterdir():
            os.chmod(p, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output)
    except BaseException:
        if stage.exists():
            os.chmod(stage, 0o755)
            for p in stage.iterdir(): os.chmod(p, 0o644)
            import shutil
            shutil.rmtree(stage)
        raise
    print(f"Score diagnostic: {output / 'report.json'}")
    print(json.dumps({"combined": all_report, "windows": window_reports}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Laya depth score diagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
