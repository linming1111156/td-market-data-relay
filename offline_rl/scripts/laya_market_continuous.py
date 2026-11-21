#!/usr/bin/env python3
"""Seed, evaluate, serve, and export a persistent Kaggle CPU Laya 0.5 market learner."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
QUESTION_ID = "target_position"
CLASSES = ("short", "flat", "long")
FEATURES = (
    "mark_return_1m_bps",
    "mark_return_5m_bps",
    "mark_return_15m_bps",
    "mark_realized_volatility_15m_bps",
    "mark_realized_volatility_60m_bps",
    "aggtrade_signed_volume_imbalance_1m",
    "aggtrade_signed_volume_imbalance_5m",
)
QUESTIONS = {
    QUESTION_ID: {
        "type": "choice",
        "instructions": (
            "Choose a BTCUSDT USD-M perpetual target position for the next 15 minutes. "
            "Use only the supplied point-in-time market features and fixed risk limits."
        ),
        "criteria": {
            "short": "Take a one-lot short position.",
            "flat": "Hold no position when the expected move does not clear round-trip costs.",
            "long": "Take a one-lot long position.",
        },
    }
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def state_text(state: dict[str, Any]) -> str:
    """Stable, point-in-time text passed to Vega; labels/future outcomes stay out of state."""
    market = state.get("market_features")
    if not isinstance(market, dict) or not market:
        raise ValueError("market state has no market_features object")
    fields = [f"Market: {state.get('market', 'BTCUSDT USD-M perpetual')}"]
    if "decision_interval_minutes" in state:
        fields.append(f"Decision interval minutes: {state['decision_interval_minutes']}")
    fields.extend(f"{name}: {value}" for name, value in sorted(market.items()))
    portfolio = state.get("portfolio")
    if isinstance(portfolio, dict):
        fields.extend(f"portfolio {name}: {value}" for name, value in sorted(portfolio.items()))
    risk = state.get("risk_constraints")
    if isinstance(risk, dict):
        fields.extend(f"risk {name}: {value}" for name, value in sorted(risk.items()))
    if "mark_price_usdt" in state:
        fields.append(f"mark price USDT: {state['mark_price_usdt']}")
    return "; ".join(fields)


def answer_for(row: dict[str, Any], *, confirmed: bool) -> str:
    key = "answers" if confirmed else "expected"
    answers = row.get(key)
    if not isinstance(answers, dict) or answers.get(QUESTION_ID) not in CLASSES:
        raise ValueError(f"row needs {key}.{QUESTION_ID} in {CLASSES}")
    return answers[QUESTION_ID]


def question_config(rows: list[dict[str, Any]]) -> dict[str, Any]:
    config: dict[str, Any] | None = None
    for line_no, row in enumerate(rows, 1):
        questions = row.get("questions")
        question = questions.get(QUESTION_ID) if isinstance(questions, dict) else None
        if not isinstance(question, dict) or question.get("type") != "choice":
            raise ValueError(f"row {line_no} must define questions.{QUESTION_ID} as a choice")
        criteria = question.get("criteria")
        if not isinstance(criteria, dict) or set(criteria) != set(CLASSES):
            raise ValueError(f"row {line_no} choice criteria must exactly match {CLASSES}")
        if not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"row {line_no} question instructions are missing")
        current = {QUESTION_ID: question}
        if config is None:
            config = current
        elif current != config:
            raise ValueError(f"row {line_no} question definition differs from earlier rows")
    if config is None:
        raise ValueError("training and evaluation rows are empty")
    return config


def read_seed_examples(path: Path) -> list[tuple[str, dict[str, str]]]:
    """Load the CSV format used by vegaml's Learner constructor."""
    examples: list[tuple[str, dict[str, str]]] = []
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        required = {"text", QUESTION_ID}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must have CSV columns: text, {QUESTION_ID}")
        for line_no, row in enumerate(reader, 2):
            text = (row.get("text") or "").strip()
            label = (row.get(QUESTION_ID) or "").strip()
            if not text or label not in CLASSES:
                raise ValueError(f"{path}:{line_no}: invalid text or {QUESTION_ID} label")
            examples.append((text, {QUESTION_ID: label}))
    if not examples:
        raise ValueError(f"seed examples are empty: {path}")
    return examples


def outside_repo(path: Path, label: str) -> Path:
    path = path.expanduser().absolute()
    if path == ROOT or ROOT in path.parents:
        raise ValueError(f"{label} must be outside the repository to keep private state out of Git")
    return path


def cmd_prepare(args: argparse.Namespace) -> int:
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya data preparation must run on Kaggle; the local machine is code-only")
    train_path = args.train_jsonl.expanduser().resolve(strict=True)
    eval_path = args.evaluation_jsonl.expanduser().resolve(strict=True)
    output = outside_repo(args.output_dir, "output directory")
    if output.exists() or output.is_symlink():
        raise ValueError(f"output directory must be new: {output}")
    train_rows, eval_rows = read_jsonl(train_path), read_jsonl(eval_path)
    if not train_rows or not eval_rows:
        raise ValueError("training and evaluation inputs must both contain rows")
    questions = question_config(train_rows + eval_rows)
    train_labels = Counter(answer_for(row, confirmed=False) for row in train_rows)
    eval_labels = Counter(answer_for(row, confirmed=False) for row in eval_rows)
    output.mkdir(parents=True)
    (output / "questions.json").write_text(
        json.dumps(questions, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "seed_examples.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("text", QUESTION_ID))
        writer.writeheader()
        for row in train_rows:
            writer.writerow({"text": state_text(row["state"]), QUESTION_ID: answer_for(row, confirmed=False)})
    (output / "baseline_train.jsonl").write_text(
        "".join(json.dumps({
            "decision_time_unix_ms": row.get("decision_time_unix_ms"),
            "decision_interval_minutes": row["state"].get("decision_interval_minutes", 15),
            "market_features": row["state"].get("market_features"),
            "outcome": row.get("outcome"),
            "label": answer_for(row, confirmed=False),
        }, ensure_ascii=False) + "\n" for row in train_rows),
        encoding="utf-8",
    )
    (output / "evaluation.jsonl").write_text(
        "".join(json.dumps({"decision_time_unix_ms": row.get("decision_time_unix_ms"),
                            "decision_interval_minutes": row["state"].get("decision_interval_minutes", 15),
                            "market_features": row["state"].get("market_features"),
                            "outcome": row.get("outcome"),
                            "state_text": state_text(row["state"]),
                            "label": answer_for(row, confirmed=False)}, ensure_ascii=False) + "\n"
                 for row in eval_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema": "td_market_laya_continuous_seed",
        "version": 1,
        "laya_version": "0.5.0",
        "learner": "Laya Learner / Vega frozen features",
        "train_source_sha256": sha256_file(train_path),
        "evaluation_source_sha256": sha256_file(eval_path),
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "train_labels": dict(sorted(train_labels.items())),
        "evaluation_labels": dict(sorted(eval_labels.items())),
        "questions_sha256": sha256_file(output / "questions.json"),
        "seed_examples_sha256": sha256_file(output / "seed_examples.csv"),
        "baseline_train_sha256": sha256_file(output / "baseline_train.jsonl"),
        "evaluation_sha256": sha256_file(output / "evaluation.jsonl"),
    }
    (output / "seed-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"prepared": str(output), "train_rows": len(train_rows),
                      "evaluation_rows": len(eval_rows), "train_labels": dict(train_labels),
                      "evaluation_labels": dict(eval_labels)}, ensure_ascii=False, indent=2))
    return 0


def metrics(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    matrix = {truth: {guess: 0 for guess in CLASSES} for truth in CLASSES}
    for truth, guess in zip(actual, predicted, strict=True):
        matrix[truth][guess] += 1
    recalls, f1s = [], []
    for label in CLASSES:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_n = sum(matrix[row][label] for row in CLASSES)
        recall = tp / support if support else 0.0
        precision = tp / predicted_n if predicted_n else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "items": len(actual),
        "accuracy": sum(a == p for a, p in zip(actual, predicted, strict=True)) / len(actual),
        "balanced_accuracy": sum(recalls) / len(CLASSES),
        "macro_f1": sum(f1s) / len(CLASSES),
        "confusion_actual_by_predicted": matrix,
    }


def clean_probabilities(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(CLASSES):
        return None
    try:
        probabilities = {label: float(value[label]) for label in CLASSES}
    except (TypeError, ValueError, OverflowError):
        return None
    if any(not 0.0 <= p <= 1.0 for p in probabilities.values()):
        return None
    if abs(sum(probabilities.values()) - 1.0) > 1e-3:
        return None
    return probabilities


def calibration_metrics(actual: list[str], distributions: list[Any], bins: int = 10) -> dict[str, Any]:
    """Multiclass Brier score and confidence ECE; omit when the engine exposes no class probabilities."""
    if len(actual) != len(distributions) or not actual:
        return {"available": False, "reason": "no complete probability predictions"}
    probabilities = [clean_probabilities(value) for value in distributions]
    if any(value is None for value in probabilities):
        return {"available": False, "reason": "Laya did not return valid probabilities for every row"}
    valid = [value for value in probabilities if value is not None]
    brier = 0.0
    confidences, correct = [], []
    for truth, distribution in zip(actual, valid, strict=True):
        predicted = max(CLASSES, key=distribution.__getitem__)
        brier += sum((distribution[label] - float(label == truth)) ** 2 for label in CLASSES)
        confidences.append(distribution[predicted])
        correct.append(float(predicted == truth))
    ece = 0.0
    bin_counts = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, confidence in enumerate(confidences)
                   if low <= confidence < high or (index == bins - 1 and confidence == 1.0)]
        bin_counts.append(len(members))
        if members:
            mean_confidence = sum(confidences[i] for i in members) / len(members)
            accuracy = sum(correct[i] for i in members) / len(members)
            ece += len(members) / len(actual) * abs(mean_confidence - accuracy)
    return {
        "available": True,
        "method": "multiclass_brier_and_10_bin_top_label_ece",
        "items": len(actual),
        "brier_score": brier / len(actual),
        "expected_calibration_error": ece,
        "ece_bins": bin_counts,
    }


def strategy_metrics(rows: list[dict[str, Any]], actions: list[str]) -> dict[str, Any]:
    """One independent 15-minute one-lot trade per row; returns stay outside model input."""
    actual = [row["label"] for row in rows]
    result: dict[str, Any] = {"classification": metrics(actual, actions)}
    if not all(isinstance(row.get("outcome"), dict) for row in rows):
        result["backtest"] = {"available": False, "reason": "rows lack forward outcome metadata"}
        return result
    pnl, trades, gross, costs = [], 0, 0.0, 0.0
    for row, action in zip(rows, actions, strict=True):
        outcome = row["outcome"]
        forward = outcome.get("forward_mark_return_bps")
        cost = outcome.get("round_trip_cost_bps")
        if not isinstance(forward, (int, float)) or not isinstance(cost, (int, float)):
            result["backtest"] = {"available": False, "reason": "invalid forward return or cost"}
            return result
        position = 1 if action == "long" else -1 if action == "short" else 0
        trade_gross = position * float(forward)
        trade_cost = float(cost) if position else 0.0
        gross += trade_gross
        costs += trade_cost
        pnl.append(trade_gross - trade_cost)
        trades += bool(position)
    equity = drawdown = peak = 0.0
    for value in pnl:
        equity += value
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    result["backtest"] = {
        "available": True,
        "decisions": len(rows),
        "trades": trades,
        "turnover_sides": 2 * trades,
        "gross_pnl_bps": round(gross, 4),
        "cost_bps": round(costs, 4),
        "net_pnl_bps": round(sum(pnl), 4),
        "net_pnl_bps_per_decision": round(sum(pnl) / len(pnl), 4) if pnl else 0.0,
        "max_drawdown_bps": round(drawdown, 4),
    }
    return result


def prime_features(learner, texts: list[str]) -> None:
    """Compute frozen Vega features once and reuse them for scoring and feedback."""
    from vegaml.learn.learner import _key

    missing = [text for text in texts if QUESTION_ID not in learner._cache.get(_key(text), {})]
    if not missing:
        return
    questions = {QUESTION_ID: learner.questions[QUESTION_ID]}
    feature_rows = learner.reader.features(missing, questions)
    for text, feature_row in zip(missing, feature_rows, strict=True):
        learner._cache[_key(text)] = {QUESTION_ID: feature_row[0]}


def build_seed_store_batched(reader, questions, rows, batch_size: int = 32):
    """Build a Laya 0.5 seed store with visible CPU progress between bounded batches."""
    from vegaml.learn.learner import Store, bundle_bytes, fit_head
    from vegaml.learn.reader import label_of

    qids = list(questions)
    clean, skipped = [], {qid: 0 for qid in qids}
    for text, labels in rows:
        normalized = {}
        for qid, value in labels.items():
            if qid not in questions:
                continue
            try:
                normalized[qid] = label_of(questions[qid], value)
            except (ValueError, TypeError):
                skipped[qid] += 1
        if normalized:
            clean.append((text, normalized))

    store = Store(qids)
    total = len(clean)
    for start in range(0, total, batch_size):
        batch = clean[start : start + batch_size]
        texts = [text for text, _ in batch]
        features = reader.features(texts, questions)
        batch_engine = [reader.engine_labels(text, questions) for text in texts]
        for q_index, qid in enumerate(qids):
            indices = [i for i, (_, labels) in enumerate(batch) if qid in labels]
            if indices:
                store.add(
                    qid,
                    features[indices, q_index],
                    [batch[i][1][qid] for i in indices],
                    [batch_engine[i][qid] for i in indices],
                    "start",
                )
        print(f"seed initialization features and Vega labels: {min(start + len(batch), total)}/{total}",
              flush=True)

    report = {}
    for qid in qids:
        indices = store.rows(qid, phase_out=False)
        head = (fit_head(questions[qid], store.X[qid][indices],
                         [store.y[qid][i] for i in indices],
                         [store.e[qid][i] for i in indices]) if indices else None)
        report[qid] = {
            "rows": len(indices),
            "skipped_rows": skipped[qid],
            "starts_with": ("a head fitted on the starting set"
                            if head and head["usable"] else "vega's own answers"),
            "cross_validated_accuracy": (None if not head or head["cv"] is None
                                         else round(head["cv"], 3)),
            "vega_alone_accuracy": None if not head else round(head["engine"], 3),
        }
    learn_bytes, start_bytes = bundle_bytes(reader, questions, store)
    return learn_bytes, start_bytes, report


GPU_PLAN_LIMIT_HOURS = 27.0
GPU_HARD_LIMIT_HOURS = 30.0


def select_runtime_device(requested: str, gpu_hours_used_7d: float | None,
                          gpu_planned_hours: float | None) -> tuple[str, dict[str, Any]]:
    """Prefer CUDA only with a fresh caller-supplied quota observation and a bounded plan."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported device request: {requested}")
    quota_verified = (
        gpu_hours_used_7d is not None and gpu_planned_hours is not None
        and math.isfinite(gpu_hours_used_7d) and math.isfinite(gpu_planned_hours)
        and 0 <= gpu_hours_used_7d <= GPU_HARD_LIMIT_HOURS and gpu_planned_hours > 0
    )
    planned_total = gpu_hours_used_7d + gpu_planned_hours if quota_verified else None
    within_plan = bool(quota_verified and planned_total <= GPU_PLAN_LIMIT_HOURS)
    budget: dict[str, Any] = {
        "observed_hours_used_rolling_7d": gpu_hours_used_7d if quota_verified else None,
        "planned_hours": gpu_planned_hours if quota_verified else None,
        "planned_total_hours": planned_total,
        "quota_verified": quota_verified,
        "within_plan": within_plan,
        "plan_limit_hours": GPU_PLAN_LIMIT_HOURS,
        "hard_limit_hours": GPU_HARD_LIMIT_HOURS,
        "requested_device": requested,
    }
    if requested == "cpu":
        budget.update(selected_device="cpu", fallback_reason="CPU explicitly requested")
        return "cpu", budget
    if requested == "cuda" and not quota_verified:
        raise ValueError("CUDA requires a fresh quota reading and --gpu-planned-hours")
    if requested == "cuda" and not within_plan:
        raise ValueError("GPU plan must keep rolling 7-day use at or below 27 hours (hard cap 30 hours)")
    if not quota_verified:
        budget.update(selected_device="cpu", fallback_reason="fresh Kaggle GPU quota was not provided")
        return "cpu", budget
    if not within_plan:
        budget.update(selected_device="cpu", fallback_reason="GPU plan limit reached; falling back to Kaggle CPU")
        return "cpu", budget
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False
    if not cuda_available:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        budget.update(selected_device="cpu", fallback_reason="CUDA is unavailable; falling back to Kaggle CPU")
        return "cpu", budget
    budget.update(selected_device="cuda", fallback_reason=None)
    return "cuda", budget


def add_device_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="prefer quota-approved CUDA for Laya model work; otherwise use Kaggle CPU")
    parser.add_argument("--gpu-hours-used-7d", type=float,
                        help="fresh Kaggle CLI quota reading from immediately before the run")
    parser.add_argument("--gpu-planned-hours", type=float,
                        help="conservative total GPU time budget for this Kaggle run")


def traditional_baseline_diagnostics(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run CPU-only baselines and report their metrics by real evaluation session."""
    if not all(isinstance(row.get("market_features"), dict) for row in train_rows + eval_rows):
        unavailable = {"available": False, "reason": "rows lack numeric market_features"}
        return unavailable, unavailable
    if not all(isinstance(row.get("market_features", {}).get(k), (int, float)) for row in train_rows + eval_rows for k in FEATURES):
        unavailable = {"available": False, "reason": "market feature schema is incomplete or nonnumeric"}
        return unavailable, unavailable

    def rule(sign: float) -> list[str]:
        actions = []
        for row in eval_rows:
            move = sign * float(row["market_features"]["mark_return_15m_bps"])
            actions.append("long" if move > 30.0 else "short" if move < -30.0 else "flat")
        return actions

    predictions = {
        "15m_trend_threshold_30bps": rule(1.0),
        "15m_reversal_threshold_30bps": rule(-1.0),
    }
    metadata: dict[str, Any] = {"available": True, "features": list(FEATURES), "class_weighted": True}
    try:
        import numpy as np
    except ImportError:
        metadata["class_weighted"] = False
        metadata["logistic_unavailable"] = "numpy unavailable"
        return _format_traditional_baseline_diagnostics(eval_rows, predictions, metadata)
    x = np.asarray([[float(row["market_features"][k]) for k in FEATURES] for row in train_rows], dtype=np.float64)
    xv = np.asarray([[float(row["market_features"][k]) for k in FEATURES] for row in eval_rows], dtype=np.float64)
    y = np.asarray([CLASSES.index(row["label"]) for row in train_rows], dtype=np.int64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x, xv = (x - mean) / scale, (xv - mean) / scale
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float64)
    if not np.all(counts > 0):
        unavailable = {"available": False, "reason": "training rows do not contain all three classes"}
        return unavailable, unavailable
    weights = len(y) / (len(CLASSES) * np.maximum(counts, 1.0))
    one_hot = np.eye(len(CLASSES))[y]
    coef = np.zeros((x.shape[1], len(CLASSES)), dtype=np.float64)
    intercept = np.zeros(len(CLASSES), dtype=np.float64)
    for _ in range(800):
        logits = x @ coef + intercept
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = (probs - one_hot) * weights[y, None] / len(y)
        coef -= 0.1 * (x.T @ error + 1e-3 * coef)
        intercept -= 0.1 * error.sum(axis=0)
    predictions["balanced_multinomial_logistic"] = [
        CLASSES[i] for i in (xv @ coef + intercept).argmax(axis=1)
    ]
    return _format_traditional_baseline_diagnostics(eval_rows, predictions, metadata)


def _format_traditional_baseline_diagnostics(
    eval_rows: list[dict[str, Any]], predictions: dict[str, list[str]], metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reports: dict[str, Any] = {}
    time_slices: dict[str, Any] = {}
    for name, actions in predictions.items():
        reports[name] = strategy_metrics(eval_rows, actions)
        scored = {"predictions": [
            {"decision_time_unix_ms": row.get("decision_time_unix_ms"),
             "predicted": action, "probabilities": None}
            for row, action in zip(eval_rows, actions, strict=True)
        ]}
        time_slices[name] = temporal_slices(eval_rows, scored)
    rule_names = {"15m_trend_threshold_30bps", "15m_reversal_threshold_30bps"}
    result = {**metadata, "rules": {name: reports[name] for name in rule_names if name in reports}}
    result.update({name: value for name, value in reports.items() if name not in rule_names})
    return result, time_slices


def traditional_baselines(train_rows: list[dict[str, Any]],
                          eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """CPU-only trend, reversal, and class-balanced multinomial logistic baselines."""
    return traditional_baseline_diagnostics(train_rows, eval_rows)[0]


def cmd_baseline(args: argparse.Namespace) -> int:
    """Run the traditional CPU baselines without loading the slow Vega model."""
    if not Path("/kaggle").is_dir():
        raise RuntimeError("market baseline diagnostics must run on Kaggle; the local machine is code-only")
    prepared = args.prepared_dir.expanduser().resolve(strict=True)
    train_path, eval_path = prepared / "baseline_train.jsonl", prepared / "evaluation.jsonl"
    if not train_path.is_file() or not eval_path.is_file():
        raise ValueError("prepared directory must contain baseline_train.jsonl and evaluation.jsonl")
    started = time.monotonic()
    train_rows, eval_rows = read_jsonl(train_path), read_jsonl(eval_path)
    if not train_rows or not eval_rows:
        raise ValueError("baseline training and evaluation rows must both be nonempty")
    baseline_report, temporal = traditional_baseline_diagnostics(train_rows, eval_rows)
    report = {
        "schema": "td_market_laya_cpu_baseline_diagnostic",
        "version": 1,
        "device": "cpu",
        "train_rows": len(train_rows),
        "evaluation_rows": len(eval_rows),
        "evaluation_labels": dict(Counter(row["label"] for row in eval_rows)),
        "prepared_manifest_sha256": sha256_file(prepared / "seed-manifest.json")
        if (prepared / "seed-manifest.json").is_file() else None,
        "baseline_elapsed_seconds": round(time.monotonic() - started, 6),
        "traditional_baselines": baseline_report,
        "traditional_baseline_time_slices": temporal,
        "limitations": [
            "This is a CPU-only diagnostic; it does not train or load a language model.",
            "The supplied Evaluation sample is reused only for fixed reporting, not for model promotion or tuning.",
        ],
    }
    output = args.output.expanduser()
    if not output.is_absolute() or not str(output).startswith("/kaggle/"):
        raise ValueError("baseline diagnostic output must remain under /kaggle")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), flush=True)
    return 0


def score(learner, rows: list[dict[str, Any]], stage: str = "score") -> dict[str, Any]:
    actual, predicted, readouts, examples, distributions = [], [], Counter(), [], []
    print(f"{stage}: starting {len(rows)} rows", flush=True)
    status = learner.status().get("questions", {}).get(QUESTION_ID, {})
    if status.get("readout") == "head":
        # Learner.decide accepts one text at a time. Prime its pinned 0.17.2 cache in
        # Reader.features batches so held-out CPU scoring does not make one model pass per row.
        prime_features(learner, [row["state_text"] for row in rows])
    for index, row in enumerate(rows, 1):
        expected = row["label"]
        result = learner.decide(row["state_text"], questions=[QUESTION_ID])
        answer = result.get("answers", {}).get(QUESTION_ID, {})
        predicted_label = answer.get("answer")
        if predicted_label not in CLASSES:
            raise ValueError(f"Laya returned unsupported answer: {predicted_label!r}")
        actual.append(expected)
        predicted.append(predicted_label)
        readouts[answer.get("readout", "unknown")] += 1
        probabilities = clean_probabilities(answer.get("probs"))
        distributions.append(probabilities)
        examples.append({
            "decision_time_unix_ms": row.get("decision_time_unix_ms"),
            "actual": expected,
            "predicted": predicted_label,
            "readout": answer.get("readout", "unknown"),
            "probabilities": probabilities,
        })
        if index % 5 == 0 or index == len(rows):
            print(f"{stage}: scored {index}/{len(rows)} rows", flush=True)
    out = metrics(actual, predicted)
    out["readout_counts"] = dict(readouts)
    out["calibration"] = calibration_metrics(actual, distributions)
    out["predictions"] = examples
    out["backtest"] = strategy_metrics(rows, predicted)["backtest"]
    return out


def prequential_score(learner, rows: list[dict[str, Any]], every: int) -> dict[str, Any]:
    """Predict each point in time first, then feed its confirmed future label."""
    timestamps = [row.get("decision_time_unix_ms") for row in rows]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in timestamps):
        raise ValueError("prequential evaluation requires integer decision timestamps")
    ordered = sorted(rows, key=lambda row: row["decision_time_unix_ms"])
    timestamps = [row["decision_time_unix_ms"] for row in ordered]
    if any(left >= right for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("prequential evaluation timestamps must be strictly increasing")

    texts = [row["state_text"] for row in ordered]
    print(f"prequential: starting {len(ordered)} chronological rows", flush=True)
    prime_features(learner, texts)

    actual, predicted, trace, distributions = [], [], [], []
    batch_metrics = []
    batch_start = 0
    for index, row in enumerate(ordered, 1):
        version_before = learner.status()["version"]
        decision = learner.decide(row["state_text"], questions=[QUESTION_ID])
        answer = decision.get("answers", {}).get(QUESTION_ID, {})
        guess = answer.get("answer")
        if guess not in CLASSES:
            raise ValueError(f"Laya returned unsupported answer: {guess!r}")
        truth = row["label"]
        probabilities = clean_probabilities(answer.get("probs"))
        actual.append(truth)
        predicted.append(guess)
        distributions.append(probabilities)
        feedback = learner.feedback(row["state_text"], {QUESTION_ID: truth})
        trace.append({
            "decision_time_unix_ms": row["decision_time_unix_ms"],
            "actual": truth,
            "predicted": guess,
            "predicted_before_feedback": guess,
            "readout": answer.get("readout", "unknown"),
            "probabilities": probabilities,
            "version_before_feedback": version_before,
            "feedback": feedback,
        })
        if index % 5 == 0 or index == len(ordered):
            print(f"prequential: predicted and confirmed {index}/{len(ordered)} rows", flush=True)

        # Laya trains asynchronously every N confirmed answers. Drain at each
        # boundary so later chronological batches see the newly fitted head.
        if index % every == 0:
            learner.flush(timeout=600)
            status = learner.status()
            if status.get("last_error"):
                raise RuntimeError(f"Laya background learning failed: {status['last_error']}")
            batch_metrics.append({
                "through_rows": index,
                "version_after_flush": status["version"],
                "held_after_flush": status.get("held", {}),
                **metrics(actual[batch_start:index], predicted[batch_start:index]),
                "calibration": calibration_metrics(actual[batch_start:index],
                                                    distributions[batch_start:index]),
                "backtest": strategy_metrics(ordered[batch_start:index],
                                              predicted[batch_start:index])["backtest"],
            })
            batch_start = index

    learner.flush(timeout=600)
    final_status = learner.status()
    if final_status.get("last_error"):
        raise RuntimeError(f"Laya background learning failed: {final_status['last_error']}")
    if batch_start < len(ordered):
        batch_metrics.append({
            "through_rows": len(ordered),
            "version_after_flush": final_status["version"],
            "held_after_flush": final_status.get("held", {}),
            **metrics(actual[batch_start:], predicted[batch_start:]),
            "calibration": calibration_metrics(actual[batch_start:], distributions[batch_start:]),
            "backtest": strategy_metrics(ordered[batch_start:], predicted[batch_start:])["backtest"],
        })
    result = metrics(actual, predicted)
    result.update({
        "method": "predict_then_feedback_chronological",
        "feedback_count": len(trace),
        "every": every,
        "calibration": calibration_metrics(actual, distributions),
        "batches": batch_metrics,
        "predictions": trace,
        "backtest": strategy_metrics(ordered, predicted)["backtest"],
    })
    return result


def temporal_slices(rows: list[dict[str, Any]], scored: dict[str, Any]) -> dict[str, Any]:
    """Report contiguous out-of-time sessions, preserving real gaps in the capture timeline."""
    paired = []
    for row, prediction in zip(rows, scored["predictions"], strict=True):
        timestamp = row.get("decision_time_unix_ms")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool):
            return {"available": False, "reason": "evaluation rows lack decision_time_unix_ms"}
        paired.append((timestamp, row["label"], prediction["predicted"]))
    paired.sort(key=lambda item: item[0])
    interval_minutes = max(
        (int(row.get("decision_interval_minutes", 15)) for row in rows), default=15
    )
    gap_limit_ms = interval_minutes * 2 * 60_000
    chunks: list[list[tuple[int, str, str]]] = []
    for item in paired:
        if chunks and item[0] - chunks[-1][-1][0] > gap_limit_ms:
            chunks.append([])
        if not chunks:
            chunks.append([])
        chunks[-1].append(item)
    slices = []
    for index, chunk in enumerate(chunks, 1):
        result = metrics([x[1] for x in chunk], [x[2] for x in chunk])
        rows_by_timestamp = {row["decision_time_unix_ms"]: row for row in rows}
        slice_rows = [rows_by_timestamp[x[0]] for x in chunk]
        result["backtest"] = strategy_metrics(slice_rows, [x[2] for x in chunk])["backtest"]
        prediction_by_timestamp = {
            item.get("decision_time_unix_ms"): item for item in scored["predictions"]
        }
        result["calibration"] = calibration_metrics(
            [x[1] for x in chunk],
            [prediction_by_timestamp.get(x[0], {}).get("probabilities") for x in chunk],
        )
        result.update({
            "slice": index,
            "session_id": f"evaluation-session-{index:02}",
            "from_utc": dt.datetime.fromtimestamp(chunk[0][0] / 1000, tz=dt.timezone.utc).isoformat(),
            "to_utc": dt.datetime.fromtimestamp(chunk[-1][0] / 1000, tz=dt.timezone.utc).isoformat(),
        })
        slices.append(result)
    return {
        "available": True,
        "method": "contiguous_decision_sessions_gap_over_two_intervals",
        "gap_limit_minutes": interval_minutes * 2,
        "slices": slices,
    }


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya inference must run on Kaggle; the local machine is code-only")
    prepared = args.prepared_dir.expanduser().resolve(strict=True)
    examples = prepared / "seed_examples.csv"
    eval_path = prepared / "evaluation.jsonl"
    baseline_train_path = prepared / "baseline_train.jsonl"
    if not examples.is_file() or not eval_path.is_file():
        raise ValueError("prepared directory must contain seed_examples.csv and evaluation.jsonl")
    questions_path = prepared / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.is_file() else QUESTIONS
    rows = read_jsonl(eval_path)
    train_rows = read_jsonl(baseline_train_path) if baseline_train_path.is_file() else []
    traditional_report, traditional_time_slices = (
        traditional_baseline_diagnostics(train_rows, rows) if train_rows else
        ({"available": False, "reason": "prepared data lacks baseline_train.jsonl"}, {})
    )
    manifest_path = prepared / "seed-manifest.json"
    seed_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    seed_examples = read_seed_examples(examples)
    labels = [row["label"] for row in rows]
    baseline_predictions = ["flat"] * len(rows)
    from laya import Learner
    from vegaml.learn.reader import Reader

    device, gpu_budget = select_runtime_device(
        args.device, args.gpu_hours_used_7d, args.gpu_planned_hours
    )
    print("evaluation device selection: " + json.dumps(gpu_budget, sort_keys=True), flush=True)
    reader = Reader.load(device=device, threads=1)
    print("prequential: scoring zero-shot Vega baseline", flush=True)
    baseline = Learner(questions, reader=reader, device=device, auto=False)
    try:
        vega_result = score(baseline, rows, stage="vega_zero_shot")
        if args.vega_only:
            result = {
                "schema": "td_market_laya_vega_only_eval",
                "version": 1,
                "device": device,
                "gpu_budget": gpu_budget,
                "seed_manifest": seed_manifest,
                "seed_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
                "evaluation_labels": dict(Counter(labels)),
                "always_flat": metrics(labels, baseline_predictions),
                "always_flat_backtest": strategy_metrics(rows, baseline_predictions)["backtest"],
                "traditional_baselines": traditional_report,
                "traditional_baseline_time_slices": traditional_time_slices,
                "vega_zero_shot": vega_result,
                "temporal_slices": {"vega_zero_shot": temporal_slices(rows, vega_result)},
                "limitations": [
                    "This is a zero-shot diagnostic on the supplied Evaluation sample.",
                    "A sampled Evaluation set is not model-promotion evidence.",
                ],
            }
            report_path = prepared / "vega-evaluation-report.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                                   encoding="utf-8")
            print(json.dumps({"report": str(report_path), "device": device,
                              "evaluation_rows": len(rows), "vega_zero_shot": vega_result},
                             ensure_ascii=False, indent=2, default=str))
            return 0
        candidate_init_started = time.monotonic()
        print(f"prequential: initializing seeded Learner from {len(seed_examples)} examples",
              flush=True)
        candidate = Learner(questions, examples=seed_examples, reader=reader,
                            device=device, auto=False, workdir=str(prepared / "workdir"))
        print(f"prequential: seeded Learner initialized in "
              f"{time.monotonic() - candidate_init_started:.1f}s", flush=True)
        try:
            candidate_result = score(candidate, rows, stage="seeded_learner")
            candidate.save_bundle(str(prepared / "learner-seed.bundle.zip"))
            result = {
                "schema": "td_market_laya_continuous_eval",
                "version": 1,
                "device": device,
                "gpu_enabled": device == "cuda",
                "gpu_budget": gpu_budget,
                "seed_rows": len(seed_examples),
                "seed_manifest": seed_manifest,
                "seed_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
                "evaluation_labels": dict(Counter(labels)),
                "always_flat": metrics(labels, baseline_predictions),
                "always_flat_backtest": strategy_metrics(rows, baseline_predictions)["backtest"],
                "traditional_baselines": traditional_report,
                "traditional_baseline_time_slices": traditional_time_slices,
                "vega_zero_shot": vega_result,
                "seeded_learner": candidate_result,
                "temporal_slices": {
                    "vega_zero_shot": temporal_slices(rows, vega_result),
                    "seeded_learner": temporal_slices(rows, candidate_result),
                },
                "learner_status": candidate.status(),
                "seed_report": candidate.report,
            }
            report_path = prepared / "evaluation-report.json"
            report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n",
                                   encoding="utf-8")
            print(json.dumps({"report": str(report_path), "always_flat": result["always_flat"],
                              "vega_zero_shot": vega_result, "seeded_learner": candidate_result,
                              "learner_status": candidate.status()}, ensure_ascii=False, indent=2, default=str))
        finally:
            candidate.close()
    finally:
        baseline.close()
    return 0


def cmd_prequential(args: argparse.Namespace) -> int:
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya prequential evaluation must run on Kaggle; the local machine is code-only")
    if args.every < 1:
        raise ValueError("--every must be a positive number of confirmed answers")
    prepared = args.prepared_dir.expanduser().resolve(strict=True)
    examples_path, eval_path = prepared / "seed_examples.csv", prepared / "evaluation.jsonl"
    if not eval_path.is_file():
        raise ValueError("prepared directory must contain evaluation.jsonl")
    if not args.bundle and not examples_path.is_file():
        raise ValueError("seed_examples.csv is required unless --bundle is supplied")
    initial_bundle = args.bundle.expanduser().resolve(strict=True) if args.bundle else None
    questions_path = prepared / "questions.json"
    questions = json.loads(questions_path.read_text(encoding="utf-8")) if questions_path.is_file() else QUESTIONS
    rows = read_jsonl(eval_path)
    if not rows:
        raise ValueError("evaluation.jsonl is empty")
    baseline_train_path = prepared / "baseline_train.jsonl"
    train_rows = read_jsonl(baseline_train_path) if baseline_train_path.is_file() else []
    traditional_report, traditional_time_slices = (
        traditional_baseline_diagnostics(train_rows, rows) if train_rows else
        ({"available": False, "reason": "prepared data lacks baseline_train.jsonl"}, {})
    )
    manifest_path = prepared / "seed-manifest.json"
    seed_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    seed_examples = read_seed_examples(examples_path) if examples_path.is_file() else []
    workdir = outside_repo(args.workdir, "prequential workdir")
    workdir.mkdir(parents=True, exist_ok=True)
    if any(workdir.iterdir()):
        raise ValueError(f"prequential workdir must be empty for a reproducible run: {workdir}")

    from laya import Learner
    from vegaml.learn.reader import Reader

    device, gpu_budget = select_runtime_device(
        args.device, args.gpu_hours_used_7d, args.gpu_planned_hours
    )
    print("prequential device selection: " + json.dumps(gpu_budget, sort_keys=True), flush=True)
    reader = Reader.load(device=device, threads=1)
    baseline = Learner(questions, reader=reader, device=device, auto=False)
    try:
        frozen_vega = score(baseline, rows, stage="vega_zero_shot")
    finally:
        baseline.close()

    learner_init_started = time.monotonic()
    initial_seed_report = None
    seed_bundle_path = initial_bundle
    if initial_bundle:
        print("prequential: restoring learner bundle", flush=True)
        learner = Learner(bundle=str(initial_bundle), reader=reader, device=device,
                          every=args.every, auto=True, workdir=str(workdir))
    else:
        print(f"prequential: building seed store from {len(seed_examples)} examples "
              f"in 32-row batches on {device}", flush=True)
        learn_bytes, start_bytes, initial_seed_report = build_seed_store_batched(
            reader, questions, seed_examples
        )
        seed_bundle = prepared / "learner-seed.bundle.zip"
        with zipfile.ZipFile(seed_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("vega/learn.json", learn_bytes)
            archive.writestr("vega/start.npz", start_bytes)
        seed_bundle_path = seed_bundle
        print(f"prequential: created seed bundle at {seed_bundle}", flush=True)
        learner = Learner(bundle=str(seed_bundle), reader=reader, device=device,
                          every=args.every, auto=True, workdir=str(workdir))
    print(f"prequential: learner initialization finished in "
          f"{time.monotonic() - learner_init_started:.1f}s", flush=True)
    try:
        if learner.questions != questions:
            raise ValueError("previous bundle question schema differs from the prepared data")
        initial_status = learner.status()
        print("prequential: scoring frozen starting learner", flush=True)
        frozen_seeded = score(learner, rows, stage="seeded_frozen_learner")
        print("prequential: scoring online predict-then-feedback learner", flush=True)
        result = prequential_score(learner, rows, args.every)
        bundle = prepared / "learner-prequential.bundle.zip"
        learner.save_bundle(str(bundle))
        report = {
            "schema": "td_market_laya_prequential_eval",
            "version": 1,
            "device": device,
            "gpu_enabled": device == "cuda",
            "gpu_budget": gpu_budget,
            "seed_rows": len(seed_examples),
            "initial_bundle_path": str(initial_bundle) if initial_bundle else None,
            "initial_bundle_sha256": sha256_file(initial_bundle) if initial_bundle else None,
            "seed_bundle_path": str(seed_bundle_path) if seed_bundle_path else None,
            "seed_bundle_sha256": sha256_file(seed_bundle_path) if seed_bundle_path else None,
            "seed_manifest": seed_manifest,
            "seed_manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
            "evaluation_labels": dict(Counter(row["label"] for row in rows)),
            "always_flat": strategy_metrics(rows, ["flat"] * len(rows)),
            "traditional_baselines": traditional_report,
            "traditional_baseline_time_slices": traditional_time_slices,
            "vega_zero_shot": frozen_vega,
            "seeded_frozen_learner": frozen_seeded,
            "prequential_learner": result,
            "frozen_vega_time_slices": temporal_slices(rows, frozen_vega),
            "prequential_time_slices": temporal_slices(
                sorted(rows, key=lambda row: row["decision_time_unix_ms"]), result
            ),
            "initial_learner_status": initial_status,
            "learner_status": learner.status(),
            "seed_report": initial_seed_report if initial_seed_report is not None else learner.report,
            "bundle": str(bundle),
        }
        report_path = prepared / "prequential-report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
                               encoding="utf-8")
        print(json.dumps({"report": str(report_path), "bundle": str(bundle),
                          "vega_zero_shot": frozen_vega,
                          "seeded_frozen_learner": frozen_seeded,
                          "prequential_learner": result,
                          "learner_status": learner.status()}, ensure_ascii=False, indent=2, default=str))
    finally:
        learner.close()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya feedback service must run on Kaggle; the local machine is code-only")
    prepared = args.prepared_dir.expanduser().resolve(strict=True)
    questions = prepared / "questions.json"
    examples = prepared / "seed_examples.csv"
    bundle = prepared / "learner-seed.bundle.zip"
    workdir = outside_repo(args.workdir, "learner workdir")
    audit_log = outside_repo(
        args.audit_log or workdir.with_name(workdir.name + "-feedback.jsonl"),
        "feedback audit log",
    )
    if not questions.is_file() or not examples.is_file() or not bundle.is_file():
        raise ValueError("run prepare and evaluate first; expected questions, seed examples, and seed bundle")
    if not os.environ.get("VEGAML_LEARN_JWT_SECRET"):
        raise ValueError("set VEGAML_LEARN_JWT_SECRET before starting the feedback service")
    command = [sys.executable, str(Path(__file__).with_name("laya_market_feedback_server.py")),
               "--questions", str(questions), "--bundle", str(bundle),
               "--workdir", str(workdir), "--audit-log", str(audit_log),
               "--every", str(args.every), "--host", args.host, "--port", str(args.port),
               "--device", args.device, "--max-runtime-seconds", str(args.max_runtime_seconds)]
    if args.gpu_hours_used_7d is not None:
        command.extend(["--gpu-hours-used-7d", str(args.gpu_hours_used_7d)])
    if args.gpu_planned_hours is not None:
        command.extend(["--gpu-planned-hours", str(args.gpu_planned_hours)])
    print(f"Starting quota-aware Laya 0.5 learner; feedback journal: {audit_log}", flush=True)
    return subprocess.run(command, check=False).returncode


def cmd_export(args: argparse.Namespace) -> int:
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya learner state must be exported inside Kaggle; the local machine is code-only")
    prepared = args.prepared_dir.expanduser().resolve(strict=True)
    bundle = prepared / "learner-seed.bundle.zip"
    workdir = outside_repo(args.workdir, "learner workdir")
    output = outside_repo(args.output_bundle, "bundle output")
    if not bundle.is_file():
        raise ValueError(f"seed bundle not found: {bundle}")
    output.parent.mkdir(parents=True, exist_ok=True)
    from laya import Learner

    learner = Learner(bundle=bundle, workdir=str(workdir), device="cpu", auto=False)
    try:
        learner.save_bundle(str(output))
        print(json.dumps({"bundle": str(output), "status": learner.status()}, ensure_ascii=False, indent=2))
    finally:
        learner.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prepare = subs.add_parser("prepare", help="convert Rust JSONL rows into Vega seed/evaluation inputs")
    prepare.add_argument("--train-jsonl", required=True, type=Path)
    prepare.add_argument("--evaluation-jsonl", required=True, type=Path)
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.set_defaults(func=cmd_prepare)
    evaluate = subs.add_parser("evaluate", help="GPU-preferred Vega and seeded Learner comparison")
    evaluate.add_argument("--prepared-dir", required=True, type=Path)
    add_device_budget_arguments(evaluate)
    evaluate.add_argument("--vega-only", action="store_true",
                          help="persist a zero-shot diagnostic and skip slow seed Learner initialization")
    evaluate.set_defaults(func=cmd_evaluate)
    baseline = subs.add_parser("baseline", help="fast CPU-only traditional baseline diagnostic by time session")
    baseline.add_argument("--prepared-dir", required=True, type=Path)
    baseline.add_argument("--output", type=Path,
                          default=Path("/kaggle/working/laya-market-cpu-baseline-diagnostic.json"))
    baseline.set_defaults(func=cmd_baseline)
    prequential = subs.add_parser("prequential", help="chronological GPU-preferred predict-then-feedback evaluation")
    prequential.add_argument("--prepared-dir", required=True, type=Path)
    prequential.add_argument("--workdir", required=True, type=Path)
    add_device_budget_arguments(prequential)
    prequential.add_argument("--bundle", type=Path, help="resume from a previous Laya ZIP bundle")
    prequential.add_argument("--every", type=int, default=50)
    prequential.set_defaults(func=cmd_prequential)
    serve = subs.add_parser("serve", help="serve quota-aware decisions and confirmed feedback on Kaggle")
    serve.add_argument("--prepared-dir", required=True, type=Path)
    serve.add_argument("--workdir", required=True, type=Path)
    add_device_budget_arguments(serve)
    serve.add_argument("--max-runtime-seconds", type=int, default=1800,
                       help="stop the Kaggle feedback service before its planned GPU allocation expires")
    serve.add_argument("--audit-log", type=Path,
                       help="confirmed feedback journal outside the repository (defaults beside workdir)")
    serve.add_argument("--every", type=int, default=50)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)
    export = subs.add_parser("export", help="export latest workdir state as a portable Laya bundle")
    export.add_argument("--prepared-dir", required=True, type=Path)
    export.add_argument("--workdir", required=True, type=Path)
    export.add_argument("--output-bundle", required=True, type=Path)
    export.set_defaults(func=cmd_export)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        raise SystemExit(arguments.func(arguments))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Laya market continuous learner failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
