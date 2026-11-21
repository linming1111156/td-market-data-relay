#!/usr/bin/env python3
"""Run the quota-aware Kaggle Laya 0.5 feedback API with an append-only journal."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def append_event(path: Path, lock: threading.Lock, event: dict[str, Any]) -> None:
    record = {"recorded_at_utc": utc_now(), **event}
    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    with lock, path.open("a", encoding="utf-8") as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())


def serve() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--audit-log", required=True, type=Path)
    parser.add_argument("--every", type=int, default=50)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpu-hours-used-7d", type=float)
    parser.add_argument("--gpu-planned-hours", type=float)
    parser.add_argument("--max-runtime-seconds", type=int, default=1800)
    args = parser.parse_args()
    if not Path("/kaggle").is_dir():
        raise RuntimeError("Laya feedback data and service must stay in Kaggle; the local machine is code-only")

    secret = os.environ.get("VEGAML_LEARN_JWT_SECRET")
    if not secret:
        raise ValueError("set VEGAML_LEARN_JWT_SECRET before starting the feedback service")
    if not args.questions.is_file() or not args.bundle.is_file():
        raise ValueError("questions or learner bundle does not exist")
    if args.max_runtime_seconds < 1:
        raise ValueError("--max-runtime-seconds must be positive")
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)
    audit_lock = threading.Lock()

    from laya import Learner
    from vegaml.learn.serve import create_app
    import uvicorn
    from laya_market_continuous import select_runtime_device

    device, gpu_budget = select_runtime_device(
        args.device, args.gpu_hours_used_7d, args.gpu_planned_hours
    )

    learner = Learner(
        bundle=args.bundle,
        workdir=str(args.workdir),
        device=device,
        every=args.every,
        auto=True,
        on_event=lambda event: append_event(args.audit_log, audit_lock, {
            "kind": "learner_event", "event": event,
        }),
    )
    original_feedback = learner.feedback

    def audited_feedback(text: str, answers: dict[str, Any]) -> dict[str, Any]:
        version_before = learner.status()["version"]
        prediction: dict[str, Any] = {}
        try:
            decision = learner.decide(text, questions=list(answers))
            prediction = decision.get("answers", {})
        except Exception as error:
            append_event(args.audit_log, audit_lock, {
                "kind": "decision_error_before_feedback",
                "text": text,
                "answers": answers,
                "version": version_before,
                "error": repr(error),
            })
            raise
        result = original_feedback(text, answers)
        append_event(args.audit_log, audit_lock, {
            "kind": "confirmed_feedback",
            "text": text,
            "prediction_before_confirmation": prediction,
            "confirmed_answers": answers,
            "version_before": version_before,
            "version_after_acceptance": learner.status()["version"],
            "feedback_result": result,
        })
        return result

    learner.feedback = audited_feedback
    original_accept = learner.accept
    original_discard = learner.discard

    def audited_accept(qid: str) -> Any:
        before = learner.status()["version"]
        result = original_accept(qid)
        append_event(args.audit_log, audit_lock, {
            "kind": "held_batch_accepted",
            "question_id": qid,
            "version_before": before,
            "version_after": learner.status()["version"],
            "result": result,
        })
        return result

    def audited_discard(qid: str) -> None:
        before = learner.status()["version"]
        original_discard(qid)
        append_event(args.audit_log, audit_lock, {
            "kind": "held_batch_discarded",
            "question_id": qid,
            "version": before,
        })

    learner.accept = audited_accept
    learner.discard = audited_discard
    app = create_app(learner, secret=secret)
    append_event(args.audit_log, audit_lock, {
        "kind": "service_started",
        "workdir": str(args.workdir),
        "bundle": str(args.bundle),
        "every": args.every,
        "device": device,
        "gpu_budget": gpu_budget,
        "max_runtime_seconds": args.max_runtime_seconds,
        "learner_status": learner.status(),
    })
    try:
        server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
        timer = threading.Timer(args.max_runtime_seconds, setattr, args=(server, "should_exit", True))
        timer.daemon = True
        timer.start()
        try:
            server.run()
        finally:
            timer.cancel()
    finally:
        try:
            learner.flush(timeout=600)
        finally:
            append_event(args.audit_log, audit_lock, {
                "kind": "service_stopped",
                "learner_status": learner.status(),
            })
            learner.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(serve())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Laya market feedback server failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
