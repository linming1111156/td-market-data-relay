#!/usr/bin/env python3
"""Point-in-time Spot IOC execution replay over a verified Python Dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any


class ReplayRejected(ValueError):
    pass


MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_EVENTS_BYTES = 256 * 1024 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ORDERS_BYTES = 16 * 1024 * 1024
MAX_ORDERS = 10_000
MAX_REPORT_BYTES = 64 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_bounded(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ReplayRejected(f"cannot inspect {path}: {error}") from error
    if size > limit:
        raise ReplayRejected(f"{path.name} exceeds the {limit}-byte limit")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ReplayRejected(f"cannot read {path}: {error}") from error


def parse_decimal(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ReplayRejected(f"{name} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ReplayRejected(f"invalid {name}") from error
    if not result.is_finite() or (positive and result <= 0):
        raise ReplayRejected(f"invalid {name}")
    return result


def verify_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest_path = path / "manifest.json"
    snapshot_path = path / "snapshot.json"
    events_path = path / "events.jsonl"
    try:
        manifest = json.loads(read_bounded(manifest_path, MAX_MANIFEST_BYTES))
        snapshot_bytes = read_bounded(snapshot_path, MAX_SNAPSHOT_BYTES)
        events_bytes = read_bounded(events_path, MAX_EVENTS_BYTES)
        snapshot = json.loads(snapshot_bytes)
        events = [json.loads(line) for line in events_bytes.splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayRejected(f"cannot read Dataset: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != "td_market_data_spot_depth_dataset" or manifest.get("version") != 1:
        raise ReplayRejected("unsupported Spot depth Dataset schema")
    if not isinstance(snapshot, dict) or not isinstance(events, list) or any(not isinstance(row, dict) for row in events):
        raise ReplayRejected("Dataset files must contain an object snapshot and object events")
    if manifest.get("replay_usable") is not True:
        raise ReplayRejected("Dataset is not marked replay-usable")
    if sha256(snapshot_bytes) != manifest.get("snapshot_sha256") or sha256(events_bytes) != manifest.get("events_sha256"):
        raise ReplayRejected("Dataset file hash mismatch")
    if len(events) != manifest.get("event_count"):
        raise ReplayRejected("Dataset event count mismatch")
    window = manifest.get("window")
    if (not isinstance(window, dict) or not isinstance(window.get("start_ns"), int) or
            isinstance(window.get("start_ns"), bool) or not isinstance(window.get("end_ns"), int) or
            isinstance(window.get("end_ns"), bool) or window["start_ns"] >= window["end_ns"]):
        raise ReplayRejected("invalid Dataset window")
    logical = hashlib.sha256(b"td-market-data-spot-depth-dataset-v1\0" + canonical(window)
                             + snapshot_bytes + events_bytes).hexdigest()
    if logical != manifest.get("logical_fingerprint_sha256"):
        raise ReplayRejected("Dataset logical fingerprint mismatch")
    identity = (manifest.get("symbol"), manifest.get("session_id"))
    if (snapshot.get("schema") != "td_market_data_spot_depth_snapshot" or
            (snapshot.get("symbol"), snapshot.get("session_id")) != identity):
        raise ReplayRejected("snapshot identity mismatch")
    snapshot_update_id = snapshot.get("last_update_id")
    if not isinstance(snapshot_update_id, int) or isinstance(snapshot_update_id, bool) or snapshot_update_id < 0:
        raise ReplayRejected("invalid snapshot update ID")
    for index, event in enumerate(events):
        if (event.get("schema") != "td_market_data_spot_depth_event" or
                (event.get("symbol"), event.get("session_id")) != identity):
            raise ReplayRejected(f"event {index}: identity mismatch")
        event_ns, receive_ns = event.get("event_ts_ns"), event.get("receive_ts_ns")
        if not isinstance(event_ns, int) or not isinstance(receive_ns, int) or event_ns > receive_ns:
            raise ReplayRejected(f"event {index}: invalid event/receive time")
        if not (window["start_ns"] <= max(event_ns, receive_ns) < window["end_ns"]):
            raise ReplayRejected(f"event {index}: outside Dataset window")
        first_id, final_id = event.get("first_update_id"), event.get("final_update_id")
        if (not isinstance(first_id, int) or isinstance(first_id, bool) or
                not isinstance(final_id, int) or isinstance(final_id, bool) or
                first_id < 0 or final_id < first_id):
            raise ReplayRejected(f"event {index}: invalid update ID range")
        prior_id = snapshot_update_id if index == 0 else events[index - 1]["final_update_id"]
        if not (first_id <= prior_id + 1 <= final_id):
            raise ReplayRejected(f"event {index}: update ID chain does not bridge prior state")
        if index and max(event_ns, receive_ns) < max(events[index - 1]["event_ts_ns"], events[index - 1]["receive_ts_ns"]):
            raise ReplayRejected(f"event {index}: availability time regressed")
    return manifest, snapshot, events


@dataclass
class Book:
    bids: dict[Decimal, Decimal]
    asks: dict[Decimal, Decimal]

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "Book":
        try:
            if any(not isinstance(row, list) or len(row) != 2 for row in snapshot["bids"] + snapshot["asks"]):
                raise TypeError("snapshot levels must be [price, quantity]")
            bids = {parse_decimal(row[0], "bid price", positive=True): parse_decimal(row[1], "bid quantity", positive=True)
                    for row in snapshot["bids"]}
            asks = {parse_decimal(row[0], "ask price", positive=True): parse_decimal(row[1], "ask quantity", positive=True)
                    for row in snapshot["asks"]}
        except (KeyError, TypeError, IndexError) as error:
            raise ReplayRejected("malformed snapshot levels") from error
        book = cls(bids, asks)
        book.validate()
        return book

    def apply(self, event: dict[str, Any]) -> None:
        for change in event.get("changes", []):
            side = change.get("side")
            price = parse_decimal(change.get("price"), "update price", positive=True)
            quantity = parse_decimal(change.get("quantity"), "update quantity")
            if quantity < 0 or side not in {"BID", "ASK"}:
                raise ReplayRejected("invalid book change")
            levels = self.bids if side == "BID" else self.asks
            if quantity == 0:
                levels.pop(price, None)
            else:
                levels[price] = quantity
        self.validate()

    def validate(self) -> None:
        if not self.bids or not self.asks or max(self.bids) >= min(self.asks):
            raise ReplayRejected("replayed book is empty or crossed")

    def mid(self) -> Decimal:
        return (max(self.bids) + min(self.asks)) / 2


def run_account(
    manifest: dict[str, Any], snapshot: dict[str, Any], events: list[dict[str, Any]],
    orders: list[dict[str, Any]], initial_quote: Decimal, initial_base: Decimal,
    fee_rate: Decimal, latency_ns: int,
) -> dict[str, Any]:
    book = Book.from_snapshot(snapshot)
    window = manifest["window"]
    quote, base = initial_quote, initial_base
    cursor = 0
    fills: list[dict[str, Any]] = []
    order_reports: list[dict[str, Any]] = []
    total_fee = Decimal(0)
    turnover = Decimal(0)
    adverse_cost = Decimal(0)

    for order in orders:
        decision = order["decision_ts_ns"]
        arrival = decision + latency_ns
        while cursor < len(events) and max(events[cursor]["event_ts_ns"], events[cursor]["receive_ts_ns"]) <= decision:
            book.apply(events[cursor]); cursor += 1
        decision_mid = book.mid()
        while cursor < len(events) and max(events[cursor]["event_ts_ns"], events[cursor]["receive_ts_ns"]) <= arrival:
            book.apply(events[cursor]); cursor += 1
        arrival_mid = book.mid()
        side = order["side"]
        remaining = parse_decimal(order["quantity"], "order quantity", positive=True)
        requested = remaining
        before_quote, before_base = quote, base
        order_fee = Decimal(0)
        order_turnover = Decimal(0)
        fill_side = book.asks if side == "BUY" else book.bids
        prices = sorted(fill_side) if side == "BUY" else sorted(fill_side, reverse=True)
        for price in prices:
            if remaining <= 0:
                break
            available = fill_side[price]
            if side == "BUY":
                affordable = quote / (price * (1 + fee_rate)) if price > 0 else Decimal(0)
                quantity = min(remaining, available, affordable)
                if quantity <= 0:
                    break
                notional = quantity * price
                fee = notional * fee_rate
                quote -= notional + fee
                base += quantity
            else:
                quantity = min(remaining, available, base)
                if quantity <= 0:
                    break
                notional = quantity * price
                fee = notional * fee_rate
                base -= quantity
                quote += notional - fee
            fill_side[price] -= quantity
            remaining -= quantity
            total_fee += fee
            order_fee += fee
            turnover += notional
            order_turnover += notional
            adverse = (price - decision_mid) * quantity if side == "BUY" else (decision_mid - price) * quantity
            adverse_cost += adverse
            fill = {"order_id": order["order_id"], "side": side, "quantity": format(quantity, "f"),
                    "price": format(price, "f"), "notional_quote": format(notional, "f"),
                    "fee_quote": format(fee, "f"), "simulated_fill_ts_ns": arrival,
                    "decision_ts_ns": decision, "decision_mid": format(decision_mid, "f"),
                    "arrival_mid": format(arrival_mid, "f"), "adverse_cost_quote": format(adverse, "f")}
            fills.append(fill)
        filled = requested - remaining
        order_reports.append({"order_id": order["order_id"], "side": side,
                              "requested_quantity": format(requested, "f"),
                              "filled_quantity": format(filled, "f"),
                              "remaining_quantity": format(remaining, "f"),
                              "status": "FILLED" if remaining == 0 else "PARTIALLY_FILLED" if filled > 0 else "UNFILLED",
                              "fee_quote": format(order_fee, "f"),
                              "turnover_quote": format(order_turnover, "f"),
                              "quote_balance_before": format(before_quote, "f"),
                              "base_balance_before": format(before_base, "f"),
                              "quote_balance_after": format(quote, "f"),
                              "base_balance_after": format(base, "f")})

    while cursor < len(events):
        book.apply(events[cursor]); cursor += 1
    terminal_mid = book.mid()
    initial_mid = Book.from_snapshot(snapshot).mid()
    initial_equity = initial_quote + initial_base * initial_mid
    final_equity = quote + base * terminal_mid
    return {"initial_balances": {"quote": format(initial_quote, "f"), "base": format(initial_base, "f")},
            "final_balances": {"quote": format(quote, "f"), "base": format(base, "f")},
            "initial_equity_quote": format(initial_equity, "f"),
            "final_equity_quote": format(final_equity, "f"),
            "net_pnl_quote": format(final_equity - initial_equity, "f"),
            "total_fees_quote": format(total_fee, "f"),
            "turnover_quote": format(turnover, "f"),
            "adverse_cost_vs_decision_mid_quote": format(adverse_cost, "f"),
            "terminal_mid": format(terminal_mid, "f"),
            "fills": fills, "orders": order_reports}


def replay(dataset_path: Path, config_path: Path, orders_path: Path) -> dict[str, Any]:
    manifest, snapshot, events = verify_dataset(dataset_path)
    try:
        config_bytes = read_bounded(config_path, 1024 * 1024)
        orders_bytes = read_bounded(orders_path, MAX_ORDERS_BYTES)
        config = json.loads(config_bytes)
        orders = json.loads(orders_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayRejected(f"cannot read replay inputs: {error}") from error
    if not isinstance(config, dict) or config.get("schema") != "td_spot_execution_replay_config" or config.get("version") != 1:
        raise ReplayRejected("unsupported replay config schema")
    if not isinstance(orders, list) or len(orders) > MAX_ORDERS:
        raise ReplayRejected(f"orders must be an array of at most {MAX_ORDERS} intents")
    try:
        initial_quote = parse_decimal(config.get("initial_quote_balance"), "initial quote balance")
        initial_base = parse_decimal(config.get("initial_base_balance"), "initial base balance")
        fee_bps = parse_decimal(config.get("taker_fee_bps"), "taker fee bps")
        latency_ms = config.get("latency_ms")
    except (TypeError, InvalidOperation) as error:
        raise ReplayRejected("invalid config values") from error
    if initial_quote < 0 or initial_base < 0 or not (0 <= fee_bps <= 10_000):
        raise ReplayRejected("balances or taker fee are out of range")
    if not isinstance(latency_ms, int) or isinstance(latency_ms, bool) or not (0 <= latency_ms <= 60_000):
        raise ReplayRejected("latency_ms must be between zero and 60000")
    parsed_orders: list[dict[str, Any]] = []
    previous_decision = -1
    seen_ids: set[str] = set()
    for index, order in enumerate(orders):
        if not isinstance(order, dict) or order.get("type", "MARKET_IOC") != "MARKET_IOC":
            raise ReplayRejected(f"order {index}: only MARKET_IOC intents are supported")
        order_id, side, decision = order.get("order_id"), order.get("side"), order.get("decision_ts_ns")
        if not isinstance(order_id, str) or not order_id or order_id in seen_ids:
            raise ReplayRejected(f"order {index}: invalid or duplicate order_id")
        if side not in {"BUY", "SELL"} or not isinstance(decision, int) or isinstance(decision, bool):
            raise ReplayRejected(f"order {index}: invalid side or decision timestamp")
        if decision < previous_decision or not (manifest["window"]["start_ns"] <= decision < manifest["window"]["end_ns"]):
            raise ReplayRejected(f"order {index}: decisions must be ordered and inside the Dataset window")
        if decision + latency_ms * 1_000_000 >= manifest["window"]["end_ns"]:
            raise ReplayRejected(f"order {index}: arrival occurs outside the Dataset window")
        parse_decimal(order.get("quantity"), f"order {index} quantity", positive=True)
        parsed_orders.append(order)
        previous_decision = decision
        seen_ids.add(order_id)

    fee_rate = fee_bps / Decimal(10_000)
    with localcontext() as context:
        context.prec = 40
        strategy = run_account(manifest, snapshot, events, parsed_orders, initial_quote,
                               initial_base, fee_rate, latency_ms * 1_000_000)
        start = manifest["window"]["start_ns"]
        noop = run_account(manifest, snapshot, events, [], initial_quote, initial_base,
                           fee_rate, 0)
        # Buy once at the opening book, then hold through the window. IOC
        # remainder is left in quote balance when visible depth is insufficient.
        buy_hold_order = []
        if initial_quote > 0:
            # Express a market-buy budget as base quantity at the opening
            # best ask; actual fills still walk the arrival-time book and
            # remain capped by quote balance including fees.
            requested_bh = initial_quote / min(Book.from_snapshot(snapshot).asks)
            if requested_bh > 0:
                buy_hold_order = [{"order_id": "BUY_AND_HOLD", "side": "BUY",
                                   "quantity": format(requested_bh, "f"), "decision_ts_ns": start}]
        buy_hold = run_account(manifest, snapshot, events, buy_hold_order, initial_quote,
                               initial_base, fee_rate, latency_ms * 1_000_000)
    return {"schema": "td_spot_execution_replay_report", "version": 1,
            "dataset_manifest_sha256": sha256(read_bounded(dataset_path / "manifest.json", MAX_MANIFEST_BYTES)),
            "dataset_logical_fingerprint_sha256": manifest["logical_fingerprint_sha256"],
            "config_sha256": sha256(config_bytes), "orders_sha256": sha256(orders_bytes),
            "replay_code_sha256": sha256(Path(__file__).resolve().read_bytes()),
            "replay_git_revision": git_revision(),
            "symbol": manifest["symbol"], "window": manifest["window"],
            "execution_model": {"order_type": "MARKET_IOC", "latency_ms": latency_ms,
                                "taker_fee_bps": format(fee_bps, "f"),
                                "fill_rule": "visible L2 depth at arrival; no hidden liquidity"},
            "accounts": {"strategy": strategy, "NOOP": noop, "BUY_AND_HOLD": buy_hold}}


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or not path.parent.is_dir() or path.exists():
        raise ReplayRejected("output must be a new absolute path in an existing directory")
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n"
    if len(raw) > MAX_REPORT_BYTES:
        raise ReplayRejected("execution report exceeds the 64 MiB limit")
    fd, staging_name = tempfile.mkstemp(prefix=f".{path.name}.partial-", dir=path.parent)
    staging = Path(staging_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.link(staging, path)
        staging.unlink()
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--orders", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = replay(args.dataset, args.config, args.orders)
        write_create_only(args.output, report)
    except (ReplayRejected, OSError, ValueError) as error:
        raise SystemExit(f"Spot execution replay rejected: {error}") from error
    for name, account in report["accounts"].items():
        print(f"{name}: pnl_quote={account['net_pnl_quote']} fees={account['total_fees_quote']} "
              f"turnover={account['turnover_quote']}")
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()
