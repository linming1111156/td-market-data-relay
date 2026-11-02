from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

RELAY_ROOT = Path(__file__).resolve().parents[3]
if str(RELAY_ROOT) not in sys.path:
    sys.path.insert(0, str(RELAY_ROOT))
from verify_usdm_rl_features import verify as verify_features  # noqa: E402

FEATURE_NAMES = (
    "mark_log_return_1m", "mark_log_return_5m", "mark_log_return_15m",
    "mark_realized_volatility_15m", "mark_realized_volatility_60m",
    "aggtrade_signed_volume_imbalance_1m", "aggtrade_signed_volume_imbalance_5m",
)
ACTION_TARGETS_LOTS = (-1, 0, 1)
LOT_BTC = 0.001
POLICY_INPUT_SIZE = 8
MINUTE_NS = 60_000_000_000
DECISION_NS = 15 * MINUTE_NS
MAX_TRAIN_MINUTES = 525_600
MAX_UPDATES = 20
MINIMUM_NOTIONAL_USDT = 50.0
ENTRY_NOTIONAL_CAP_USDT = 100.0


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_system_temp(path: Path, label: str) -> None:
    resolved = path.resolve()
    if any(resolved == root or root in resolved.parents
           for root in (Path("/tmp"), Path("/var/tmp"))):
        raise ValueError(f"{label} must be on the selected data volume, not /tmp")


def feasible_action_mask(mark_price: float | None, position_lots: int) -> tuple[bool, bool, bool]:
    """Mask only entries ruled out by decision-time price and frozen notional limits.

    Execution volume is intentionally excluded: the replay's participation
    volume belongs to the following minute and isn't known at decision time.
    Action order is short, flat, long.
    """
    if position_lots != 0:
        return True, True, True
    if (mark_price is None or not isinstance(mark_price, (int, float))
            or not math.isfinite(mark_price) or mark_price <= 0):
        return False, True, False
    notional = LOT_BTC * mark_price
    if MINIMUM_NOTIONAL_USDT <= notional <= ENTRY_NOTIONAL_CAP_USDT:
        return True, True, True
    return False, True, False


def read_verified_features(features_path: Path, dataset_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    features_path = features_path.resolve(strict=True)
    dataset_path = dataset_path.resolve(strict=True)
    feature_manifest = verify_features(features_path, dataset_path)
    source_manifest = json.loads((dataset_path / "manifest.json").read_bytes())
    return feature_manifest, source_manifest


def read_role_plan(path: Path, feature_manifest: dict[str, Any], source_manifest: dict[str, Any],
                   feature_manifest_sha256: str, source_manifest_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("role plan must be a regular file")
    try:
        plan = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("role plan is invalid JSON") from error
    if not isinstance(plan, dict) or plan.get("schema") != "td_market_data_usdm_rl_role_plan" or plan.get("version") != 1:
        raise ValueError("unsupported role plan")
    if (plan.get("features_manifest_sha256") != feature_manifest_sha256
            or plan.get("source_dataset_manifest_sha256") != source_manifest_sha256
            or plan.get("feature_window") != feature_manifest["window"]
            or plan.get("source_capture_set_sha256") != source_manifest.get("source_capture_set_sha256")):
        raise ValueError("role plan is not bound to the supplied feature/source manifests")
    windows = plan.get("windows")
    if not isinstance(windows, dict) or "train" not in windows or "validation" not in windows:
        raise ValueError("role plan must predeclare Train and Validation windows")
    if set(windows) - {"train", "validation", "test", "oos"}:
        raise ValueError("role plan contains an unknown data role")
    bounds = []
    for role in ("train", "validation", "test", "oos"):
        if role not in windows:
            continue
        row = windows[role]
        if (not isinstance(row, dict) or not isinstance(row.get("start_ns"), int)
                or isinstance(row.get("start_ns"), bool) or not isinstance(row.get("end_ns"), int)
                or isinstance(row.get("end_ns"), bool)):
            raise ValueError(f"role window {role} is malformed")
        start, end = row["start_ns"], row["end_ns"]
        if (start >= end or start % MINUTE_NS or end % MINUTE_NS
                or start < feature_manifest["window"]["start_ns"]
                or end > feature_manifest["window"]["end_ns"]):
            raise ValueError(f"role window {role} is outside the fixed Dataset or not minute aligned")
        bounds.append((start, end, role))
    if any(right[0] < left[1] for left, right in zip(bounds, bounds[1:])):
        raise ValueError("role windows overlap or are out of chronological order")
    if (not bounds or bounds[0][0] != feature_manifest["window"]["start_ns"]
            or bounds[-1][1] != feature_manifest["window"]["end_ns"]
            or any(left[1] != right[0] for left, right in zip(bounds, bounds[1:]))):
        raise ValueError("role plan must cover the fixed window without gaps")
    train_minutes = (windows["train"]["end_ns"] - windows["train"]["start_ns"]) // MINUTE_NS
    if train_minutes > MAX_TRAIN_MINUTES:
        raise ValueError("Train window exceeds the one-year hard cap")
    if plan.get("protocol") != "chronological_half_open_no_random_split_v1":
        raise ValueError("role plan must declare chronological half-open windows")
    return plan
