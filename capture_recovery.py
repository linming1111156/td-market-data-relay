"""Validation helpers for provenance attached by the local VPS recovery worker."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "td_market_data_relay_recovery_provenance_v1"


def capture_id_for(path: Path, raw_sha256: str) -> str:
    name_id = path.name.rsplit("-", 1)[-1].removesuffix(".jsonl")
    return name_id if re.fullmatch(r"[0-9a-f]{32}", name_id) else raw_sha256[:32]


def recovery_provenance(path: Path, raw_sha256: str, capture_id: str) -> dict[str, Any] | None:
    sidecar = path.with_name(path.name + ".recovery.json")
    if not sidecar.exists() and not sidecar.is_symlink():
        return None
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError(f"recovery provenance must be a regular file: {sidecar}")
    try:
        value = json.loads(sidecar.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"recovery provenance is invalid: {sidecar}") from error
    recovered = value.get("recovered_at_unix_ns") if isinstance(value, dict) else None
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA or value.get("version") != 1
            or value.get("capture_id") != capture_id or value.get("origin") != "vps_recovered"
            or value.get("raw_sha256") != raw_sha256
            or not isinstance(recovered, int) or isinstance(recovered, bool) or recovered <= 0
            or not isinstance(value.get("relay_host"), str)
            or not value["relay_host"] or len(value["relay_host"]) > 255
            or re.fullmatch(r"[0-9a-f]{64}", raw_sha256) is None):
        raise ValueError(f"recovery provenance does not match capture: {sidecar}")
    return value

