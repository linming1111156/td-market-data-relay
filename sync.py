#!/usr/bin/env python3
"""Restore retained VPS captures into a local append-only archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import zstandard


def request(url: str, api_key: str, timeout: int = 60) -> urllib.request.urlopen:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    return urllib.request.urlopen(req, timeout=timeout)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Cloudflare Tunnel URL for the relay")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--since-ns", type=int, default=0, help="Only captures modified since this Unix nanosecond time")
    parser.add_argument("--stream", default="", help="Optional stream filter")
    parser.add_argument("--symbol", default="", help="Optional symbol filter")
    parser.add_argument("--api-key-env", default="TD_RELAY_API_KEY")
    args = parser.parse_args()
    key = os.environ.get(args.api_key_env, "")
    if len(key) < 32:
        raise SystemExit(f"environment variable {args.api_key_env} must contain the relay API key")
    base = args.base_url.rstrip("/")
    parsed_base = urllib.parse.urlsplit(base)
    local_http = parsed_base.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed_base.scheme != "https" and not (parsed_base.scheme == "http" and local_http):
        raise SystemExit("relay base URL must use HTTPS (HTTP is allowed only for localhost)")
    query = urllib.parse.urlencode({"since_ns": args.since_ns, "stream": args.stream, "symbol": args.symbol})
    try:
        with request(f"{base}/v1/captures?{query}", key) as response:
            index = json.load(response)
    except urllib.error.URLError as error:
        raise SystemExit(f"could not fetch relay index: {error}") from error
    args.output.mkdir(parents=True, exist_ok=True)
    restored = skipped = 0
    for entry in index.get("captures", []):
        filename = Path(entry["path"]).name
        if filename in ("", ".", "..") or "/" in filename:
            raise SystemExit("relay returned an unsafe filename")
        target = args.output / entry["stream"] / entry["symbol"] / filename.removesuffix(".zst")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if entry["compressed"]:
                skipped += 1
                continue
            if digest(target) == entry["sha256"]:
                skipped += 1
                continue
            raise SystemExit(f"local capture differs from relay copy: {target}")
        temp = target.with_name(target.name + f".download-{os.getpid()}")
        route = "/v1/captures/" + urllib.parse.quote(str(entry["id"]), safe="")
        try:
            with request(base + route, key, timeout=180) as response, temp.open("xb") as output:
                h = hashlib.sha256()
                while block := response.read(1024 * 1024):
                    h.update(block)
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
                if h.hexdigest() != entry["sha256"]:
                    raise SystemExit(f"download hash mismatch for capture {entry['id']}")
            if entry["compressed"]:
                decoded = target.with_name(target.name + f".decoded-{os.getpid()}")
                try:
                    with temp.open("rb") as source, decoded.open("xb") as output:
                        zstandard.ZstdDecompressor().copy_stream(source, output)
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(decoded, target)
                finally:
                    decoded.unlink(missing_ok=True)
            else:
                os.replace(temp, target)
            restored += 1
            print(f"restored {entry['stream']}/{entry['symbol']}/{target.name}")
        finally:
            temp.unlink(missing_ok=True)
    print(f"sync complete: restored={restored} already_present={skipped}")


if __name__ == "__main__":
    main()
