#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run this installer as root (for example: sudo $0)" >&2
  exit 1
fi

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_ROOT=/opt/td-market-data-relay
ETC_ROOT=/etc/td-market-data-relay
DATA_ROOT=/mnt/data/td-market-data-local
UV_BIN=${UV_BIN:-uv}
PYTHON_BIN=${PYTHON_BIN:-python3.13}

for required in collector.py relay.py requirements.txt \
  deploy/td-market-data-local-collector@.service \
  deploy/td-market-data-local-relay.service; do
  [[ -f "$SOURCE_ROOT/$required" ]] || {
    echo "required source file is missing: $SOURCE_ROOT/$required" >&2
    exit 1
  }
done
command -v systemctl >/dev/null 2>&1 || { echo "systemd is required" >&2; exit 1; }
command -v "$UV_BIN" >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "$PYTHON_BIN is required" >&2; exit 1; }
mountpoint -q /mnt/data || { echo "/mnt/data must be mounted before local capture is installed" >&2; exit 1; }
[[ ! -L "$DATA_ROOT" ]] || { echo "$DATA_ROOT must not be a symlink" >&2; exit 1; }
for path in "$INSTALL_ROOT/.venv" "$ETC_ROOT/local-capture.env" \
  "$ETC_ROOT/local-proxy.env" "$ETC_ROOT/local-relay.env"; do
  [[ ! -L "$path" ]] || { echo "$path must not be a symlink" >&2; exit 1; }
done

if ! getent group td-marketdata >/dev/null 2>&1; then
  groupadd --system td-marketdata
fi
if ! id td-marketdata >/dev/null 2>&1; then
  useradd --system --gid td-marketdata --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin td-marketdata
fi

install -d -o root -g root -m 0755 "$INSTALL_ROOT" "$ETC_ROOT"
install -d -o td-marketdata -g td-marketdata -m 0750 \
  "$DATA_ROOT" "$DATA_ROOT/hot" "$DATA_ROOT/archive" "$DATA_ROOT/state"
install -m 0644 "$SOURCE_ROOT/collector.py" "$INSTALL_ROOT/collector.py"
install -m 0644 "$SOURCE_ROOT/relay.py" "$INSTALL_ROOT/relay.py"
install -m 0644 "$SOURCE_ROOT/requirements.txt" "$INSTALL_ROOT/requirements.txt"

if [[ ! -e "$INSTALL_ROOT/.venv" ]]; then
  install -d -o root -g root -m 0755 "$INSTALL_ROOT/.uv-tmp"
  TMPDIR="$INSTALL_ROOT/.uv-tmp" "$UV_BIN" venv --python "$PYTHON_BIN" "$INSTALL_ROOT/.venv"
  TMPDIR="$INSTALL_ROOT/.uv-tmp" UV_CACHE_DIR="$INSTALL_ROOT/.uv-cache" \
    "$UV_BIN" pip install --no-cache --python "$INSTALL_ROOT/.venv/bin/python" \
    -r "$INSTALL_ROOT/requirements.txt"
  rm -rf "$INSTALL_ROOT/.uv-tmp"
else
  if ! "$INSTALL_ROOT/.venv/bin/python" -c 'import python_socks, websockets, zstandard' >/dev/null 2>&1; then
    install -d -o root -g root -m 0755 "$INSTALL_ROOT/.uv-tmp"
    TMPDIR="$INSTALL_ROOT/.uv-tmp" UV_CACHE_DIR="$INSTALL_ROOT/.uv-cache" \
      "$UV_BIN" pip install --no-cache --python "$INSTALL_ROOT/.venv/bin/python" \
      -r "$INSTALL_ROOT/requirements.txt"
    rm -rf "$INSTALL_ROOT/.uv-tmp"
  fi
fi

if [[ ! -e "$ETC_ROOT/local-proxy.env" ]]; then
  umask 077
  : >"$ETC_ROOT/local-proxy.env"
fi

if [[ ! -e "$ETC_ROOT/local-capture.env" ]]; then
  umask 077
  printf 'HOT_ROOT=%s/hot\nCURSOR_ROOT=%s/state\nSOURCE_HOST_ID=codex-local\nSOURCE_REVISION=%s\n' \
    "$DATA_ROOT" "$DATA_ROOT" "$(git -C "$SOURCE_ROOT" rev-parse --short=12 HEAD)" \
    >"$ETC_ROOT/local-capture.env"
else
  /usr/bin/python3 - "$ETC_ROOT/local-capture.env" \
    "$(git -C "$SOURCE_ROOT" rev-parse --short=12 HEAD)" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
revision = sys.argv[2]
lines = path.read_text().splitlines()
matches = [index for index, line in enumerate(lines) if line.startswith("SOURCE_REVISION=")]
if len(matches) != 1:
    raise SystemExit("local capture environment must contain exactly one SOURCE_REVISION")
lines[matches[0]] = f"SOURCE_REVISION={revision}"
fd, temporary_name = tempfile.mkstemp(prefix=".local-capture.", dir=path.parent)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_name, path)
except BaseException:
    try:
        os.unlink(temporary_name)
    except FileNotFoundError:
        pass
    raise
PY
fi
if [[ ! -e "$ETC_ROOT/local-relay.env" ]]; then
  relay_key=$(/usr/bin/python3 -c 'import secrets; print(secrets.token_hex(32))')
  umask 077
  cat >"$ETC_ROOT/local-relay.env" <<EOF
HOT_ROOT=$DATA_ROOT/hot
ARCHIVE_ROOT=$DATA_ROOT/archive
HOT_SECONDS=3600
DELETE_SECONDS=262800
MAX_ARCHIVE_BYTES=12884901888
MAX_HOT_BYTES=1073741824
RELAY_BIND=127.0.0.1
RELAY_PORT=18080
RELAY_API_KEY=$relay_key
EOF
  unset relay_key
fi
chown root:root "$ETC_ROOT/local-capture.env" "$ETC_ROOT/local-proxy.env" "$ETC_ROOT/local-relay.env"
chmod 0600 "$ETC_ROOT/local-capture.env" "$ETC_ROOT/local-proxy.env" "$ETC_ROOT/local-relay.env"
/usr/bin/python3 - "$ETC_ROOT/local-capture.env" "$ETC_ROOT/local-relay.env" \
  "$ETC_ROOT/local-proxy.env" "$DATA_ROOT" <<'PY'
import re
import sys
from urllib.parse import urlsplit
from pathlib import Path

capture_path, relay_path, proxy_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
data_root = sys.argv[4]

def read_env(path, allowed):
    values = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"invalid environment file syntax at line {number}: {path.name}")
        name, value = line.split("=", 1)
        if name not in allowed or name in values or not value:
            raise SystemExit(f"invalid or duplicate environment setting at line {number}: {path.name}")
        values[name] = value
    if values.keys() != allowed:
        raise SystemExit(f"environment file has missing or unexpected settings: {path.name}")
    return values

capture = read_env(capture_path, {"HOT_ROOT", "CURSOR_ROOT", "SOURCE_HOST_ID", "SOURCE_REVISION"})
if (capture["HOT_ROOT"] != f"{data_root}/hot"
        or capture["CURSOR_ROOT"] != f"{data_root}/state"
        or capture["SOURCE_HOST_ID"] != "codex-local"
        or re.fullmatch(r"[0-9a-f]{12,40}", capture["SOURCE_REVISION"]) is None):
    raise SystemExit("local capture environment does not bind the expected roots and source")

relay = read_env(relay_path, {"HOT_ROOT", "ARCHIVE_ROOT", "HOT_SECONDS", "DELETE_SECONDS",
                              "MAX_ARCHIVE_BYTES", "MAX_HOT_BYTES", "RELAY_BIND",
                              "RELAY_PORT", "RELAY_API_KEY"})
expected = {"HOT_ROOT": f"{data_root}/hot", "ARCHIVE_ROOT": f"{data_root}/archive",
            "HOT_SECONDS": "3600", "DELETE_SECONDS": "262800",
            "MAX_ARCHIVE_BYTES": "12884901888", "MAX_HOT_BYTES": "1073741824",
            "RELAY_BIND": "127.0.0.1", "RELAY_PORT": "18080"}
if any(relay[key] != value for key, value in expected.items()):
    raise SystemExit("local relay environment does not match the bounded loopback policy")
if re.fullmatch(r"[0-9a-f]{64}", relay["RELAY_API_KEY"]) is None:
    raise SystemExit("local relay API key must be a 32-byte hexadecimal value")

proxy_lines = [line for line in proxy_path.read_text().splitlines() if line and not line.startswith("#")]
if any(not line.startswith("TD_MARKET_DATA_SOCKS5_PROXY=") for line in proxy_lines):
    raise SystemExit("local proxy environment accepts only TD_MARKET_DATA_SOCKS5_PROXY")
if sum(line.startswith("TD_MARKET_DATA_SOCKS5_PROXY=") for line in proxy_lines) > 1:
    raise SystemExit("local proxy environment contains duplicate SOCKS5 settings")
if proxy_lines:
    proxy_url = proxy_lines[0].split("=", 1)[1]
    if proxy_url:
        try:
            parsed = urlsplit(proxy_url)
            valid_port = parsed.port is not None
        except ValueError:
            valid_port = False
            parsed = None
        if (not valid_port or parsed.scheme not in {"socks5", "socks5h"}
                or not parsed.hostname or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise SystemExit("local SOCKS5 proxy setting must be socks5://HOST:PORT or socks5h://HOST:PORT")
PY

install -o root -g root -m 0644 \
  "$SOURCE_ROOT/deploy/td-market-data-local-collector@.service" \
  /etc/systemd/system/td-market-data-local-collector@.service
install -o root -g root -m 0644 \
  "$SOURCE_ROOT/deploy/td-market-data-local-relay.service" \
  /etc/systemd/system/td-market-data-local-relay.service
systemd-analyze verify \
  /etc/systemd/system/td-market-data-local-collector@.service \
  /etc/systemd/system/td-market-data-local-relay.service
systemctl daemon-reload
systemctl enable --now td-market-data-local-relay.service
for stream in \
  spot-depth \
  spot-trades \
  spot-klines-1m \
  usdm-aggtrades \
  usdm-klines-1m \
  usdm-finalized-funding \
  usdm-depth \
  usdm-funding-mark; do
  systemctl enable --now "td-market-data-local-collector@${stream}.service"
done

echo "Local collector data root: $DATA_ROOT"
echo "Hot JSONL retention: 1 hour; compressed archive retention: 73 hours"
echo "Compressed archive cap: 12 GiB; hot cap: 1 GiB"
echo "Relay binds to 127.0.0.1:18080; no public tunnel is configured"
systemctl --no-pager --full status td-market-data-local-relay.service \
  td-market-data-local-collector@spot-depth.service \
  td-market-data-local-collector@spot-trades.service \
  td-market-data-local-collector@spot-klines-1m.service \
  td-market-data-local-collector@usdm-aggtrades.service \
  td-market-data-local-collector@usdm-klines-1m.service \
  td-market-data-local-collector@usdm-finalized-funding.service \
  td-market-data-local-collector@usdm-depth.service \
  td-market-data-local-collector@usdm-funding-mark.service
