#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "run this installer as root (for example: sudo $0)" >&2
  exit 1
fi

SOURCE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
INSTALL_ROOT=/opt/td-market-data-relay
ENV_DIR=/etc/td-market-data-relay
ENV_FILE="$ENV_DIR/sync.env"
UNIT_NAME=td-market-data-sync.service
PYTHON_BIN=${PYTHON_BIN:-python3.13}
UV_BIN=${UV_BIN:-uv}

for required in collector.py sync.py requirements.txt deploy/td-market-data-sync.service; do
  if [[ ! -f "$SOURCE_ROOT/$required" ]]; then
    echo "required source file is missing: $SOURCE_ROOT/$required" >&2
    exit 1
  fi
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN is required; install Python 3.13 first" >&2
  exit 1
fi
if ! command -v "$UV_BIN" >/dev/null 2>&1; then
  echo "$UV_BIN is required; install uv first or set UV_BIN to its absolute path" >&2
  exit 1
fi
if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemd is required for this installer" >&2
  exit 1
fi

install -d -o root -g root -m 0755 "$INSTALL_ROOT"
install -d -o root -g root -m 0750 "$ENV_DIR"
install -m 0644 "$SOURCE_ROOT/collector.py" "$INSTALL_ROOT/collector.py"
install -m 0644 "$SOURCE_ROOT/sync.py" "$INSTALL_ROOT/sync.py"
install -m 0644 "$SOURCE_ROOT/requirements.txt" "$INSTALL_ROOT/requirements.txt"

if ! getent group td-market-data-sync >/dev/null 2>&1; then
  groupadd --system td-market-data-sync
fi
if ! id td-market-data-sync >/dev/null 2>&1; then
  useradd --system --gid td-market-data-sync --no-create-home --home-dir /nonexistent \
    --shell /usr/sbin/nologin td-market-data-sync
fi

if [[ -e "$ENV_FILE" || -L "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" || -L "$ENV_FILE" ]]; then
    echo "$ENV_FILE must be a regular file, not a symlink" >&2
    exit 1
  fi
  chown root:root "$ENV_FILE"
  chmod 0600 "$ENV_FILE"
fi

if [[ ! -e "$ENV_FILE" ]]; then
  if [[ ! -t 0 ]]; then
    echo "missing $ENV_FILE; run this installer from a TTY to enter the Relay URL and API key" >&2
    exit 1
  fi
  read -r -p "Relay HTTPS URL [https://us-relay-2.linair99.de5.net]: " relay_url
  relay_url=${relay_url:-https://us-relay-2.linair99.de5.net}
  read -r -s -p "Relay API key (same value as VPS RELAY_API_KEY): " relay_key
  printf '\n'
  if [[ ! "$relay_url" =~ ^https://[^[:space:]]+$ ]]; then
    echo "Relay URL must be an HTTPS URL without whitespace" >&2
    exit 1
  fi
  if [[ ! "$relay_key" =~ ^[A-Za-z0-9_-]{32,}$ ]]; then
    echo "Relay API key must contain at least 32 letters, digits, '_' or '-'" >&2
    unset relay_key
    exit 1
  fi
  umask 077
  printf 'TD_RELAY_BASE_URL=%s\nTD_RELAY_API_KEY=%s\n' "$relay_url" "$relay_key" >"$ENV_FILE"
  unset relay_key
  chown root:root "$ENV_FILE"
  chmod 0600 "$ENV_FILE"
fi

python3 - "$ENV_FILE" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
values = {}
for line_number, line in enumerate(path.read_text().splitlines(), 1):
    if not line or line.startswith("#"):
        continue
    if "=" not in line:
        raise SystemExit(f"invalid sync environment file at line {line_number}")
    name, value = line.split("=", 1)
    if name not in {"TD_RELAY_BASE_URL", "TD_RELAY_API_KEY"} or name in values:
        raise SystemExit(f"invalid or duplicate setting in sync environment file at line {line_number}")
    values[name] = value
if not re.fullmatch(r"https://[^\s]+", values.get("TD_RELAY_BASE_URL", "")):
    raise SystemExit("sync environment file needs an HTTPS TD_RELAY_BASE_URL")
if not re.fullmatch(r"[A-Za-z0-9_-]{32,}", values.get("TD_RELAY_API_KEY", "")):
    raise SystemExit("sync environment file needs a 32-character TD_RELAY_API_KEY")
PY

if [[ $(stat -c '%a' "$ENV_FILE") != 600 ]]; then
  chmod 0600 "$ENV_FILE"
fi
chown root:root "$ENV_FILE"

if [[ -L "$INSTALL_ROOT/.venv" ]]; then
  echo "$INSTALL_ROOT/.venv must not be a symlink" >&2
  exit 1
fi
install -d -o root -g root -m 0755 "$INSTALL_ROOT/.tmp"
TMPDIR="$INSTALL_ROOT/.tmp" "$UV_BIN" venv --clear --python "$PYTHON_BIN" "$INSTALL_ROOT/.venv"
TMPDIR="$INSTALL_ROOT/.tmp" UV_CACHE_DIR="$INSTALL_ROOT/.uv-cache" \
  "$UV_BIN" pip install --no-cache --python "$INSTALL_ROOT/.venv/bin/python" \
  -r "$INSTALL_ROOT/requirements.txt"
rm -rf "$INSTALL_ROOT/.tmp"
install -m 0644 "$SOURCE_ROOT/deploy/td-market-data-sync.service" \
  "/etc/systemd/system/$UNIT_NAME"
systemctl daemon-reload
systemctl enable --now "$UNIT_NAME"

echo "Local Relay recovery service is enabled."
echo "State and recovered captures: /var/lib/td-market-data-recovered"
echo "Environment file: $ENV_FILE (mode 0600; secret value was not displayed)"
echo "Check status with: systemctl status $UNIT_NAME"
echo "Follow logs with: journalctl -u $UNIT_NAME -f"
