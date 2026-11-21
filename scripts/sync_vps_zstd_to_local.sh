#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
SYNC_ROOT=${SYNC_ROOT:-$SOURCE_ROOT}
SYNC_ENV_FILE=${SYNC_ENV_FILE:-/etc/td-market-data-relay/sync.env}
PYTHON_BIN=${PYTHON_BIN:-$SOURCE_ROOT/.venv/bin/python3}

if [[ ! -f "$SYNC_ENV_FILE" || -L "$SYNC_ENV_FILE" ]]; then
  echo "sync environment file is missing or not a regular file: $SYNC_ENV_FILE" >&2
  echo "Set SYNC_ENV_FILE or create it with TD_RELAY_BASE_URL and TD_RELAY_API_KEY." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable is missing or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN to a Python 3 executable with requirements.txt installed." >&2
  exit 1
fi

# This file is the trusted, mode-0600 config created by install_local_sync_service.sh.
set -a
# shellcheck disable=SC1090
source "$SYNC_ENV_FILE"
set +a
if [[ -z ${TD_RELAY_API_KEY:-} && -n ${RELAY_API_KEY:-} ]]; then
  export TD_RELAY_API_KEY=$RELAY_API_KEY
fi

mkdir -p "$SYNC_ROOT/archive"

exec "$PYTHON_BIN" "$SOURCE_ROOT/sync.py" \
  --output "$SYNC_ROOT/state" \
  --archive-root "$SYNC_ROOT/archive" \
  --direct-archive \
  "$@"
