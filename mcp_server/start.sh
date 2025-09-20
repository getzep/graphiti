#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${PORT:=8080}"
PYTHON_BIN="${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}"
if [[ -z "${PYTHON_BIN}" && -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="$(pwd)/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" graphiti_mcp_server.py --transport http --host 0.0.0.0 --port "$PORT"
