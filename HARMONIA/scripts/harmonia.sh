#!/usr/bin/env bash
# Wrap any shell command with a dashboard "command" event.
#
# Usage:
#   scripts/harmonia.sh <command> [args...]
#
# Examples:
#   scripts/harmonia.sh python scripts/train.py
#   scripts/harmonia.sh make check
#
set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 64
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${HARMONIA_PYTHON:-${PROJECT_DIR}/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi

CMD_DISPLAY="$*"
START=$(date +%s)

"$@"
STATUS=$?

END=$(date +%s)
DURATION=$((END - START))

if [ -n "$PYTHON_BIN" ]; then
  STATUS_LABEL=$([ "$STATUS" -eq 0 ] && echo "ok" || echo "error")
  "$PYTHON_BIN" - <<PY 2>/dev/null || true
import os, sys
sys.path.insert(0, "$PROJECT_DIR")
try:
    from src.dashboard_events import publish_command
    publish_command(
        ${CMD_DISPLAY@Q},
        status=${STATUS_LABEL@Q},
        duration_seconds=${DURATION},
        detail={"exit_code": int(${STATUS})},
    )
except Exception:
    pass
PY
fi

exit "$STATUS"
