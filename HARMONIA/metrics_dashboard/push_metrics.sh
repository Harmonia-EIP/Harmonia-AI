#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./push_metrics.sh /path/to/eval_report.json
# Env vars:
#   METRICS_URL   (default: https://harmonia.mcoet.com/receiver.php)
#   METRICS_TOKEN (required)

METRICS_URL="${METRICS_URL:-https://harmonia.mcoet.com/receiver.php}"
METRICS_TOKEN="${METRICS_TOKEN:-}"
INPUT_FILE="${1:-}"

if [[ -z "$INPUT_FILE" ]]; then
  echo "Usage: $0 /path/to/eval_report.json"
  exit 1
fi

if [[ ! -f "$INPUT_FILE" ]]; then
  echo "Error: file not found: $INPUT_FILE"
  exit 1
fi

if [[ -z "$METRICS_TOKEN" ]]; then
  echo "Error: METRICS_TOKEN is not set. Set it in your environment or export before running."
  exit 1
fi

curl --fail --show-error \
  -X POST "$METRICS_URL" \
  -H "Authorization: Bearer $METRICS_TOKEN" \
  -F "metrics_file=@$INPUT_FILE;type=application/json"

echo

