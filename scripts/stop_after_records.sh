#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="$1"
RECORDS="$2"
shift 2

while true; do
  count=$(wc -l < "$LOG_PATH")
  if [[ "$count" -ge "$RECORDS" ]]; then
    kill "$@" 2>/dev/null || true
    exit 0
  fi
  sleep 2
done
