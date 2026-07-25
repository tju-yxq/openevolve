#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-/home/20262202788/equivariant-nas}"
RUN_ROOT="${1:-$PROJECT/runs/dsl_formal_v1_seed201}"
mkdir -p "$RUN_ROOT"

while [ ! -f "$RUN_ROOT/STOP" ]; do
  if ! pgrep -af "launch_dsl_formal_v1.sh.*$RUN_ROOT" >/dev/null 2>&1; then
    bash "$PROJECT/scripts/launch_dsl_formal_v1.sh" "$RUN_ROOT" >>"$RUN_ROOT/supervisor.log" 2>&1
    code=$?
    printf '[%s] launcher exited %s\n' "$(date -Iseconds)" "$code" >>"$RUN_ROOT/supervisor.log"
  fi
  sleep 30
done
