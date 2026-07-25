#!/usr/bin/env bash
set -u

PROJECT="${PROJECT:-/home/20262202788/equivariant-nas}"
RUN_ROOT="${1:-$PROJECT/runs/dsl_formal_v1_seed201}"
SUPERVISOR_ONCE="${SUPERVISOR_ONCE:-0}"
LAUNCH_MODE="${LAUNCH_MODE:-}"
PYTHON="${EQUIFORMER_PYTHON:-/home/20262202788/conda-envs/equiformer/bin/python}"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/supervisor.lock"
if ! flock -n 9; then
  printf '[%s] another supervisor holds the run lock; exiting\n' "$(date -Iseconds)" >>"$RUN_ROOT/supervisor.log"
  exit 2
fi

is_completed() {
  local state="$RUN_ROOT/multifidelity/state.json"
  [ -f "$state" ] || return 1
  "$PYTHON" - "$state" <<'PY'
import json
import sys
try:
    state = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if state.get("stage") == "completed_validation_selection" else 1)
PY
}

while [ ! -f "$RUN_ROOT/STOP" ]; do
  if is_completed; then
    printf '[%s] formal V1 run already completed; supervisor exiting\n' "$(date -Iseconds)" >>"$RUN_ROOT/supervisor.log"
    touch "$RUN_ROOT/COMPLETED"
    break
  fi
  launch_args=("$RUN_ROOT")
  if [ -n "$LAUNCH_MODE" ]; then
    launch_args+=("$LAUNCH_MODE")
  fi
  bash "$PROJECT/scripts/launch_dsl_formal_v1.sh" "${launch_args[@]}" >>"$RUN_ROOT/supervisor.log" 2>&1
  code=$?
  printf '[%s] launcher exited %s\n' "$(date -Iseconds)" "$code" >>"$RUN_ROOT/supervisor.log"
  if [ "$code" -eq 0 ] && is_completed; then
    touch "$RUN_ROOT/COMPLETED"
    break
  fi
  if [ "$SUPERVISOR_ONCE" = "1" ]; then
    break
  fi
  sleep 30
done
