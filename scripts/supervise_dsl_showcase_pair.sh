#!/bin/bash -l
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
RUN="$ROOT/runs/dsl_showcase_pair8k_seed201_20260725"
mkdir -p "$RUN"

while [[ ! -f "$RUN/STOP" ]]; do
  if [[ -f "$RUN/state.json" ]] && grep -q '"status": "completed"' "$RUN/state.json"; then
    exit 0
  fi
  set +e
  "$ROOT/scripts/launch_dsl_showcase_pair.sh" >>"$RUN/controller.log" 2>&1
  code=$?
  set -e
  echo "$(date --iso-8601=seconds) controller_exit=$code" >>"$RUN/supervisor.log"
  if [[ $code -eq 0 ]]; then
    /home/20262202788/conda-envs/equiformer/bin/python \
      "$ROOT/reporting/generate_dsl_showcase_report.py" \
      --run-dir "$RUN" \
      --output-dir "$RUN/report" >>"$RUN/reporting.log" 2>&1
    exit 0
  fi
  sleep 30
done
