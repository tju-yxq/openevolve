#!/bin/bash -l
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
RUN="$ROOT/runs/dsl_showcase_pair8k_seed201_20260725"

while ! grep -q '"status": "completed"' "$RUN/state.json" 2>/dev/null; do
  sleep 30
done

/home/20262202788/conda-envs/equiformer/bin/python \
  "$ROOT/reporting/generate_dsl_showcase_report.py" \
  --run-dir "$RUN" \
  --output-dir "$RUN/report" >>"$RUN/reporting.log" 2>&1
