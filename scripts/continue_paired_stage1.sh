#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
DRIVER_PID="${1:-184749}"
EVOLUTION="$ROOT/runs/stable_factorized_seed45/evolution.jsonl"

while [[ ! -f "$EVOLUTION" ]] || [[ $(wc -l < "$EVOLUTION") -lt 3 ]]; do
  sleep 5
done
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  sleep 2
done

python - "$ROOT/runs/stable_factorized_seed45/paired_budget_stop.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "reason": "intentional paired-budget stop",
            "completed_evolutionary_candidates": 2,
            "matched_random_valid_target": 2,
            "fidelity_steps": 5000,
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY

cd "$ROOT"
export PYTHONPATH=.
/home/20262202788/conda-envs/equiformer/bin/python scripts/run_random_search.py \
  --output runs/stable_random_seed45 \
  --max-steps 5000 \
  --seed 45 \
  --valid-target 2 \
  --max-proposals 10
