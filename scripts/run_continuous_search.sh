#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/20262202788/equivariant-nas
RUN=${1:?continuous run directory is required}
STOP="$RUN/STOP"
PY=/home/20262202788/conda-envs/openevolve/bin/python

mkdir -p "$RUN"
cd "$ROOT"
source /home/20262202788/.config/openevolve/apis.env
export PYTHONPATH="$ROOT"
export EQUIVARIANT_NAS_ROOT="$ROOT"
export http_proxy=http://127.0.0.1:12356
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"
export NAS_BUDGET_LEDGER="$RUN/budget_ledger.jsonl"
export NAS_GPU_BUDGET_HOURS=1000000

while [[ ! -f "$STOP" ]]; do
  date -Is >> "$RUN/supervisor.log"
  echo "starting or resuming continuous search" >> "$RUN/supervisor.log"
  "$PY" scripts/run_factorized_evolution.py \
    --initial-program openevolve_adapter/initial_program.py \
    --evaluator-file openevolve_adapter/evaluator.py \
    --config /home/20262202788/openevolve/configs/local_glm_5_2.yaml \
    --output "$RUN" \
    --max-steps 5000 \
    --seed 201 \
    --router-mode evidence \
    --router-prior configs/stage1_factor_memory.json \
    --repair-attempts 1 \
    --initial-metrics runs/candidates/8639c8a64dad5d25/seed0_steps5000_87020e2c61/result.json \
    --continuous \
    --resume \
    --stop-file "$STOP" \
    >> "$RUN/console.log" 2>&1 &
  child=$!
  while kill -0 "$child" 2>/dev/null; do
    "$PY" scripts/summarize_continuous_run.py --run "$RUN" \
      >> "$RUN/status_builder.log" 2>&1 || true
    [[ -f "$STOP" ]] && break
    sleep 60
  done
  wait "$child"
  code=$?
  "$PY" scripts/summarize_continuous_run.py --run "$RUN" \
    >> "$RUN/status_builder.log" 2>&1 || true
  echo "driver exit code $code" >> "$RUN/supervisor.log"
  [[ -f "$STOP" ]] && break
  sleep 15
done

echo "continuous search stopped" >> "$RUN/supervisor.log"
