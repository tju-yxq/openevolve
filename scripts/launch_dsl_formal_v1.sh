#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/20262202788/equivariant-nas}"
CONFIG="${CONFIG:-$PROJECT/configs/dsl_formal_v1_qm9_alpha_seed201.json}"
RUN_ROOT="${1:-$PROJECT/runs/dsl_formal_v1_seed201}"
MODE="${2:-}"
OPENEVOLVE_PY="${OPENEVOLVE_PY:-/home/20262202788/conda-envs/openevolve/bin/python}"
EQUIFORMER_PY="${EQUIFORMER_PY:-/home/20262202788/conda-envs/equiformer/bin/python}"

mkdir -p "$RUN_ROOT"
touch "$RUN_ROOT/supervisor.log"
exec > >(tee -a "$RUN_ROOT/controller.log") 2>&1

export PYTHONPATH="$PROJECT"
export EQUIFORMER_PYTHON="$EQUIFORMER_PY"
export NAS_GPU_BUDGET_HOURS="${NAS_GPU_BUDGET_HOURS:-80.0}"
export NAS_FALLBACK_SECONDS_PER_STEP="${NAS_FALLBACK_SECONDS_PER_STEP:-0.12}"
export NAS_BUDGET_LEDGER="${NAS_BUDGET_LEDGER:-$RUN_ROOT/budget_ledger.jsonl}"

"$EQUIFORMER_PY" "$PROJECT/scripts/preflight_dsl_formal_v1.py" --config "$CONFIG" --run-root "$RUN_ROOT"

if [ "$MODE" = "--preflight-only" ]; then
  exit 0
fi

"$OPENEVOLVE_PY" "$PROJECT/scripts/run_dsl_evolution.py" \
  --initial-program "$RUN_ROOT/initial_program.dsl.json" \
  --task-contract "$PROJECT/configs/qm9_alpha_formal_v1_task.json" \
  --evaluator-file "$PROJECT/openevolve_adapter/evaluator.py" \
  --config /home/20262202788/openevolve/configs/local_glm_5_2.yaml \
  --output "$RUN_ROOT/search" \
  --openevolve-root /home/20262202788/openevolve \
  --iterations 32 \
  --valid-candidate-target 8 \
  --valid-per-factor-target 2 \
  --max-steps 8000 \
  --batch-size 32 \
  --train-subset-file "$PROJECT/data_splits/qm9_train_quarter_seed201.npz" \
  --eval-interval-epochs 10 \
  --seed 201 \
  --repair-attempts 1 \
  --forced-factor-sequence F2.2,F4.4,F5.3,F6.3,F2.2,F4.4,F5.3,F6.3

"$EQUIFORMER_PY" "$PROJECT/scripts/run_dsl_multifidelity_cycles.py" \
  --root "$RUN_ROOT/multifidelity" \
  --search-dir "$RUN_ROOT/search" \
  --parent-program "$RUN_ROOT/initial_program.dsl.json" \
  --project-root "$PROJECT" \
  --equiformer-root /home/20262202788/equiformer \
  --data-path /home/20262202788/equiformer/datasets/qm9 \
  --task-contract "$PROJECT/configs/qm9_alpha_formal_v1_task.json" \
  --quarter-subset-file "$PROJECT/data_splits/qm9_train_quarter_seed201.npz" \
  --python "$EQUIFORMER_PY" \
  --gpu-budget-hours "$NAS_GPU_BUDGET_HOURS" \
  --seed 201

"$EQUIFORMER_PY" "$PROJECT/scripts/finalize_dsl_formal_v1.py" \
  --root "$RUN_ROOT" \
  --search-dir "$RUN_ROOT/search" \
  --multifidelity-root "$RUN_ROOT/multifidelity" \
  --project-root "$PROJECT" \
  --equiformer-root /home/20262202788/equiformer \
  --data-path /home/20262202788/equiformer/datasets/qm9 \
  --task-contract "$PROJECT/configs/qm9_alpha_formal_v1_task.json" \
  --python "$EQUIFORMER_PY" \
  --seed 201
