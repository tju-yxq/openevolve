#!/bin/bash -l
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
RUN="$ROOT/runs/dsl_showcase_pair8k_seed201_20260725"
GEN="$ROOT/runs/dsl_showcase_llm_smoke_v2_seed201_20260725/search"

cd "$ROOT"
export PYTHONPATH="$ROOT"
export EQUIFORMER_ROOT=/home/20262202788/equiformer
export EQUIVARIANT_NAS_ROOT="$ROOT"
export NAS_TRAIN_SUBSET_FILE="$ROOT/data_splits/qm9_train_quarter_seed201.npz"
export NAS_GPU_BUDGET_HOURS=12
export NAS_BUDGET_LEDGER="$RUN/budget_ledger.jsonl"
export NAS_TRAINING_TIMEOUT_SECONDS=36000
export NAS_EVALUATOR_TIMEOUT_SECONDS=38000

mkdir -p "$RUN"

exec /home/20262202788/conda-envs/equiformer/bin/python -u scripts/run_dsl_showcase_pair.py \
  --run-dir "$RUN" \
  --generation-search "$GEN" \
  --parent-program "$GEN/candidates/iteration_0000.dsl.json" \
  --child-program "$GEN/candidates/iteration_0001_3cebdda4b3529ea4.dsl.json" \
  --task-contract "$ROOT/runs/dsl_showcase_exact_v1_seed201_20260725/task_contract.json" \
  --project-root "$ROOT" \
  --equiformer-root /home/20262202788/equiformer \
  --data-path /home/20262202788/equiformer/datasets/qm9 \
  --train-subset-file "$ROOT/data_splits/qm9_train_quarter_seed201.npz" \
  --max-steps 8000 \
  --batch-size 32 \
  --seed 201 \
  --eval-interval-epochs 10
