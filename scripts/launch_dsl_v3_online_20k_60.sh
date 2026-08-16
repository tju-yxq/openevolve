#!/usr/bin/env bash
set -euo pipefail

: "${EQUINAS_ROOT:=/home/yifei/equiNAS}"
: "${EQUINAS_RUN_ROOT:=/mlplatform/equiNAS/online_v3_20k_60}"
: "${EQUIFORMER_PYTHON:=python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EQUIFORMER_ROOT:=$EQUINAS_ROOT/source/equiformer}"
: "${EQUIFORMER_V3_ROOT:=$EQUINAS_ROOT/workspace/equiformer_v3}"
: "${EQUINAS_QM9_PATH:=$EQUIFORMER_ROOT/datasets/qm9}"
: "${EQUINAS_QUARTER_SUBSET_FILE:=$PROJECT_ROOT/data_splits/qm9_train_quarter_seed201.npz}"
: "${EQUINAS_INITIAL_PROGRAM:=$PROJECT_ROOT/artifacts/online_seed/equiformer_v3_seed.dsl.json}"
export EQUINAS_ROOT EQUINAS_RUN_ROOT EQUIFORMER_PYTHON EQUIFORMER_ROOT EQUIFORMER_V3_ROOT
export EQUINAS_QM9_PATH EQUINAS_QUARTER_SUBSET_FILE EQUINAS_INITIAL_PROGRAM

cd "$PROJECT_ROOT"
if [[ ! -f "$EQUINAS_INITIAL_PROGRAM" ]]; then
  mkdir -p "$(dirname "$EQUINAS_INITIAL_PROGRAM")"
  "$EQUIFORMER_PYTHON" scripts/export_dsl_v3_seed.py \
    --model-config configs/dsl_v3_qm9_alpha_seed.json \
    --output "$(dirname "$EQUINAS_INITIAL_PROGRAM")"
fi

"$EQUIFORMER_PYTHON" scripts/preflight_dsl_v3_online.py --run-root "$EQUINAS_RUN_ROOT"

# Deliberately stop at the frozen winner. Final Test requires a separately reviewed,
# evaluation-only adapter and is never run implicitly during search/training.
exec "$EQUIFORMER_PYTHON" scripts/run_dsl_v3_online_multifidelity.py \
  --run-root "$EQUINAS_RUN_ROOT" \
  --generator-command "$EQUIFORMER_PYTHON scripts/generate_dsl_v3_online_child.py" \
  --trainer-command "$EQUIFORMER_PYTHON scripts/train_dsl_v3_online_candidate.py --endpoint {endpoint} --checkpoint {checkpoint}" \
  --stop-after-stage evaluate_test
