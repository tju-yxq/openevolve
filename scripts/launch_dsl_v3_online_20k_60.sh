#!/usr/bin/env bash
set -euo pipefail

: "${EQUINAS_RUN_ROOT:=/mlplatform/equiNAS/online_v3_20k_60}"
: "${EQUIFORMER_PYTHON:=python}"
export EQUINAS_RUN_ROOT EQUIFORMER_PYTHON

"$EQUIFORMER_PYTHON" scripts/preflight_dsl_v3_online.py --run-root "$EQUINAS_RUN_ROOT"

# Deliberately stop at the frozen winner. Final Test requires a separately reviewed,
# evaluation-only adapter and is never run implicitly during search/training.
exec "$EQUIFORMER_PYTHON" scripts/run_dsl_v3_online_multifidelity.py \
  --run-root "$EQUINAS_RUN_ROOT" \
  --generator-command "$EQUIFORMER_PYTHON scripts/generate_dsl_v3_online_child.py" \
  --trainer-command "$EQUIFORMER_PYTHON scripts/train_dsl_v3_online_candidate.py --endpoint {endpoint} --checkpoint {checkpoint}" \
  --stop-after-stage evaluate_test
