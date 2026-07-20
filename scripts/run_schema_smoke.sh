#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-48}"
ITERATIONS="${2:-8}"
OUTPUT="runs/smoke_qd_repair_seed${SEED}"

source /home/20262202788/.config/openevolve/apis.env
export http_proxy=http://127.0.0.1:12356
export https_proxy=http://127.0.0.1:12356
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export PYTHONPATH=.
export EQUIVARIANT_NAS_ROOT=/home/20262202788/equivariant-nas

cd /home/20262202788/equivariant-nas
rm -rf "$OUTPUT"
/home/20262202788/conda-envs/openevolve/bin/python \
  scripts/run_factorized_evolution.py \
  --initial-program openevolve_adapter/initial_program.py \
  --evaluator-file openevolve_adapter/evaluator.py \
  --config /home/20262202788/openevolve/configs/local_glm_5_2.yaml \
  --output "$OUTPUT" \
  --iterations "$ITERATIONS" \
  --max-steps 0 \
  --seed "$SEED" \
  --skip-symmetry \
  --repair-attempts 1
