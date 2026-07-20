#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
while [[ ! -f "$ROOT/runs/stable_random_seed45/summary.json" ]]; do
  sleep 10
done

cd "$ROOT"
export PYTHONPATH=.
/home/20262202788/conda-envs/equiformer/bin/python \
  scripts/run_counterfactual_requests.py \
  --requests runs/stable_factorized_seed45/counterfactual_requests.jsonl \
  --output runs/stage1_interaction_ablation \
  --max-steps 5000 \
  --seed 0
