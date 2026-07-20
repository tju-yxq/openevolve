#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
while [[ ! -f "$ROOT/runs/stage1_interaction_ablation/results.json" ]]; do
  sleep 15
done

cd "$ROOT"
mkdir -p reports/stage1
{
  /home/20262202788/conda-envs/openevolve/bin/python -m pytest -q
  /home/20262202788/conda-envs/openevolve/bin/python -m py_compile \
    equivariant_nas/*.py equivariant_nas/training/*.py \
    scripts/*.py reporting/generate_stage1_report.py
  echo "py_compile: PASS"
} > reports/stage1/test_output.txt 2>&1

PYTHONPATH=. /home/20262202788/conda-envs/equiformer/bin/python \
  reporting/generate_stage1_report.py
touch reports/stage1/SERVER_READY
