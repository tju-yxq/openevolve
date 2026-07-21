#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/20262202788/equivariant-nas
ROOT=${1:?full-fidelity run root is required}
FIRST_SEARCH=${2:-}
PY=/home/20262202788/conda-envs/openevolve/bin/python
mkdir -p "$ROOT"
cd "$PROJECT"
source /home/20262202788/.config/openevolve/apis.env
export PYTHONPATH="$PROJECT"
export EQUIVARIANT_NAS_ROOT="$PROJECT"
export http_proxy=http://127.0.0.1:12356
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

while [[ ! -f "$ROOT/STOP" ]]; do
  command=(
    "$PY" scripts/run_full_fidelity_cycles.py
    --root "$ROOT"
    --search-valid-target 6
    --search-max-proposals 30
    --promote-20000 3
    --promote-full 3
    --base-search-seed 201
  )
  if [[ -n "$FIRST_SEARCH" ]]; then
    command+=(--cycle-one-search "$FIRST_SEARCH")
  fi
  date -Is >> "$ROOT/supervisor.log"
  "${command[@]}" >> "$ROOT/controller.log" 2>&1
  code=$?
  echo "controller exit code $code" >> "$ROOT/supervisor.log"
  [[ -f "$ROOT/STOP" ]] && break
  sleep 30
done
