#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/20262202788/equivariant-nas
RUNS="${ROOT}/runs"
PYTHON=/home/20262202788/conda-envs/equiformer/bin/python
DATA=/home/20262202788/equiformer/datasets/qm9
export PYTHONPATH="${ROOT}"
export EQUIFORMER_ROOT=/home/20262202788/equiformer

common=(
  --architecture-spec "${ROOT}/configs/baseline_spec.json"
  --equiformer-root "${EQUIFORMER_ROOT}"
  --input-irreps 5x0e --target 1 --data-path "${DATA}"
  --feature-type one_hot --batch-size 64
  --reference-steps-per-epoch 859 --eval-interval-steps 2 --epochs 300
  --radius 5 --num-basis 128 --drop-path 0 --weight-decay 5e-3
  --lr 5e-4 --min-lr 1e-6 --workers 0 --print-freq 100
  --no-model-ema --no-amp
)

rm -rf "${RUNS}/gate0_cont4" "${RUNS}/gate0_resume4"
"${PYTHON}" -u -m equivariant_nas.training.fixed_step_trainer \
  --output-dir "${RUNS}/gate0_cont4" --max-steps 4 "${common[@]}" >/dev/null
"${PYTHON}" -u -m equivariant_nas.training.fixed_step_trainer \
  --output-dir "${RUNS}/gate0_resume4" --max-steps 4 \
  --resume-step "${RUNS}/gate0_trainer_2steps/checkpoint_last.pth" \
  "${common[@]}" >/dev/null

"${PYTHON}" - <<'PY'
import torch

p1 = "/home/20262202788/equivariant-nas/runs/gate0_cont4/checkpoint_last.pth"
p2 = "/home/20262202788/equivariant-nas/runs/gate0_resume4/checkpoint_last.pth"
a = torch.load(p1, map_location="cpu")
b = torch.load(p2, map_location="cpu")
maxdiff = max(
    (a["model"][key] - b["model"][key]).abs().max().item()
    for key in a["model"]
    if a["model"][key].numel()
)
print(
    {
        "global_steps": (a["global_step"], b["global_step"]),
        "max_parameter_abs_diff": maxdiff,
        "best_val_mae": (a["best_val_err"], b["best_val_err"]),
    }
)
# torch_scatter uses CUDA atomic reductions, so bitwise equality is not a
# realistic promise even with an identical sample stream. This threshold
# detects a resume/data-order bug while accepting expected kernel-level noise.
if maxdiff > 1.0e-5:
    raise SystemExit("numerically equivalent resume failed")
PY
