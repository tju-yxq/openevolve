# Batch-32 QM9 Quarter-to-Full Multi-Fidelity Protocol

## Fixed protocol

- Target: QM9 isotropic polarizability alpha (`target=1`).
- Batch size: 32.
- Search training subset: the fixed 27,500-sample artifact
  `data_splits/qm9_train_quarter_seed201.npz`.
- Validation: every 10 completed epochs of the current training dataset, plus a
  mandatory validation at every stage endpoint.
- Test split: locked throughout search and promotion.

## Successive-halving stages

| Stage | Candidates | Cumulative optimizer steps | Training data |
|---|---:|---:|---|
| Search | 8 | 8,000 | fixed quarter |
| Medium | 4 | 80,000 | same fixed quarter |
| Full | 2 | 250,000 | full QM9 train split |

The 8k -> 80k transition is an exact resume on the same subset. At 80k the
training data changes to the full train split and the data-epoch counter resets
to zero while `global_step` remains cumulative. The default transition keeps
model, optimizer, and LR-scheduler state. `--full-transition-mode warmup` keeps
the model/global step but resets optimizer and scheduler so LR warm-up restarts.

With `drop_last=True`, the current sizes imply 859 optimizer steps per quarter
training epoch and 3,437 steps per full-data epoch. These values are computed
from the actual loader rather than hard-coded.

## Launch after the current legacy run is no longer using the server checkout

```bash
cd /home/20262202788/equivariant-nas
nohup bash scripts/supervise_quarter_multifidelity_cycles.sh \
  /home/20262202788/equivariant-nas/runs/quarter_multifidelity_seed201 \
  > /home/20262202788/equivariant-nas/runs/quarter_multifidelity_seed201/launcher.log 2>&1 &
```

Do not deploy these common trainer/pipeline files into the checkout used by the
currently running legacy experiment. Use a separate worktree or wait for that
run to finish.