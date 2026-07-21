# Continuous Equiformer NAS Run

Active run:

```text
/home/20262202788/equivariant-nas/runs/continuous_full_seed201
```

The run uses search seed 201, trainer seed 0, QM9 target 1, batch size 64,
5,000 optimizer steps per trained candidate, validation-only selection, and no
test evaluation. It runs until the `STOP` file is created.

## Inspect

```bash
RUN=/home/20262202788/equivariant-nas/runs/continuous_full_seed201
cat "$RUN/STATUS.md"
cat "$RUN/heartbeat.json"
cat "$RUN/current_candidate.json"
tail -f "$RUN/console.log"
```

Candidate training progress and the latest resumable checkpoint are stored
under:

```text
/home/20262202788/equivariant-nas/runs/candidates/<architecture_id>/seed0_steps5000_<protocol_id>/training/
```

Each candidate writes `progress.json` and `checkpoint_last.pth` every 859
optimizer steps. Complete proposals update `evolution.jsonl`, `router_state.json`,
the OpenEvolve `database/`, `summary.json`, `candidate_index.json`, and
`STATUS.md`.

## Stop

```bash
touch /home/20262202788/equivariant-nas/runs/continuous_full_seed201/STOP
```

The supervisor stops after the current process observes the marker. Removing
the marker and starting `scripts/run_continuous_search.sh` with the same run
directory resumes the database, lineage, router statistics, and partial
candidate checkpoint.
