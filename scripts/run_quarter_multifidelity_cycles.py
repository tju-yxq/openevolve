#!/usr/bin/env python
"""Run resumable 8 -> 4 -> 2 QM9 quarter/full multi-fidelity cycles."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/home/20262202788/equivariant-nas")
OPENEVOLVE_PYTHON = Path("/home/20262202788/conda-envs/openevolve/bin/python")
EQUIFORMER_PYTHON = Path("/home/20262202788/conda-envs/equiformer/bin/python")
BASELINE_PROGRAM = PROJECT / "openevolve_adapter/initial_program.py"
BASELINE_METRICS = None



def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_logged(command, log_path: Path, environment):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(now(), " ".join(map(str, command))))
        log.flush()
        completed = subprocess.run(
            list(map(str, command)),
            cwd=str(PROJECT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed with code {}: {}".format(
                completed.returncode, " ".join(map(str, command))
            )
        )


def candidate_checkpoint(metrics, steps):
    value = metrics.get("checkpoint_last")
    if value and Path(value).exists():
        return Path(value)
    architecture_id = metrics.get("architecture_id")
    if not architecture_id:
        return None
    root = PROJECT / "runs/candidates" / architecture_id
    matches = sorted(root.glob("seed0_steps{}_*/training/checkpoint_last.pth".format(steps)))
    return matches[-1] if matches else None


def collect_search_candidates(search_dir: Path, search_steps: int):
    records = read_jsonl(search_dir / "evolution.jsonl")
    candidates = []
    for record in records:
        metrics = record.get("metrics") or {}
        if not metrics.get("valid") or metrics.get("validation_alpha_mae") is None:
            continue
        measured_steps = int(
            metrics.get("endpoint_step", metrics.get("fidelity_steps", 0)) or 0
        )
        if measured_steps != int(search_steps):
            continue
        if metrics.get("test_evaluated") is not False:
            continue
        iteration = int(record.get("iteration", 0))
        if iteration <= 0:
            # The fixed baseline remains a control, but full-training slots are
            # reserved for architectures produced by the current search cycle.
            continue
        program = search_dir / "candidates" / "iteration_{:04d}.py".format(iteration)
        if not program.exists():
            continue
        checkpoint = candidate_checkpoint(metrics, search_steps)
        if checkpoint is None:
            continue
        candidates.append(
            {
                "iteration": iteration,
                "architecture_id": metrics["architecture_id"],
                "program": str(program),
                "metrics_8000": metrics,
                "checkpoint_8000": str(checkpoint),
            }
        )
    unique = {}
    for candidate in candidates:
        unique[candidate["architecture_id"]] = candidate
    return sorted(
        unique.values(), key=lambda item: item["metrics_8000"]["validation_alpha_mae"]
    )


def evaluate_promotion(
    candidate,
    steps,
    resume_checkpoint,
    environment,
    log_path,
    train_subset_file="",
    data_epoch_origin_step=0,
    allow_data_transition=False,
    resume_model_only=False,
    lr_schedule_origin_step=0,
    retry_partial=True,
):
    command = [
        EQUIFORMER_PYTHON,
        PROJECT / "scripts/run_pipeline.py",
        "--program",
        candidate["program"],
        "--project-root",
        PROJECT,
        "--equiformer-root",
        "/home/20262202788/equiformer",
        "--data-path",
        "/home/20262202788/equiformer/datasets/qm9",
        "--max-steps",
        str(steps),
        "--seed",
        "0",
        "--resume-checkpoint",
        resume_checkpoint,
        "--batch-size",
        "32",
        "--eval-interval-epochs",
        "10",
        "--data-epoch-origin-step",
        str(data_epoch_origin_step),
        "--lr-schedule-origin-step",
        str(lr_schedule_origin_step),
    ]
    if train_subset_file:
        command.extend(["--train-subset-file", str(train_subset_file)])
    if allow_data_transition:
        command.append("--allow-data-transition")
    if resume_model_only:
        command.append("--resume-model-only")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(now(), " ".join(map(str, command))))
        log.flush()
        completed = subprocess.run(
            list(map(str, command)),
            cwd=str(PROJECT),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=log,
            check=False,
        )
    if completed.returncode != 0:
        result = {
            "valid": False,
            "error": "promotion command exited {}".format(completed.returncode),
        }
    else:
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            result = {"valid": False, "error": "promotion returned non-JSON output"}
    if (
        retry_partial
        and not result.get("valid")
        and result.get("checkpoint_last")
    ):
        # Retry the identical architecture once; pipeline v7 resumes from the
        # partial checkpoint rather than restarting the candidate.
        return evaluate_promotion(
            candidate,
            steps,
            result["checkpoint_last"],
            environment,
            log_path,
            train_subset_file=train_subset_file,
            data_epoch_origin_step=data_epoch_origin_step,
            allow_data_transition=allow_data_transition,
            resume_model_only=False,
            lr_schedule_origin_step=lr_schedule_origin_step,
            retry_partial=False,
        )
    return result


def update_status(root: Path, state):
    lines = [
        "# Quarter-to-Full Multi-Fidelity Equiformer Evolution",
        "",
        "Updated: `{}`".format(now()),
        "",
        "- Cycle: `{}`".format(state.get("cycle")),
        "- Stage: `{}`".format(state.get("stage")),
        "- Test split used: `false`",
        "- Stop file: `{}`".format(root / "STOP"),
    ]
    current = state.get("current") or {}
    if current:
        lines.extend(
            [
                "- Current architecture: `{}`".format(
                    current.get("architecture_id")
                ),
                "- Current fidelity: `{}` steps".format(current.get("steps")),
            ]
        )
    if state.get("candidates_8000"):
        lines.append(
            "- Completed 8k cohort entries: `{}`".format(
                len(state["candidates_8000"])
            )
        )
    if state.get("candidates_80000"):
        lines.append(
            "- Completed 80k promotions: `{}`".format(
                len(state["candidates_80000"])
            )
        )
    if state.get("candidates_250000"):
        lines.append(
            "- Completed full trainings: `{}`".format(
                len(state["candidates_250000"])
            )
        )
    winner = state.get("last_full_winner")
    if winner:
        lines.extend(
            [
                "",
                "## Last Full-Fidelity Winner",
                "",
                "- Architecture: `{}`".format(winner["architecture_id"]),
                "- Validation MAE: `{:.9f} a0^3`".format(
                    winner["metrics_250000"]["validation_alpha_mae"]
                ),
                "- Checkpoint: `{}`".format(winner["checkpoint_250000"]),
            ]
        )
    (root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cycle-one-search", default="")
    parser.add_argument("--search-valid-target", type=int, default=8)
    parser.add_argument("--search-max-proposals", type=int, default=30)
    parser.add_argument("--promote-80000", type=int, default=4)
    parser.add_argument("--promote-full", type=int, default=2)
    parser.add_argument("--base-search-seed", type=int, default=201)
    parser.add_argument(
        "--quarter-subset-file",
        default=str(PROJECT / "data_splits/qm9_train_quarter_seed201.npz"),
    )
    parser.add_argument(
        "--full-transition-mode",
        choices=("continue", "warmup"),
        default="continue",
        help="Continue optimizer state, or load model only and restart LR warm-up at 80k.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    quarter_subset_file = Path(args.quarter_subset_file).resolve()
    if not quarter_subset_file.is_file():
        raise FileNotFoundError(quarter_subset_file)
    subset_sha256 = hashlib.sha256(quarter_subset_file.read_bytes()).hexdigest()
    protocol = {
        "batch_size": 32,
        "search_candidates": int(args.search_valid_target),
        "promotion_candidates_80000": int(args.promote_80000),
        "promotion_candidates_250000": int(args.promote_full),
        "stage_steps": [8000, 80000, 250000],
        "validation_interval_data_epochs": 10,
        "quarter_subset_file": str(quarter_subset_file),
        "quarter_subset_sha256": subset_sha256,
        "quarter_subset_size": 27500,
        "full_transition_mode": args.full_transition_mode,
        "test_during_search": False,
    }
    stop_file = root / "STOP"
    state_path = root / "state.json"
    state = read_json(
        state_path,
        {
            "cycle": 1,
            "stage": "search",
            "seed_program": str(BASELINE_PROGRAM),
            "seed_metrics": "",
            "protocol": protocol,
            "created_at": now(),
        },
    )
    if state.get("protocol") != protocol:
        raise ValueError(
            "existing run protocol differs from the requested protocol; "
            "resume with the original arguments instead of mutating the run"
        )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    environment["EQUIVARIANT_NAS_ROOT"] = str(PROJECT)
    environment["NAS_GPU_BUDGET_HOURS"] = "1000000"
    environment["NAS_BATCH_SIZE"] = "32"
    environment["NAS_EVAL_INTERVAL_EPOCHS"] = "10"
    environment["NAS_TRAIN_SUBSET_FILE"] = str(quarter_subset_file)
    environment["NAS_DATA_EPOCH_ORIGIN_STEP"] = "0"
    environment["NAS_LR_SCHEDULE_ORIGIN_STEP"] = "0"
    environment.update(
        {
            "http_proxy": "http://127.0.0.1:12356",
            "https_proxy": "http://127.0.0.1:12356",
            "HTTP_PROXY": "http://127.0.0.1:12356",
            "HTTPS_PROXY": "http://127.0.0.1:12356",
        }
    )

    while not stop_file.exists():
        cycle = int(state["cycle"])
        cycle_dir = root / "cycle_{:04d}".format(cycle)
        search_dir = (
            Path(args.cycle_one_search)
            if cycle == 1 and args.cycle_one_search
            else cycle_dir / "search"
        )
        cycle_dir.mkdir(parents=True, exist_ok=True)
        environment["NAS_BUDGET_LEDGER"] = str(root / "budget_ledger.jsonl")

        if state["stage"] == "search":
            search_stop = search_dir / "STOP"
            if search_stop.exists():
                search_stop.unlink()
            state["search_dir"] = str(search_dir)
            state["updated_at"] = now()
            write_json(state_path, state)
            update_status(root, state)
            command = [
                OPENEVOLVE_PYTHON,
                PROJECT / "scripts/run_factorized_evolution.py",
                "--initial-program",
                state["seed_program"],
                "--evaluator-file",
                PROJECT / "openevolve_adapter/evaluator.py",
                "--config",
                "/home/20262202788/openevolve/configs/local_glm_5_2.yaml",
                "--output",
                search_dir,
                "--max-steps",
                "8000",
                "--seed",
                str(args.base_search_seed + cycle - 1),
                "--router-mode",
                "evidence",
                "--router-prior",
                PROJECT / "configs/stage1_factor_memory.json",
                "--repair-attempts",
                "1",
                "--valid-target",
                str(args.search_valid_target),
                "--max-proposals",
                str(args.search_max_proposals),
                "--resume",
            ]
            if state.get("seed_metrics"):
                command.extend(["--initial-metrics", state["seed_metrics"]])
            run_logged(command, cycle_dir / "search.log", environment)
            candidates = collect_search_candidates(search_dir, 8000)
            if len(candidates) < args.search_valid_target:
                raise RuntimeError(
                    "cycle {} produced only {} resumable 8k candidates".format(
                        cycle, len(candidates)
                    )
                )
            state["candidates_8000"] = candidates
            state["stage"] = "promote_80000"
            state["updated_at"] = now()
            write_json(state_path, state)
            update_status(root, state)

        if state["stage"] == "promote_80000":
            promoted = state.get("candidates_80000", [])
            attempted = {item["architecture_id"] for item in state.get("attempted_80000", [])}
            for candidate in state["candidates_8000"]:
                if len(promoted) >= args.promote_80000 or stop_file.exists():
                    break
                if candidate["architecture_id"] in attempted:
                    continue
                state["current"] = {
                    "architecture_id": candidate["architecture_id"],
                    "steps": 80000,
                }
                write_json(state_path, state)
                update_status(root, state)
                metrics = evaluate_promotion(
                    candidate,
                    80000,
                    candidate["checkpoint_8000"],
                    environment,
                    cycle_dir / "promote_80000.log",
                    train_subset_file=quarter_subset_file,
                    data_epoch_origin_step=0,
                )
                attempt = dict(candidate, metrics_80000=metrics)
                state.setdefault("attempted_80000", []).append(attempt)
                if metrics.get("valid") and metrics.get("validation_alpha_mae") is not None:
                    checkpoint = candidate_checkpoint(metrics, 80000)
                    if checkpoint:
                        attempt["checkpoint_80000"] = str(checkpoint)
                        promoted.append(attempt)
                        state["candidates_80000"] = promoted
                write_json(state_path, state)
            if len(promoted) < args.promote_80000 and not stop_file.exists():
                raise RuntimeError("fewer than requested candidates passed 80k promotion")
            promoted.sort(key=lambda item: item["metrics_80000"]["validation_alpha_mae"])
            state["candidates_80000"] = promoted
            state["stage"] = "promote_250000"
            state.pop("current", None)
            write_json(state_path, state)
            update_status(root, state)

        if state["stage"] == "promote_250000":
            promoted = state.get("candidates_250000", [])
            attempted = {item["architecture_id"] for item in state.get("attempted_250000", [])}
            for candidate in state["candidates_80000"]:
                if len(promoted) >= args.promote_full or stop_file.exists():
                    break
                if candidate["architecture_id"] in attempted:
                    continue
                state["current"] = {
                    "architecture_id": candidate["architecture_id"],
                    "steps": 250000,
                }
                write_json(state_path, state)
                update_status(root, state)
                metrics = evaluate_promotion(
                    candidate,
                    250000,
                    candidate["checkpoint_80000"],
                    environment,
                    cycle_dir / "promote_250000.log",
                    train_subset_file="",
                    data_epoch_origin_step=80000,
                    allow_data_transition=True,
                    resume_model_only=args.full_transition_mode == "warmup",
                    lr_schedule_origin_step=(
                        80000 if args.full_transition_mode == "warmup" else 0
                    ),
                )
                attempt = dict(candidate, metrics_250000=metrics)
                state.setdefault("attempted_250000", []).append(attempt)
                if metrics.get("valid") and metrics.get("validation_alpha_mae") is not None:
                    checkpoint = candidate_checkpoint(metrics, 250000)
                    if checkpoint:
                        attempt["checkpoint_250000"] = str(checkpoint)
                        promoted.append(attempt)
                        state["candidates_250000"] = promoted
                write_json(state_path, state)
            if len(promoted) < args.promote_full and not stop_file.exists():
                raise RuntimeError("fewer than requested candidates completed full training")
            promoted.sort(key=lambda item: item["metrics_250000"]["validation_alpha_mae"])
            winner = promoted[0]
            append_jsonl(
                root / "full_fidelity_archive.jsonl",
                dict(winner, cycle=cycle, selected_at=now()),
            )
            seed_dir = root / "seeds"
            seed_dir.mkdir(exist_ok=True)
            next_program = seed_dir / "cycle_{:04d}_winner.py".format(cycle)
            shutil.copyfile(winner["program"], next_program)
            next_metrics = seed_dir / "cycle_{:04d}_winner_8000_metrics.json".format(cycle)
            write_json(next_metrics, winner["metrics_8000"])
            state["last_full_winner"] = winner
            state["cycle"] = cycle + 1
            state["stage"] = "search"
            state["seed_program"] = str(next_program)
            state["seed_metrics"] = str(next_metrics)
            for key in (
                "candidates_8000",
                "candidates_80000",
                "candidates_250000",
                "attempted_80000",
                "attempted_250000",
                "current",
            ):
                state.pop(key, None)
            state["updated_at"] = now()
            write_json(state_path, state)
            update_status(root, state)

    state["status"] = "stopped"
    state["updated_at"] = now()
    write_json(state_path, state)
    update_status(root, state)


if __name__ == "__main__":
    main()
