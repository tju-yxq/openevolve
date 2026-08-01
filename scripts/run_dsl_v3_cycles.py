#!/usr/bin/env python
"""Run iterative V3 evolution cycles whose 250k winner becomes the next fixed parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_dsl_v3_cohort import load_frozen_protocol


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_logged(command, log_path, environment):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(_now(), " ".join(command)))
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError("cycle subprocess exited {}: {}".format(completed.returncode, " ".join(command)))


def run(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = load_frozen_protocol(Path(args.protocol_config))
    quarter_subset_file = Path(args.quarter_subset_file).resolve()
    if not quarter_subset_file.is_file():
        raise FileNotFoundError(quarter_subset_file)
    if _sha256(quarter_subset_file) != protocol.quarter_subset_sha256:
        raise RuntimeError("requested quarter subset differs from the frozen protocol")
    args.quarter_subset_file = str(quarter_subset_file)
    requested_cycles = args.cycles or protocol.cycle_count
    if requested_cycles != protocol.cycle_count:
        raise ValueError("cycle count is frozen by the protocol and cannot change during the run")
    if args.mutation_mode == "structural" and args.generation_validation_level != "full":
        raise ValueError(
            "formal structural evolution requires full numerical equivariance auditing before training"
        )
    manifest_path = root / "cycles_manifest.json"
    manifest = {
        "kind": "iterative_v3_best_parent_cycles",
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.content_hash(),
        "cycle_count": requested_cycles,
        "initial_model_config": str(Path(args.model_config).resolve()),
        "selection_mode": args.selection_mode,
        "mutation_mode": args.mutation_mode,
        "generation_validation_level": args.generation_validation_level,
        "candidate_replacement_attempts": args.candidate_replacement_attempts,
        "llm_model": args.model if args.selection_mode == "glm" else "",
        "equiformer_root": str(Path(args.equiformer_root).resolve()),
        "equiformer_v3_root": str(Path(args.equiformer_v3_root).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "quarter_subset_file": str(Path(args.quarter_subset_file).resolve()),
        "batch_size": protocol.batch_size,
        "workflow": [
            "generate_8_unique_direct_children_from_fixed_parent",
            "train_all_8_to_8000_on_fixed_quarter",
            "promote_validation_top4_to_80000_on_same_quarter",
            "promote_validation_top2_to_250000_on_full_train",
            "set_validation_winner_as_next_cycle_parent",
        ],
        "test_during_search": False,
        "created_at": _now(),
    }
    if manifest_path.exists():
        existing = _read_json(manifest_path)
        frozen_keys = tuple(key for key in manifest if key != "created_at")
        changed = {key: (existing.get(key), manifest.get(key)) for key in frozen_keys if existing.get(key) != manifest.get(key)}
        if changed:
            raise RuntimeError("resume changed the frozen ten-cycle identity: {}".format(changed))
        manifest = existing
    _write_json(manifest_path, manifest)

    state_path = root / "state.json"
    state = _read_json(state_path, {
        "protocol_hash": protocol.content_hash(),
        "completed_cycles": [],
        "architecture_archive": [],
        "next_cycle": 1,
        "created_at": _now(),
    })
    if state.get("protocol_hash") != protocol.content_hash():
        raise RuntimeError("cycle state protocol hash changed")
    state.setdefault("architecture_archive", [])
    stop = root / "STOP"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    parent_program = ""
    completed_cycles = state.get("completed_cycles", [])
    if completed_cycles:
        parent_program = str(completed_cycles[-1]["next_cycle_parent"]["program"])

    for cycle_index in range(int(state.get("next_cycle", 1)), requested_cycles + 1):
        if stop.exists():
            break
        cycle_root = root / "cycle_{:03d}".format(cycle_index)
        cohort_dir = cycle_root / "cohort"
        training_dir = cycle_root / "multifidelity"
        archive_snapshot = cycle_root / "architecture_archive_input.json"
        if not archive_snapshot.exists():
            _write_json(
                archive_snapshot,
                {"architecture_ids": sorted(set(state.get("architecture_archive", ())))},
            )
        cohort_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_dsl_v3_cohort.py"),
            "--protocol-config",
            str(Path(args.protocol_config).resolve()),
            "--output",
            str(cohort_dir),
            "--cycle-index",
            str(cycle_index),
            "--cohort-size",
            str(protocol.cohort_size),
            "--selection-mode",
            args.selection_mode,
            "--mutation-mode",
            args.mutation_mode,
            "--architecture-archive",
            str(archive_snapshot),
            "--model",
            args.model,
            "--validation-level",
            args.generation_validation_level,
            "--candidate-replacement-attempts",
            str(args.candidate_replacement_attempts),
            "--equiformer-v3-root",
            args.equiformer_v3_root,
            "--equiformer-root",
            args.equiformer_root,
            "--data-path",
            args.data_path,
            "--train-subset-file",
            args.quarter_subset_file,
        ]
        if parent_program:
            cohort_command.extend(["--seed-program", parent_program])
        else:
            cohort_command.extend(["--model-config", str(Path(args.model_config).resolve())])
        if cohort_dir.exists() and any(cohort_dir.iterdir()):
            cohort_command.append("--resume")
        state["stage"] = "cycle_{:03d}_generate_cohort".format(cycle_index)
        state["next_cycle"] = cycle_index
        _write_json(state_path, state)
        _run_logged(cohort_command, cycle_root / "generation.log", environment)
        cohort_manifest = _read_json(cohort_dir / "cohort_manifest.json")
        if not cohort_manifest or cohort_manifest.get("protocol_hash") != protocol.content_hash():
            raise RuntimeError("cycle {} cohort lacks the frozen protocol identity".format(cycle_index))
        cohort_parent_id = str(cohort_manifest.get("fixed_parent_architecture_id", ""))
        if completed_cycles:
            expected_parent_id = str(completed_cycles[-1]["winner_architecture_id"])
            if cohort_parent_id != expected_parent_id:
                raise RuntimeError(
                    "cycle {} parent {} is not the previous 250k winner {}".format(
                        cycle_index, cohort_parent_id, expected_parent_id
                    )
                )
        cohort_records_path = cohort_dir / "cohort.jsonl"
        cohort_records = [
            json.loads(line)
            for line in cohort_records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(cohort_records) != protocol.cohort_size:
            raise RuntimeError("cycle {} did not generate eight archived structural candidates".format(cycle_index))
        generated_ids = {str(item.get("architecture_id", "")) for item in cohort_records}
        if "" in generated_ids or len(generated_ids) != protocol.cohort_size:
            raise RuntimeError("cycle {} cohort architecture identities are incomplete or duplicated".format(cycle_index))
        snapshot_payload = _read_json(archive_snapshot, {"architecture_ids": []})
        forbidden_at_generation = set(snapshot_payload.get("architecture_ids", ()))
        if generated_ids & forbidden_at_generation:
            raise RuntimeError("cycle {} regenerated an architecture from the global archive".format(cycle_index))
        previous_archive = set(state.get("architecture_archive", ()))
        rejected_records = []
        rejected_path = cycle_root / "cohort" / "rejected_candidates.jsonl"
        if rejected_path.exists():
            rejected_records = [
                json.loads(line)
                for line in rejected_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        rejected_ids = {
            str(item.get("architecture_id", ""))
            for item in rejected_records
            if str(item.get("architecture_id", ""))
        }
        state["architecture_archive"] = sorted(
            previous_archive | generated_ids | rejected_ids | {cohort_parent_id}
        )
        _write_json(state_path, state)

        training_command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/run_dsl_v3_multifidelity.py"),
            "--root",
            str(training_dir),
            "--cohort-dir",
            str(cohort_dir),
            "--protocol-config",
            str(Path(args.protocol_config).resolve()),
            "--equiformer-root",
            args.equiformer_root,
            "--equiformer-v3-root",
            args.equiformer_v3_root,
            "--data-path",
            args.data_path,
            "--quarter-subset-file",
            args.quarter_subset_file,
            "--python",
            args.training_python,
            "--gpu-budget-hours",
            str(args.gpu_budget_hours),
            "--eval-interval-epochs",
            str(args.eval_interval_epochs),
        ]
        if args.task_contract:
            training_command.extend(["--task-contract", args.task_contract])
        state["stage"] = "cycle_{:03d}_multifidelity".format(cycle_index)
        _write_json(state_path, state)
        _run_logged(training_command, cycle_root / "multifidelity.log", environment)
        training_state = _read_json(training_dir / "state.json")
        if not training_state or training_state.get("stage") != "cycle_complete":
            raise RuntimeError("V3 cycle {} did not select a 250k validation winner".format(cycle_index))
        if str(training_state.get("parent_architecture_id", "")) != cohort_parent_id:
            raise RuntimeError("V3 training state disagrees with the generated cohort parent")
        next_parent = training_state["next_cycle_parent"]
        if str(next_parent.get("architecture_id", "")) not in generated_ids:
            raise RuntimeError("V3 cycle winner is not one of the eight generated structural candidates")
        if str(next_parent["architecture_id"]) == str(training_state["parent_architecture_id"]):
            raise RuntimeError("V3 cycle winner unexpectedly equals its fixed parent architecture")
        if int(next_parent.get("endpoint_step", 0)) != protocol.stages[-1].endpoint_steps:
            raise RuntimeError("next-cycle parent was not selected at the frozen 250k endpoint")
        if str(next_parent.get("protocol_hash", "")) != protocol.content_hash():
            raise RuntimeError("next-cycle parent changed the frozen protocol identity")
        if not Path(next_parent.get("program", "")).is_file():
            raise RuntimeError("next-cycle parent DSL program is missing")
        completed_entry = {
            "cycle_index": cycle_index,
            "parent_architecture_id": training_state["parent_architecture_id"],
            "winner_architecture_id": next_parent["architecture_id"],
            "next_cycle_parent": next_parent,
            "cohort_dir": str(cohort_dir),
            "multifidelity_dir": str(training_dir),
            "completed_at": _now(),
        }
        completed_cycles.append(completed_entry)
        state["completed_cycles"] = completed_cycles
        state["next_cycle"] = cycle_index + 1
        state["stage"] = "cycle_{:03d}_complete".format(cycle_index)
        _write_json(state_path, state)
        parent_program = str(next_parent["program"])

    if len(completed_cycles) == requested_cycles:
        state["stage"] = "all_cycles_complete"
        state["final_parent"] = completed_cycles[-1]["next_cycle_parent"]
        state["completed_at"] = _now()
        _write_json(state_path, state)
    print(json.dumps({
        "root": str(root),
        "stage": state["stage"],
        "completed_cycle_count": len(completed_cycles),
        "cycle_target": requested_cycles,
        "final_parent": state.get("final_parent"),
        "test_evaluated": False,
    }, ensure_ascii=False, sort_keys=True))
    return state


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--equiformer-v3-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--quarter-subset-file", required=True)
    parser.add_argument("--task-contract", default="")
    parser.add_argument("--training-python", default=os.environ.get("EQUIFORMER_PYTHON", sys.executable))
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--selection-mode", choices=("glm", "deterministic"), default="glm")
    parser.add_argument("--mutation-mode", choices=("structural", "probability"), default="structural")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--generation-validation-level", choices=("static", "build", "full"), default="full")
    parser.add_argument("--candidate-replacement-attempts", type=int, default=16)
    parser.add_argument("--gpu-budget-hours", type=float, default=float(os.environ.get("NAS_GPU_BUDGET_HOURS", "80")))
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    return parser


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
