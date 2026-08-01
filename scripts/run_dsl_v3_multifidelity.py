#!/usr/bin/env python
"""Train one fixed-parent V3 cohort through 8k -> 80k -> 250k and select its winner."""

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

from equivariant_nas.dsl import rank_v3_stage_records
from scripts.run_dsl_v3_cohort import load_frozen_protocol


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _candidate_from_generation(record):
    return {
        "candidate_index": int(record["candidate_index"]),
        "architecture_id": str(record["architecture_id"]),
        "program": str(Path(record["candidate_path"]).resolve()),
        "factor_id": str(record.get("factor_id", "")),
        "region_id": str(record.get("region_id", "")),
        "action_id": str(record.get("action_id", "")),
    }


def _pipeline_command(args, candidate, *, stage, checkpoint, protocol):
    command = [
        args.python,
        str(PROJECT_ROOT / "scripts/run_pipeline.py"),
        "--program",
        candidate["program"],
        "--project-root",
        str(PROJECT_ROOT),
        "--equiformer-root",
        args.equiformer_root,
        "--equiformer-v3-root",
        args.equiformer_v3_root,
        "--data-path",
        args.data_path,
        "--max-steps",
        str(stage.endpoint_steps),
        "--seed",
        str(protocol.seed),
        "--batch-size",
        str(protocol.batch_size),
        "--eval-interval-epochs",
        str(args.eval_interval_epochs),
    ]
    if args.task_contract:
        command.extend(["--dsl-task-contract", args.task_contract])
    if checkpoint:
        command.extend(["--resume-checkpoint", checkpoint])
    if stage.training_data == "fixed_quarter":
        command.extend(["--train-subset-file", args.quarter_subset_file])
    if stage.optimizer_transition == "model_only_optimizer_restart":
        command.extend(
            [
                "--allow-data-transition",
                "--resume-model-only",
                "--data-epoch-origin-step",
                "80000",
                "--lr-schedule-origin-step",
                "80000",
            ]
        )
    return command


def _run_pipeline(args, candidate, *, stage, checkpoint, protocol, log_path):
    command = _pipeline_command(args, candidate, stage=stage, checkpoint=checkpoint, protocol=protocol)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    environment["EQUIFORMER_PYTHON"] = args.python
    environment["EQUIFORMER_V3_ROOT"] = args.equiformer_v3_root
    environment["NAS_GPU_BUDGET_HOURS"] = str(args.gpu_budget_hours)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n[{}] {}\n".format(_now(), " ".join(command)))
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=log,
            env=environment,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError("V3 pipeline exited {} for {}".format(completed.returncode, candidate["architecture_id"]))
    result = json.loads(completed.stdout)
    if result.get("valid") is not True:
        raise RuntimeError("V3 pipeline rejected candidate: {}".format(result.get("error", "unknown")))
    if result.get("test_evaluated") is not False:
        raise RuntimeError("test leakage detected during V3 evolution")
    if str(result.get("architecture_id", "")) != candidate["architecture_id"]:
        raise RuntimeError("V3 training changed the candidate architecture identity")
    if int(result.get("endpoint_step", 0)) != stage.endpoint_steps:
        raise RuntimeError("V3 training did not reach the frozen stage endpoint")
    if int(result.get("batch_size", 0)) != protocol.batch_size:
        raise RuntimeError("V3 training changed the frozen batch size")
    if result.get("selection_eligible") is not True:
        raise RuntimeError("V3 candidate was not admitted for formal ranking")
    expected_start = {
        "quarter_8k": 0,
        "quarter_80k": 8000,
        "full_250k": 80000,
    }[stage.name]
    if int(result.get("start_global_step", -1)) != expected_start:
        raise RuntimeError(
            "V3 training resumed from step {}, expected {} for {}".format(
                result.get("start_global_step"), expected_start, stage.name
            )
        )
    if int(result.get("steps_executed_current_job", -1)) != stage.endpoint_steps - expected_start:
        raise RuntimeError("V3 training executed the wrong number of steps for {}".format(stage.name))
    if checkpoint and str(Path(result.get("resumed_from", "")).resolve()) != str(Path(checkpoint).resolve()):
        raise RuntimeError("V3 training did not resume from the promoted checkpoint")
    if not checkpoint and result.get("resumed_from"):
        raise RuntimeError("V3 8k stage unexpectedly resumed an earlier checkpoint")
    return {
        **candidate,
        "protocol_hash": protocol.content_hash(),
        "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
        "quarter_subset_sha256": protocol.quarter_subset_sha256,
        "equivariance_contract_sha256": protocol.equivariance_contract_sha256,
        "endpoint_step": stage.endpoint_steps,
        "validation_alpha_mae": float(result["validation_alpha_mae"]),
        "test_evaluated": False,
        "checkpoint": str(Path(result["checkpoint_last"]).resolve()),
        "pipeline_result": result,
        "runtime_manifest_sha256": str(result.get("runtime_manifest_sha256", "")),
        "executable_id": str(result.get("executable_id", "")),
        "lowering_plan_hash": str(result.get("lowering_plan_hash", "")),
        "completed_at": _now(),
    }


def _update_status(root, state):
    lines = [
        "# V3 Typed DSL 8→4→2 多保真周期",
        "",
        "- 更新时间：`{}`".format(_now()),
        "- 周期：`{}`".format(state.get("cycle_index")),
        "- 阶段：`{}`".format(state.get("stage")),
        "- 固定父架构：`{}`".format(state.get("parent_architecture_id")),
        "- batch size：`8`",
        "- 8k完成：`{}/8`".format(len(state.get("quarter_8k", []))),
        "- 80k完成：`{}/4`".format(len(state.get("quarter_80k", []))),
        "- 250k完成：`{}/2`".format(len(state.get("full_250k", []))),
        "- Test参与搜索：`false`",
    ]
    if state.get("current"):
        lines.append("- 当前训练：`{}`".format(json.dumps(state["current"], ensure_ascii=False, sort_keys=True)))
    (Path(root) / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    root = Path(args.root).resolve()
    cohort_dir = Path(args.cohort_dir).resolve()
    protocol = load_frozen_protocol(Path(args.protocol_config))
    quarter_subset_file = Path(args.quarter_subset_file).resolve()
    if not quarter_subset_file.is_file():
        raise FileNotFoundError(quarter_subset_file)
    if _sha256(quarter_subset_file) != protocol.quarter_subset_sha256:
        raise RuntimeError("requested quarter subset differs from the frozen protocol")
    args.quarter_subset_file = str(quarter_subset_file)
    cohort_manifest = _read_json(cohort_dir / "cohort_manifest.json")
    if not cohort_manifest or cohort_manifest.get("protocol_hash") != protocol.content_hash():
        raise RuntimeError("cohort was not generated under the requested frozen protocol")
    generation = _read_jsonl(cohort_dir / "cohort.jsonl")
    if len(generation) != protocol.cohort_size:
        raise RuntimeError("V3 multi-fidelity cycle requires exactly eight generated candidates")
    candidates = [_candidate_from_generation(item) for item in generation]
    if len({item["architecture_id"] for item in candidates}) != protocol.cohort_size:
        raise RuntimeError("V3 cohort contains duplicate candidates")
    parent_id = str(cohort_manifest["fixed_parent_architecture_id"])
    if any(str(item["parent_architecture_id"]) != parent_id for item in generation):
        raise RuntimeError("V3 cohort candidates do not share one frozen parent")

    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    initial = {
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.content_hash(),
        "cycle_index": int(cohort_manifest["cycle_index"]),
        "parent_architecture_id": parent_id,
        "cohort_dir": str(cohort_dir),
        "stage": "quarter_8k",
        "created_at": _now(),
    }
    state = _read_json(state_path, initial)
    frozen_keys = ("protocol_hash", "cycle_index", "parent_architecture_id", "cohort_dir")
    changed = {key: (state.get(key), initial.get(key)) for key in frozen_keys if state.get(key) != initial.get(key)}
    if changed:
        raise RuntimeError("resume changed the frozen V3 cycle identity: {}".format(changed))
    stop = root / "STOP"

    stage8, stage80, stage250 = protocol.stages
    if state["stage"] == "quarter_8k" and not stop.exists():
        completed = state.get("quarter_8k", [])
        done = {item["architecture_id"] for item in completed}
        for candidate in candidates:
            if candidate["architecture_id"] in done or stop.exists():
                continue
            state["current"] = {"stage": stage8.name, "architecture_id": candidate["architecture_id"]}
            _write_json(state_path, state)
            _update_status(root, state)
            result = _run_pipeline(
                args,
                candidate,
                stage=stage8,
                checkpoint="",
                protocol=protocol,
                log_path=root / "logs" / "{}_{}.log".format(stage8.name, candidate["architecture_id"]),
            )
            completed.append(result)
            state["quarter_8k"] = completed
            _write_json(state_path, state)
        if len(completed) == stage8.candidate_count:
            state["promoted_to_80k"] = list(rank_v3_stage_records(completed, stage=stage8, protocol=protocol))
            state["stage"] = "quarter_80k"
            state.pop("current", None)
            _write_json(state_path, state)
            _update_status(root, state)

    if state["stage"] == "quarter_80k" and not stop.exists():
        completed = state.get("quarter_80k", [])
        done = {item["architecture_id"] for item in completed}
        for candidate in state["promoted_to_80k"]:
            if candidate["architecture_id"] in done or stop.exists():
                continue
            state["current"] = {"stage": stage80.name, "architecture_id": candidate["architecture_id"]}
            _write_json(state_path, state)
            _update_status(root, state)
            result = _run_pipeline(
                args,
                candidate,
                stage=stage80,
                checkpoint=candidate["checkpoint"],
                protocol=protocol,
                log_path=root / "logs" / "{}_{}.log".format(stage80.name, candidate["architecture_id"]),
            )
            completed.append(result)
            state["quarter_80k"] = completed
            _write_json(state_path, state)
        if len(completed) == stage80.candidate_count:
            state["promoted_to_250k"] = list(rank_v3_stage_records(completed, stage=stage80, protocol=protocol))
            state["stage"] = "full_250k"
            state.pop("current", None)
            _write_json(state_path, state)
            _update_status(root, state)

    if state["stage"] == "full_250k" and not stop.exists():
        completed = state.get("full_250k", [])
        done = {item["architecture_id"] for item in completed}
        for candidate in state["promoted_to_250k"]:
            if candidate["architecture_id"] in done or stop.exists():
                continue
            state["current"] = {"stage": stage250.name, "architecture_id": candidate["architecture_id"]}
            _write_json(state_path, state)
            _update_status(root, state)
            result = _run_pipeline(
                args,
                candidate,
                stage=stage250,
                checkpoint=candidate["checkpoint"],
                protocol=protocol,
                log_path=root / "logs" / "{}_{}.log".format(stage250.name, candidate["architecture_id"]),
            )
            completed.append(result)
            state["full_250k"] = completed
            _write_json(state_path, state)
        if len(completed) == stage250.candidate_count:
            winner = rank_v3_stage_records(completed, stage=stage250, protocol=protocol)[0]
            state["winner"] = winner
            state["next_cycle_parent"] = {
                "architecture_id": winner["architecture_id"],
                "program": winner["program"],
                "checkpoint": winner["checkpoint"],
                "selected_by": protocol.selection_metric,
                "selection_split": "validation",
                "endpoint_step": stage250.endpoint_steps,
                "validation_alpha_mae": winner[protocol.selection_metric],
                "protocol_hash": protocol.content_hash(),
                "runtime_manifest_sha256": winner["runtime_manifest_sha256"],
                "executable_id": winner["executable_id"],
                "lowering_plan_hash": winner["lowering_plan_hash"],
            }
            state["stage"] = "cycle_complete"
            state.pop("current", None)
            state["completed_at"] = _now()
            _write_json(state_path, state)
            _write_json(root / "next_cycle_parent.json", state["next_cycle_parent"])
            _update_status(root, state)

    print(json.dumps({
        "root": str(root),
        "cycle_index": state["cycle_index"],
        "stage": state["stage"],
        "parent_architecture_id": parent_id,
        "next_cycle_parent": state.get("next_cycle_parent"),
        "test_evaluated": False,
    }, ensure_ascii=False, sort_keys=True))
    return state


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cohort-dir", required=True)
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--equiformer-v3-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--quarter-subset-file", required=True)
    parser.add_argument("--task-contract", default="")
    parser.add_argument("--python", default=os.environ.get("EQUIFORMER_PYTHON", sys.executable))
    parser.add_argument("--gpu-budget-hours", type=float, default=float(os.environ.get("NAS_GPU_BUDGET_HOURS", "80")))
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    return parser


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
