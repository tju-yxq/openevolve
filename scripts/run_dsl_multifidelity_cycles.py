#!/usr/bin/env python
"""Resumable formal-V1 8k -> 80k -> 250k controller for DSL candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def collect_8k_candidates(search_dir, expected):
    search_dir = Path(search_dir)
    candidates = []
    for record in read_jsonl(search_dir / "evolution.jsonl"):
        metrics = record.get("metrics") or {}
        if int(record.get("iteration", 0)) <= 0:
            continue
        if not metrics.get("valid") or metrics.get("test_evaluated") is not False:
            continue
        if int(metrics.get("endpoint_step", metrics.get("fidelity_steps", 0))) != 8000:
            continue
        architecture_id = str(metrics.get("architecture_id", record.get("architecture_id", "")))
        matches = sorted((search_dir / "candidates").glob("iteration_{:04d}_{}.dsl.json".format(int(record["iteration"]), architecture_id)))
        checkpoint = Path(str(metrics.get("checkpoint_last", "")))
        if not matches or not checkpoint.is_file():
            continue
        candidates.append({
            "iteration": int(record["iteration"]),
            "architecture_id": architecture_id,
            "program_id": metrics.get("program_id", architecture_id),
            "executable_id": metrics.get("executable_id", ""),
            "protocol_id_8000": metrics.get("protocol_id", ""),
            "program": str(matches[0].resolve()),
            "checkpoint_8000": str(checkpoint.resolve()),
            "metrics_8000": metrics,
            "factor_id": (record.get("region_audit") or {}).get("factor_id", ""),
        })
    unique = {item["architecture_id"]: item for item in candidates}
    ordered = sorted(unique.values(), key=lambda item: float(item["metrics_8000"]["validation_alpha_mae"]))
    if len(ordered) < expected:
        raise RuntimeError("search produced only {} resumable valid 8k DSL candidates; expected {}".format(len(ordered), expected))
    return ordered[:expected]


def run_pipeline(args, candidate, *, max_steps, checkpoint, subset_file, transition=False):
    command = [
        args.python,
        str(Path(args.project_root) / "scripts/run_pipeline.py"),
        "--program", candidate["program"],
        "--project-root", args.project_root,
        "--equiformer-root", args.equiformer_root,
        "--data-path", args.data_path,
        "--max-steps", str(max_steps),
        "--seed", str(args.seed),
        "--resume-checkpoint", checkpoint,
        "--batch-size", "32",
        "--eval-interval-epochs", "10",
        "--dsl-task-contract", args.task_contract,
    ]
    if subset_file:
        command.extend(["--train-subset-file", subset_file])
    if transition:
        command.extend(["--allow-data-transition", "--resume-model-only", "--data-epoch-origin-step", "80000", "--lr-schedule-origin-step", "80000"])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = args.project_root
    environment["EQUIFORMER_PYTHON"] = args.python
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError("pipeline exited {}: {}".format(completed.returncode, completed.stderr[-2000:]))
    result = json.loads(completed.stdout)
    if result.get("test_evaluated") is not False:
        raise RuntimeError("test leakage detected during formal V1 promotion")
    if result.get("program_id") != candidate["program_id"]:
        raise RuntimeError("promotion changed Program ID")
    if not result.get("valid"):
        raise RuntimeError("promotion failed: {}".format(result.get("error", "unknown failure")))
    if int(result.get("endpoint_step", 0)) != int(max_steps):
        raise RuntimeError("promotion did not reach the requested endpoint")
    return result


def update_status(root, state):
    lines = [
        "# 正式V1等变DSL多保真实验",
        "",
        "- 更新时间：`{}`".format(now()),
        "- 阶段：`{}`".format(state.get("stage")),
        "- Test参与搜索：`false`",
        "- 8k完成：`{}`".format(len(state.get("candidates_8000", []))),
        "- 80k完成：`{}`".format(len(state.get("candidates_80000", []))),
        "- 250k完成：`{}`".format(len(state.get("candidates_250000", []))),
    ]
    current = state.get("current")
    if current:
        lines.append("- 当前候选：`{}`，目标`{}` steps".format(current["architecture_id"], current["max_steps"]))
    (Path(root) / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--search-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--quarter-subset-file", required=True)
    parser.add_argument("--python", default=os.environ.get("EQUIFORMER_PYTHON", "/home/20262202788/conda-envs/equiformer/bin/python"))
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--cohort", type=int, default=8)
    parser.add_argument("--promote-80k", type=int, default=4)
    parser.add_argument("--promote-250k", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    subset = Path(args.quarter_subset_file).resolve()
    protocol = {
        "protocol_version": "formal-v1-8-4-2@1",
        "batch_size": 32,
        "stage_steps": [8000, 80000, 250000],
        "candidate_counts": [args.cohort, args.promote_80k, args.promote_250k],
        "validation_interval_data_epochs": 10,
        "quarter_subset_sha256": hashlib.sha256(subset.read_bytes()).hexdigest(),
        "seed": args.seed,
        "test_during_search": False,
        "full_transition": "model_only_optimizer_restart",
    }
    state_path = root / "state.json"
    state = read_json(state_path, {"stage": "collect_8k", "protocol": protocol, "created_at": now()})
    if state.get("protocol") != protocol:
        raise RuntimeError("existing run protocol differs; resume the original protocol in the same directory")
    stop = root / "STOP"

    if state["stage"] == "collect_8k":
        state["candidates_8000"] = collect_8k_candidates(args.search_dir, args.cohort)
        state["stage"] = "promote_80k"
        write_json(state_path, state)
        update_status(root, state)

    if state["stage"] == "promote_80k" and not stop.exists():
        completed = state.get("candidates_80000", [])
        done = {item["architecture_id"] for item in completed}
        for candidate in state["candidates_8000"]:
            if len(completed) >= args.promote_80k or stop.exists():
                break
            if candidate["architecture_id"] in done:
                continue
            state["current"] = {"architecture_id": candidate["architecture_id"], "max_steps": 80000}
            write_json(state_path, state)
            update_status(root, state)
            metrics = run_pipeline(args, candidate, max_steps=80000, checkpoint=candidate["checkpoint_8000"], subset_file=str(subset))
            item = dict(candidate, metrics_80000=metrics, checkpoint_80000=metrics["checkpoint_last"])
            completed.append(item)
            state["candidates_80000"] = completed
            write_json(state_path, state)
        if len(completed) >= args.promote_80k:
            state["candidates_80000"] = sorted(completed, key=lambda item: float(item["metrics_80000"]["validation_alpha_mae"]))
            state["stage"] = "promote_250k"
            state.pop("current", None)
            write_json(state_path, state)
            update_status(root, state)

    if state["stage"] == "promote_250k" and not stop.exists():
        completed = state.get("candidates_250000", [])
        done = {item["architecture_id"] for item in completed}
        for candidate in state["candidates_80000"]:
            if len(completed) >= args.promote_250k or stop.exists():
                break
            if candidate["architecture_id"] in done:
                continue
            state["current"] = {"architecture_id": candidate["architecture_id"], "max_steps": 250000}
            write_json(state_path, state)
            update_status(root, state)
            metrics = run_pipeline(args, candidate, max_steps=250000, checkpoint=candidate["checkpoint_80000"], subset_file="", transition=True)
            item = dict(candidate, metrics_250000=metrics, checkpoint_250000=metrics["checkpoint_last"])
            completed.append(item)
            state["candidates_250000"] = completed
            write_json(state_path, state)
        if len(completed) >= args.promote_250k:
            completed = sorted(completed, key=lambda item: float(item["metrics_250000"]["validation_alpha_mae"]))
            state["candidates_250000"] = completed
            state["winner_by_validation"] = completed[0]
            state["stage"] = "completed_validation_selection"
            state.pop("current", None)
            state["completed_at"] = now()
            write_json(state_path, state)
            append_jsonl(root / "full_fidelity_archive.jsonl", dict(completed[0], selected_at=now(), selection_split="validation"))
            update_status(root, state)

    print(json.dumps({"root": str(root), "stage": state["stage"], "test_evaluated": False}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
