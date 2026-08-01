#!/usr/bin/env python
"""Resumable, evidence-checked parent/child showcase training controller."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl.pipeline import evaluate_dsl_candidate_pipeline


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _generation_evidence(search_dir: Path, child_program: Path):
    child_payload = json.loads(child_program.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (search_dir / "evolution.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matches = [record for record in records if record.get("architecture_id") and record.get("patch")]
    if len(matches) != 1:
        raise RuntimeError("showcase generation run must contain exactly one materialized LLM child")
    record = matches[0]
    required = {"router_response", "critic_response", "patch", "region_audit"}
    missing = sorted(required - set(record))
    if missing:
        raise RuntimeError("generation evidence is missing {}".format(missing))
    if record["region_audit"].get("lowering_plan", {}).get("mode") != "exact_hybrid":
        raise RuntimeError("generated child lacks exact_hybrid lowering evidence")
    with sqlite3.connect(str(search_dir / "evidence.sqlite")) as connection:
        roles = {
            role: count
            for role, count in connection.execute(
                "SELECT role, COUNT(*) FROM prompt_runs GROUP BY role"
            ).fetchall()
        }
    for role in ("region_router", "region_critic", "patch_synthesizer"):
        if roles.get(role, 0) < 1:
            raise RuntimeError("generation evidence lacks mandatory {} call".format(role))
    return {
        "architecture_id": record["architecture_id"],
        "program_id": child_payload.get("program_id", ""),
        "iteration": record["iteration"],
        "router_response": record["router_response"],
        "critic_response": record["critic_response"],
        "patch": record["patch"],
        "region_audit": record["region_audit"],
        "prompt_role_counts": roles,
        "generation_search_dir": str(search_dir),
    }


def _enrich_training_identity(result):
    """Promote immutable trainer protocol evidence into the candidate result."""
    if not result or not result.get("run_dir"):
        return result
    summary_path = Path(result["run_dir"]) / "training" / "training_summary.json"
    if not summary_path.exists():
        return result
    summary = _read(summary_path, {}) or {}
    result.update(
        {
            "training_dataset_id": summary.get("training_dataset_id"),
            "training_dataset_size": summary.get("training_dataset_size"),
            "steps_per_data_epoch": summary.get("steps_per_data_epoch"),
            "completed_data_epochs": summary.get("completed_data_epochs"),
            "eval_interval_epochs": summary.get("eval_interval_epochs"),
            "data_epoch_origin_step": summary.get("data_epoch_origin_step"),
            "train_subset": summary.get("train_subset"),
        }
    )
    return result


def run(args):
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    state = _read(state_path, {}) or {}
    generation = _generation_evidence(Path(args.generation_search).resolve(), Path(args.child_program).resolve())
    existing = _read(root / "experiment_preregistration.json")
    preregistration = {
        "experiment_id": root.name,
        "created_at": state.get("created_at", (existing or {}).get("created_at", _now())),
        "dataset": "QM9",
        "target": "alpha",
        "selection_split": "validation",
        "test_evaluated": False,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_optimizer_steps": args.max_steps,
        "training_subset": str(Path(args.train_subset_file).resolve()),
        "validation_interval_data_epochs": args.eval_interval_epochs,
        "candidates": ["official_v1_parent", "llm_v1_readout_child"],
        "generation_architecture_id": generation["architecture_id"],
    }
    if existing is not None and existing != preregistration:
        raise RuntimeError("showcase preregistration changed after the run was created")
    _write(root / "experiment_preregistration.json", preregistration)
    _write(root / "generation_evidence.json", generation)

    common = dict(
        project_root=args.project_root,
        equiformer_root=args.equiformer_root,
        data_path=args.data_path,
        max_steps=args.max_steps,
        seed=args.seed,
        batch_size=args.batch_size,
        train_subset_file=args.train_subset_file,
        eval_interval_epochs=args.eval_interval_epochs,
        task_contract_path=args.task_contract,
        run_symmetry=True,
    )
    stages = (("parent", args.parent_program, "exact_reference"), ("child", args.child_program, "exact_hybrid"))
    results = {
        name: _enrich_training_identity(dict(result))
        for name, result in dict(state.get("results", {})).items()
    }
    for name, result in results.items():
        _write(root / (name + "_result.json"), result)
    for name, program, expected_lowering in stages:
        cached = results.get(name)
        if cached and cached.get("valid") and cached.get("endpoint_step") == args.max_steps:
            continue
        _write(
            state_path,
            {
                "status": "running",
                "stage": name,
                "created_at": preregistration["created_at"],
                "updated_at": _now(),
                "results": results,
            },
        )
        result = evaluate_dsl_candidate_pipeline(program_path=program, **common)
        results[name] = _enrich_training_identity(result)
        result = results[name]
        _write(root / (name + "_result.json"), result)
        if not result.get("valid"):
            raise RuntimeError("{} evaluation failed: {}".format(name, result.get("error", "unknown error")))
        if result.get("lowering_mode") != expected_lowering:
            raise RuntimeError("{} used unexpected lowering {}".format(name, result.get("lowering_mode")))
        if result.get("test_evaluated"):
            raise RuntimeError("showcase search accessed the locked test split")
        if int(result.get("endpoint_step", -1)) != args.max_steps:
            raise RuntimeError("{} stopped before the registered endpoint".format(name))
        checkpoint = Path(result.get("checkpoint_last", ""))
        if not checkpoint.is_file():
            raise RuntimeError("{} lacks a resumable checkpoint".format(name))

    parent = results["parent"]
    child = results["child"]
    summary = {
        "status": "completed",
        "created_at": preregistration["created_at"],
        "completed_at": _now(),
        "protocol": preregistration,
        "generation": generation,
        "parent": parent,
        "child": child,
        "endpoint_validation_mae_delta": float(child["validation_alpha_mae"]) - float(parent["validation_alpha_mae"]),
        "child_improved": float(child["validation_alpha_mae"]) < float(parent["validation_alpha_mae"]),
        "test_evaluated": False,
    }
    _write(root / "showcase_summary.json", summary)
    _write(
        state_path,
        {
            "status": "completed",
            "stage": "done",
            "created_at": preregistration["created_at"],
            "updated_at": _now(),
            "results": results,
            "summary": str(root / "showcase_summary.json"),
        },
    )
    print(json.dumps({"status": "completed", "summary": str(root / "showcase_summary.json")}, ensure_ascii=False))
    return summary


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--generation-search", required=True)
    parser.add_argument("--parent-program", required=True)
    parser.add_argument("--child-program", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--equiformer-root", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--train-subset-file", required=True)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    return parser


if __name__ == "__main__":
    run(get_parser().parse_args())
