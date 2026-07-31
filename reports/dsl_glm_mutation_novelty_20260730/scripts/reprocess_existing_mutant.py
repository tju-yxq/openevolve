#!/usr/bin/env python
"""Reapply the bounded-gate repair to an existing raw GLM candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("reprocess-existing-mutant")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--generator-script", required=True)
    parser.add_argument("--factor-id", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_module(path):
    spec = importlib.util.spec_from_file_location("mutation_generator", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from equivariant_nas.dsl import Compiler, core_registry, reference_motif_registry
    from equivariant_nas.dsl.serialization import dumps_program, load_program, load_task_contract

    helper = load_module(Path(args.generator_script).resolve())
    parent = load_program(str(Path(args.parent).resolve()))
    raw = load_program(str(Path(args.candidate).resolve()))
    task = load_task_contract(str(Path(args.task_contract).resolve()))
    repaired, repairs = helper.deterministic_stability_repair(raw, args.factor_id)
    compiler = Compiler(core_registry(), reference_motif_registry())
    artifact = compiler.analyze(repaired, task)
    lowering = compiler.plan_lowering(repaired, task)
    factor_audit = helper.factor_contract_audit(parent, repaired, args.factor_id)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_program(repaired), encoding="utf-8")
    summary = {
        "raw_architecture_id": compiler.analyze(raw, task).architecture_id,
        "repaired_architecture_id": artifact.architecture_id,
        "repairs": list(repairs),
        "factor_contract": factor_audit,
        "lowering": lowering.to_dict(),
        "training_started": False,
        "test_split_loaded": False,
    }
    evidence = output.with_suffix(output.suffix + ".evidence.json")
    evidence.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
