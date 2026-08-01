#!/usr/bin/env python
"""Revalidate persisted V3 evolution candidates with generic Lowering."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_V3_ROOT = PROJECT_ROOT.parent / "equiformer_v3_official"

from equivariant_nas.dsl import core_registry
from equivariant_nas.dsl.backends import E3NNGraphBackend
from equivariant_nas.dsl.serialization import load_program
from scripts.run_dsl_v3_evolution import _validate_candidate, _write_json


def _records(run_dir: Path):
    path = run_dir / "evolution.jsonl"
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [item for item in values if int(item.get("round", 0)) > 0]


def validate_run(args):
    run_dir = Path(args.run).resolve()
    records = _records(run_dir)
    if not records:
        raise RuntimeError("V3 evolution run contains no generated candidates")
    selected = records if args.scope == "all" else [records[-1]]
    registry = core_registry()
    backend = E3NNGraphBackend(
        registry,
        equiformer_v3_root=args.equiformer_v3_root,
    )
    results = []
    for record in selected:
        program = load_program(record["candidate_path"])
        validation = _validate_candidate(
            program,
            registry,
            backend,
            validation_level=args.validation_level,
            seed=args.seed + int(record["round"]),
        )
        results.append({
            "round": int(record["round"]),
            "architecture_id": validation["architecture_id"],
            "node_count": validation["node_count"],
            "trainable_parameter_count": validation.get("trainable_parameter_count"),
            "constructor_bypass": validation["constructor_bypass"],
            "unsupported_nodes": validation["backend_support"]["unsupported_nodes"],
            "composition_errors": validation["backend_support"]["composition_errors"],
            "runtime": validation.get("runtime"),
            "status": validation["status"],
        })
        gc.collect()
    payload = {
        "status": "passed",
        "scope": args.scope,
        "validation_level": args.validation_level,
        "candidate_count": len(results),
        "unique_architecture_count": len({item["architecture_id"] for item in results}),
        "all_generic_lowering_validations_passed": all(item["status"] == "passed" for item in results),
        "all_constructor_bypass_false": all(not item["constructor_bypass"] for item in results),
        "records": results,
    }
    output = Path(args.output).resolve() if args.output else run_dir / "validation_{}_{}.json".format(
        args.scope,
        args.validation_level,
    )
    _write_json(output, payload)
    print(json.dumps({**payload, "output": str(output)}, ensure_ascii=False, sort_keys=True))
    return payload


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--scope", choices=("all", "final"), default="all")
    parser.add_argument("--validation-level", choices=("static", "build", "full"), default="build")
    parser.add_argument(
        "--equiformer-v3-root",
        default=(
            os.environ.get("EQUIFORMER_V3_ROOT", "")
            or (str(DEFAULT_V3_ROOT) if DEFAULT_V3_ROOT.exists() else "")
        ),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=20260801)
    return parser


def main():
    args = get_parser().parse_args()
    if not args.equiformer_v3_root:
        raise ValueError("V3 validation requires --equiformer-v3-root")
    validate_run(args)


if __name__ == "__main__":
    main()
