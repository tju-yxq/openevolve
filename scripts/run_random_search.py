#!/usr/bin/env python
"""Matched-budget factor-local random-search pilot baseline."""

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

from equivariant_nas.candidate import render_candidate
from equivariant_nas.pipeline import evaluate_candidate_pipeline
from equivariant_nas.spec import EvolutionFactor, baseline_spec


def mutate(parent, factor, rng):
    if factor == EvolutionFactor.REPRESENTATION:
        rep = parent.representation
        field = rng.choice(
            ["scalar_channels", "vector_channels", "tensor_channels", "mlp_multiplier", "feature_channels"]
        )
        choices = {
            "scalar_channels": [64, 96, 128, 160, 192],
            "vector_channels": [32, 48, 64, 96],
            "tensor_channels": [16, 24, 32, 48],
            "mlp_multiplier": [2, 3, 4],
            "feature_channels": [256, 384, 512, 640],
        }[field]
        values = [value for value in choices if value != getattr(rep, field)]
        return replace(parent, representation=replace(rep, **{field: rng.choice(values)}))
    if factor == EvolutionFactor.OPERATOR:
        op = parent.operator
        field = rng.choice(
            ["basis_type", "num_basis", "radial_hidden", "nonlinear_message", "num_heads"]
        )
        choices = {
            "basis_type": ["gaussian", "bessel"],
            "num_basis": [32, 64, 96, 128],
            "radial_hidden": [(32, 32), (64, 64), (96, 96), (128, 128)],
            "nonlinear_message": [False, True],
            "num_heads": [2, 4, 8],
        }[field]
        values = [value for value in choices if value != getattr(op, field)]
        return replace(parent, operator=replace(op, **{field: rng.choice(values)}))
    if factor == EvolutionFactor.ACTION:
        action = parent.action
        field = rng.choice(
            ["norm_layer", "rescale_degree", "alpha_drop", "projection_drop", "output_drop", "drop_path"]
        )
        choices = {
            "norm_layer": ["layer", "instance", "graph", "fast_layer"],
            "rescale_degree": [False, True],
            "alpha_drop": [0.0, 0.05, 0.1, 0.2],
            "projection_drop": [0.0, 0.05, 0.1, 0.2],
            "output_drop": [0.0, 0.05, 0.1, 0.2],
            "drop_path": [0.0, 0.05, 0.1, 0.2],
        }[field]
        values = [value for value in choices if value != getattr(action, field)]
        return replace(parent, action=replace(action, **{field: rng.choice(values)}))
    macro = parent.macro
    field = rng.choice(["num_layers", "radius"])
    choices = {
        "num_layers": [3, 4, 5, 6, 7, 8],
        "radius": [4.0, 5.0, 6.0],
    }[field]
    values = [value for value in choices if value != getattr(macro, field)]
    return replace(parent, macro=replace(macro, **{field: rng.choice(values)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument(
        "--valid-target",
        type=int,
        default=0,
        help="Stop after this many trained-valid candidates; 0 uses proposal count.",
    )
    parser.add_argument("--max-proposals", type=int, default=20)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    root = Path("/home/20262202788/equivariant-nas")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pool = [baseline_spec()]
    seen = {pool[0].architecture_id()}
    records = []
    proposal_limit = args.max_proposals if args.valid_target > 0 else args.iterations
    iteration = 0
    valid_count = 0
    while iteration < proposal_limit:
        iteration += 1
        for _ in range(100):
            parent = rng.choice(pool)
            factor = rng.choice(list(EvolutionFactor))
            child = mutate(parent, factor, rng).validate()
            if child.architecture_id() not in seen:
                break
        candidate = output / "candidate_{:04d}.py".format(iteration)
        candidate.write_text(render_candidate(child), encoding="utf-8")
        metrics = evaluate_candidate_pipeline(
            str(candidate),
            str(root),
            "/home/20262202788/equiformer",
            "/home/20262202788/equiformer/datasets/qm9",
            max_steps=args.max_steps,
            seed=0,
            run_symmetry=True,
            gpu_budget_hours=None,
        )
        record = {
            "iteration": iteration,
            "factor": factor.value,
            "parent_id": parent.architecture_id(),
            "architecture_id": child.architecture_id(),
            "metrics": metrics,
        }
        records.append(record)
        seen.add(child.architecture_id())
        if metrics.get("valid"):
            pool.append(child)
            valid_count += 1
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "factor": factor.value,
                    "valid": metrics.get("valid"),
                    "val_mae": metrics.get("validation_alpha_mae"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.valid_target > 0 and valid_count >= args.valid_target:
            break
    summary = {
        "iterations": len(records),
        "requested_valid_target": args.valid_target,
        "valid": valid_count,
        "best_val_mae": min(
            (
                item["metrics"]["validation_alpha_mae"]
                for item in records
                if item["metrics"].get("valid")
            ),
            default=None,
        ),
        "records": records,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
