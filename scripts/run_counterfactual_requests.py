#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from equivariant_nas.candidate import apply_factor_patch, render_candidate
from equivariant_nas.interaction import interaction_contrast
from equivariant_nas.pipeline import evaluate_candidate_pipeline
from equivariant_nas.spec import ArchitectureSpec, EvolutionFactor


ROOT = Path("/home/20262202788/equivariant-nas")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    requests = [
        json.loads(line)
        for line in Path(args.requests).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, request in enumerate(requests, 1):
        ancestor = ArchitectureSpec.from_dict(request["ancestor_spec"])
        factor = EvolutionFactor(request["selected_factor"])
        sibling = apply_factor_patch(
            ancestor, factor, request["factor_replacement"]
        )
        program = output / "counterfactual_{:04d}.py".format(index)
        program.write_text(render_candidate(sibling), encoding="utf-8")
        metrics = evaluate_candidate_pipeline(
            program_path=str(program),
            project_root=str(ROOT),
            equiformer_root="/home/20262202788/equiformer",
            data_path="/home/20262202788/equiformer/datasets/qm9",
            max_steps=args.max_steps,
            seed=args.seed,
            run_symmetry=True,
            gpu_budget_hours=None,
        )
        record = dict(request)
        record["counterfactual_architecture_id"] = sibling.architecture_id()
        record["counterfactual_metrics"] = metrics
        if metrics.get("valid") and metrics.get("validation_alpha_mae") is not None:
            contrast = interaction_contrast(
                request["ancestor_mae"],
                request["parent_mae"],
                metrics["validation_alpha_mae"],
                request["child_mae"],
            )
            record["interaction_contrast"] = contrast.to_dict()
            record["status"] = "resolved"
        else:
            record["status"] = "counterfactual_failed"
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    (output / "results.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
