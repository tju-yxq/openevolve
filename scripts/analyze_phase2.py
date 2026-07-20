#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from equivariant_nas.search_statistics import (
    bootstrap_mean_difference_ci,
    exact_paired_sign_flip_pvalue,
    summarize_trajectory,
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def values_for_run(path, method):
    path = Path(path)
    if method == "typed_random":
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        return [
            float(item["metrics"]["validation_alpha_mae"])
            for item in summary["records"]
            if item.get("metrics", {}).get("valid")
        ]
    records = read_jsonl(path / "evolution.jsonl")
    return [
        float(item["metrics"]["validation_alpha_mae"])
        for item in records
        if item.get("iteration", 0) > 0
        and item.get("metrics", {}).get("valid")
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = args.seeds or config["gate2a"]["search_seeds"]
    baseline = 0.7535404392242432
    threshold = float(config["time_to_threshold_mae"])
    methods = list(config["methods"])
    runs = {}
    for method in methods:
        runs[method] = {}
        for seed in seeds:
            path = Path(args.runs_root) / "{}_seed{}".format(method, seed)
            if not path.exists():
                continue
            values = values_for_run(path, method)
            runs[method][str(seed)] = {
                "path": str(path),
                "values": values,
                "summary": summarize_trajectory(
                    values, baseline, threshold
                ).to_dict(),
            }

    comparisons = {}
    full = runs.get("full", {})
    for control in ("uniform_router", "typed_random"):
        common = [str(seed) for seed in seeds if str(seed) in full and str(seed) in runs.get(control, {})]
        if not common:
            continue
        full_auc = [full[seed]["summary"]["normalized_best_so_far_auc"] for seed in common]
        control_auc = [
            runs[control][seed]["summary"]["normalized_best_so_far_auc"]
            for seed in common
        ]
        comparison = {
            "paired_seeds": [int(seed) for seed in common],
            "mean_full_auc": sum(full_auc) / len(full_auc),
            "mean_control_auc": sum(control_auc) / len(control_auc),
            "relative_auc_improvement": (
                (sum(control_auc) - sum(full_auc)) / sum(control_auc)
                if sum(control_auc)
                else 0.0
            ),
        }
        if len(common) >= 2:
            comparison["one_sided_exact_sign_flip_p"] = exact_paired_sign_flip_pvalue(
                full_auc, control_auc
            )
            comparison["paired_bootstrap_difference_ci"] = list(
                bootstrap_mean_difference_ci(full_auc, control_auc)
            )
        comparisons[control] = comparison

    payload = {
        "config": args.config,
        "seeds": seeds,
        "baseline_mae": baseline,
        "threshold_mae": threshold,
        "runs": runs,
        "comparisons": comparisons,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
