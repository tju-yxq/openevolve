#!/usr/bin/env python
import csv
import json
from pathlib import Path

from equivariant_nas.pipeline import evaluate_candidate_pipeline


ROOT = "/home/20262202788/equivariant-nas"


def main():
    candidate_root = Path(ROOT) / "runs" / "gate1_candidates"
    results = []
    for candidate in sorted(candidate_root.glob("*/candidate.py")):
        result = evaluate_candidate_pipeline(
            program_path=str(candidate),
            project_root=ROOT,
            equiformer_root="/home/20262202788/equiformer",
            data_path="/home/20262202788/equiformer/datasets/qm9",
            max_steps=0,
            seed=0,
            parameter_ratio_limit=1.2,
            symmetry_threshold=2.5e-1,
            run_symmetry=True,
            gpu_budget_hours=5.0,
        )
        result["candidate_name"] = candidate.parent.name
        results.append(result)
        print(
            json.dumps(
                {
                    "candidate": candidate.parent.name,
                    "valid": result.get("valid"),
                    "params": result.get("parameter_count"),
                    "symmetry": result.get("max_symmetry_error"),
                    "error": result.get("error", ""),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output = Path(ROOT) / "runs" / "gate1_feasibility.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    columns = sorted({key for row in results for key in row if key != "symmetry_report"})
    with (Path(ROOT) / "runs" / "gate1_feasibility.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in columns})


if __name__ == "__main__":
    main()
