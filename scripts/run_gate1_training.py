#!/usr/bin/env python
import csv
import json
from pathlib import Path

from equivariant_nas.pipeline import evaluate_candidate_pipeline


ROOT = Path("/home/20262202788/equivariant-nas")


def main():
    feasibility = json.loads((ROOT / "runs/gate1_feasibility.json").read_text(encoding="utf-8"))
    allowed = {row["candidate_name"] for row in feasibility if row.get("valid")}
    results = []
    for candidate in sorted((ROOT / "runs/gate1_candidates").glob("*/candidate.py")):
        name = candidate.parent.name
        if name not in allowed:
            continue
        result = evaluate_candidate_pipeline(
            program_path=str(candidate),
            project_root=str(ROOT),
            equiformer_root="/home/20262202788/equiformer",
            data_path="/home/20262202788/equiformer/datasets/qm9",
            max_steps=300,
            seed=0,
            parameter_ratio_limit=1.2,
            run_symmetry=False,
            gpu_budget_hours=5.0,
        )
        result["candidate_name"] = name
        results.append(result)
        print(
            json.dumps(
                {
                    "candidate": name,
                    "valid": result.get("valid"),
                    "val_mae": result.get("validation_alpha_mae"),
                    "params": result.get("parameter_count"),
                    "training_sec": result.get("training_time_sec"),
                    "gpu_hours_used": result.get("used_gpu_hours"),
                    "error": result.get("error", ""),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    (ROOT / "runs/gate1_training.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )
    columns = sorted({key for row in results for key in row if key != "symmetry_report"})
    with (ROOT / "runs/gate1_training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key) for key in columns})


if __name__ == "__main__":
    main()
