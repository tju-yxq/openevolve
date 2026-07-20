#!/usr/bin/env python
import json
from pathlib import Path

from equivariant_nas.pipeline import evaluate_candidate_pipeline


ROOT = Path("/home/20262202788/equivariant-nas")
NAMES = ("action_fast_layer", "action_graph_norm", "operator_bessel")


def main():
    results = []
    for name in NAMES:
        candidate = ROOT / "runs/gate1_candidates" / name / "candidate.py"
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
                    "training_sec": result.get("training_time_sec"),
                    "error": result.get("error", ""),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    (ROOT / "runs/gate1_training_additional.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
