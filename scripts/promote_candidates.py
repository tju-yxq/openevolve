#!/usr/bin/env python
import argparse
import json
from pathlib import Path

from equivariant_nas.pipeline import evaluate_candidate_pipeline
from equivariant_nas.promotion import authorize_promotion


ROOT = Path("/home/20262202788/equivariant-nas")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--purpose", choices=("calibration", "selection"), required=True
    )
    parser.add_argument(
        "--trust-report",
        default="",
        help="SCFTG JSON report; mandatory for selection promotions.",
    )
    args = parser.parse_args()
    trust_report = (
        json.loads(Path(args.trust_report).read_text(encoding="utf-8"))
        if args.trust_report
        else None
    )
    selection_eligible = authorize_promotion(args.purpose, trust_report)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = []
    for item in manifest:
        result = evaluate_candidate_pipeline(
            program_path=item["program"],
            project_root=str(ROOT),
            equiformer_root="/home/20262202788/equiformer",
            data_path="/home/20262202788/equiformer/datasets/qm9",
            max_steps=args.max_steps,
            seed=int(item.get("seed", 0)),
            run_symmetry=False,
            gpu_budget_hours=5.0,
            resume_checkpoint=item["checkpoint"],
        )
        result["name"] = item["name"]
        result["promotion_purpose"] = args.purpose
        result["selection_eligible"] = selection_eligible
        results.append(result)
        print(
            json.dumps(
                {
                    "name": item["name"],
                    "valid": result.get("valid"),
                    "val_mae": result.get("validation_alpha_mae"),
                    "used_gpu_hours": result.get("used_gpu_hours"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
