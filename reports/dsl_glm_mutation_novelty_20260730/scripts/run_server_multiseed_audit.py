#!/usr/bin/env python
"""Re-audit accepted DSL mutants on multiple synthetic random seeds.

This runner is intentionally dataset-free: it invokes the existing synthetic
SO(3) audit for the frozen parent and every accepted candidate, then aggregates
the resulting numerical evidence in the same run directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("run-server-multiseed-audit")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--audit-script", required=True)
    parser.add_argument("--seeds", default="1201,2201")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_audit(
    python: str,
    audit_script: Path,
    project_root: Path,
    task_contract: Path,
    program: Path,
    parent_program: Path,
    output: Path,
    seed: int,
    dtype: str,
    device: str,
) -> dict:
    if not output.is_file():
        command = [
            python,
            str(audit_script),
            "--project-root",
            str(project_root),
            "--task-contract",
            str(task_contract),
            "--program",
            str(program),
            "--parent-program",
            str(parent_program),
            "--output",
            str(output),
            "--device",
            device,
            "--dtype",
            dtype,
            "--seed",
            str(seed),
        ]
        subprocess.run(command, check=True)
    return read_json(output)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    task_contract = Path(args.task_contract).resolve()
    run_dir = Path(args.run_dir).resolve()
    audit_script = Path(args.audit_script).resolve()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    summary = read_json(run_dir / "summary.json")
    parent_program = run_dir / "audit_parent.dsl.json"
    output_dir = run_dir / "multi_seed_audit"

    parent_by_seed = {}
    for seed in seeds:
        parent_output = output_dir / f"parent_seed{seed}.json"
        parent_by_seed[seed] = run_audit(
            sys.executable,
            audit_script,
            project_root,
            task_contract,
            parent_program,
            parent_program,
            parent_output,
            seed,
            args.dtype,
            args.device,
        )

    candidate_rows = []
    for candidate in summary.get("candidates", []):
        architecture_id = str(candidate["architecture_id"])
        program = Path(candidate["candidate_path"]).resolve()
        seed_rows = []
        for seed in seeds:
            output = output_dir / f"{architecture_id}_seed{seed}.json"
            audit = run_audit(
                sys.executable,
                audit_script,
                project_root,
                task_contract,
                program,
                parent_program,
                output,
                seed,
                args.dtype,
                args.device,
            )
            parent_norm = float(parent_by_seed[seed]["reference_output"]["norm"])
            candidate_norm = float(audit["reference_output"]["norm"])
            seed_rows.append(
                {
                    "seed": seed,
                    "equivariance_passed": bool(audit.get("equivariance_passed")),
                    "path_activity": (audit.get("inserted_computation_path_activity") or {}).get("status"),
                    "maximum_relative_error": float(audit["relative_error"]["maximum"]),
                    "maximum_absolute_error": float(audit["absolute_error"]["maximum"]),
                    "output_norm_ratio_to_parent": candidate_norm / max(parent_norm, 1.0e-30),
                    "maximum_inserted_gradient": max(
                        [
                            float(item.get("gradient_norm", 0.0))
                            for item in audit.get("inserted_parameter_gradients", [])
                        ]
                        or [0.0]
                    ),
                    "training_started": bool(audit.get("training_started")),
                    "test_split_loaded": bool(audit.get("test_split_loaded")),
                    "evidence_path": str(output),
                }
            )
        candidate_rows.append(
            {
                "architecture_id": architecture_id,
                "factor_id": candidate.get("factor_id"),
                "all_equivariance_passed": all(row["equivariance_passed"] for row in seed_rows),
                "all_paths_active": all(row["path_activity"] == "active" for row in seed_rows),
                "maximum_relative_error": max(row["maximum_relative_error"] for row in seed_rows),
                "maximum_output_norm_ratio_to_parent": max(
                    row["output_norm_ratio_to_parent"] for row in seed_rows
                ),
                "maximum_inserted_gradient": max(
                    row["maximum_inserted_gradient"] for row in seed_rows
                ),
                "seeds": seed_rows,
            }
        )

    aggregate = {
        "protocol": "multi-seed-synthetic-so3-reaudit-v1",
        "run_dir": str(run_dir),
        "seed_list": list(seeds),
        "candidate_count": len(candidate_rows),
        "all_candidates_equivariance_passed": all(
            row["all_equivariance_passed"] for row in candidate_rows
        ),
        "all_candidate_paths_active": all(row["all_paths_active"] for row in candidate_rows),
        "training_started": any(
            seed_row["training_started"]
            for row in candidate_rows
            for seed_row in row["seeds"]
        ),
        "test_split_loaded": any(
            seed_row["test_split_loaded"]
            for row in candidate_rows
            for seed_row in row["seeds"]
        ),
        "candidates": candidate_rows,
    }
    write_json(output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
