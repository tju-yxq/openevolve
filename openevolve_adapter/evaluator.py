"""OpenEvolve evaluator entry point.

This thin adapter is imported by the OpenEvolve environment. Heavy model work
runs in the pinned Equiformer environment through a subprocess.
"""

import json
import os
import subprocess


ROOT = os.environ.get("EQUIVARIANT_NAS_ROOT", "/home/20262202788/equivariant-nas")
PYTHON = os.environ.get(
    "EQUIFORMER_PYTHON", "/home/20262202788/conda-envs/equiformer/bin/python"
)


def evaluate(program_path):
    max_steps = os.environ.get("NAS_MAX_STEPS", "0")
    command = [
        PYTHON,
        os.path.join(ROOT, "scripts", "run_pipeline.py"),
        "--program",
        program_path,
        "--project-root",
        ROOT,
        "--equiformer-root",
        "/home/20262202788/equiformer",
        "--data-path",
        "/home/20262202788/equiformer/datasets/qm9",
        "--max-steps",
        max_steps,
    ]
    if os.environ.get("NAS_SKIP_SYMMETRY", "0") == "1":
        command.append("--skip-symmetry")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ROOT
    completed = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=max(600, int(max_steps) * 2 + 1200),
        check=False,
    )
    if completed.returncode != 0:
        return {
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "adapter",
            "error": completed.stderr[-2000:],
        }
    return json.loads(completed.stdout)
