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
    command.extend(
        [
            "--batch-size",
            os.environ.get("NAS_BATCH_SIZE", "64"),
            "--eval-interval-epochs",
            os.environ.get("NAS_EVAL_INTERVAL_EPOCHS", "0"),
            "--data-epoch-origin-step",
            os.environ.get("NAS_DATA_EPOCH_ORIGIN_STEP", "0"),
            "--lr-schedule-origin-step",
            os.environ.get("NAS_LR_SCHEDULE_ORIGIN_STEP", "0"),
        ]
    )
    subset_file = os.environ.get("NAS_TRAIN_SUBSET_FILE", "")
    if subset_file:
        command.extend(["--train-subset-file", subset_file])
    if os.environ.get("NAS_ALLOW_DATA_TRANSITION", "0") == "1":
        command.append("--allow-data-transition")
    if os.environ.get("NAS_RESUME_MODEL_ONLY", "0") == "1":
        command.append("--resume-model-only")
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
