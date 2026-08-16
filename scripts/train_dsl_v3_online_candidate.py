#!/usr/bin/env python
"""Formal run_pipeline adapter for exact 0->20k->80k->250k continuation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def required_env(name):
    value = os.environ.get(name, "")
    if not value: raise RuntimeError(f"required environment variable {name} is empty")
    return value


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--endpoint", type=int, required=True); parser.add_argument("--checkpoint", default=""); args = parser.parse_args()
    if args.endpoint not in {20000, 80000, 250000}: raise ValueError("unsupported frozen endpoint")
    candidate = json.loads(os.environ["EQUINAS_REQUEST_JSON"])
    python = os.environ.get("EQUIFORMER_PYTHON", sys.executable)
    command = [python, str(PROJECT_ROOT / "scripts/run_pipeline.py"), "--program", candidate["program"], "--project-root", str(PROJECT_ROOT), "--equiformer-root", required_env("EQUIFORMER_ROOT"), "--equiformer-v3-root", required_env("EQUIFORMER_V3_ROOT"), "--data-path", required_env("EQUINAS_QM9_PATH"), "--max-steps", str(args.endpoint), "--seed", "201", "--batch-size", "8", "--eval-interval-epochs", os.environ.get("EQUINAS_EVAL_INTERVAL_EPOCHS", "10")]
    task = os.environ.get("EQUINAS_TASK_CONTRACT", "")
    if task: command += ["--dsl-task-contract", task]
    if args.endpoint in {20000, 80000}:
        command += ["--train-subset-file", required_env("EQUINAS_QUARTER_SUBSET_FILE")]
    if args.checkpoint:
        command += ["--resume-checkpoint", args.checkpoint]
    if args.endpoint == 250000:
        command += ["--allow-data-transition", "--resume-model-only", "--data-epoch-origin-step", "80000", "--lr-schedule-origin-step", "80000"]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode: raise RuntimeError(completed.stderr[-8000:])
    result = json.loads(completed.stdout)
    result["training_data"] = "full_train" if args.endpoint == 250000 else "fixed_quarter"
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
