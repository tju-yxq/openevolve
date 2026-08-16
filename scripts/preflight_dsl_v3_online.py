#!/usr/bin/env python
"""Fail-closed preflight for the formal online V3 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
from equivariant_nas.dsl import OnlineV3Protocol


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--protocol-config", default=str(PROJECT_ROOT / "configs/dsl_v3_online_20k_60_protocol.json")); parser.add_argument("--run-root", default=os.environ.get("EQUINAS_RUN_ROOT", "")); parser.add_argument("--minimum-free-gib", type=float, default=150); parser.add_argument("--allow-active-gpu-scheduler", action="store_true"); args = parser.parse_args()
    protocol = OnlineV3Protocol.load(args.protocol_config); failures = []; checks = {}
    root = Path(args.run_root or protocol.raw["storage"]["default_run_root"])
    try:
        root.mkdir(parents=True, exist_ok=True); probe = root / ".preflight-write-probe"; probe.write_text("ok"); probe.unlink(); checks["run_root_writable"] = True
        free_gib = shutil.disk_usage(root).free / 2**30; checks["free_gib"] = free_gib
        if free_gib < args.minimum_free_gib: failures.append(f"run root has only {free_gib:.1f} GiB free")
    except OSError as error: failures.append(f"run root is not writable: {error}")
    identities = {
        "dataset_manifest_sha256": PROJECT_ROOT / "configs/dsl_v3_qm9_alpha_dataset_manifest.json",
        "quarter_subset_sha256": PROJECT_ROOT / "data_splits/qm9_train_quarter_seed201.npz",
        "equivariance_contract_sha256": PROJECT_ROOT / "configs/dsl_v3_qm9_alpha_equivariance_contract.json",
    }
    for key, path in identities.items():
        actual = sha256(path) if path.is_file() else "missing"; expected = protocol.raw["dataset"][key]; checks[key] = actual
        if actual != expected: failures.append(f"{key} mismatch: {actual} != {expected}")
    for name in ("EQUIFORMER_ROOT", "EQUIFORMER_V3_ROOT", "EQUINAS_QM9_PATH", "EQUINAS_QUARTER_SUBSET_FILE", "EQUINAS_INITIAL_PROGRAM"):
        value = os.environ.get(name, ""); checks[name] = value
        if not value or not Path(value).exists(): failures.append(f"{name} is unset or missing")
    try:
        import torch
        checks.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
        if not torch.cuda.is_available(): failures.append("CUDA is unavailable")
    except Exception as error: failures.append(f"PyTorch/CUDA check failed: {error}")
    if shutil.which("tmux"):
        active = subprocess.run(["tmux", "has-session", "-t", "gpu_scheduler"], capture_output=True).returncode == 0; checks["gpu_scheduler_active"] = active
        if active and not args.allow_active_gpu_scheduler: failures.append("tmux gpu_scheduler is active; stop it before formal training")
    report = {"ready": not failures, "protocol_hash": protocol.content_hash, "run_root": str(root), "checks": checks, "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures: raise SystemExit(2)


if __name__ == "__main__": main()

