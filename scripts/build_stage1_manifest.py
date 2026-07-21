#!/usr/bin/env python
"""Create a hash-addressed manifest for stage-one code and evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path("/home/20262202788/equivariant-nas")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def dirty(path):
    try:
        return bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True
            ).strip()
        )
    except Exception:
        return None


def main():
    patterns = [
        "README.md",
        "METHOD.md",
        "REPRODUCE.md",
        "PHASE2_PREREGISTRATION.md",
        "configs/protocol.json",
        "configs/phase2_preregistration.json",
        "configs/stage1_factor_memory.json",
        "configs/stage1_factor_memory_parent_child.json",
        "configs/inheritance_calibration.json",
        "configs/stage1_fidelity_pairs.json",
        "configs/ablations/*.py",
        "equivariant_nas/*.py",
        "equivariant_nas/training/*.py",
        "openevolve_adapter/*.py",
        "reporting/*",
        "scripts/*.py",
        "scripts/*.sh",
        "tests/*.py",
        "runs/fidelity_trust_300_to_5000.json",
        "runs/budget_ledger.jsonl",
        "runs/stable_factorized_seed45/evolution.jsonl",
        "runs/stable_factorized_seed45/paired_budget_stop.json",
        "runs/stable_factorized_seed45/counterfactual_requests.jsonl",
        "runs/stable_random_seed45/summary.json",
        "runs/stage1_interaction_ablation/results.json",
        "reports/stage1/evidence/memory_smoke/*",
        "reports/stage1/evidence/state_transfer/*",
        "reports/stage1/stage1_report.md",
        "reports/stage1/stage1_report.html",
        "reports/stage1/test_output.txt",
        "reports/stage1/assets/*.png",
    ]
    files = []
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            if path.is_file():
                files.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    payload = {
        "repositories": {
            "equiformer": {
                "commit": revision(Path("/home/20262202788/equiformer")),
                "dirty": dirty(Path("/home/20262202788/equiformer")),
            },
            "openevolve": {
                "commit": revision(Path("/home/20262202788/openevolve")),
                "dirty": dirty(Path("/home/20262202788/openevolve")),
            },
            "spark": {
                "commit": revision(Path("/home/20262202788/SPARK")),
                "dirty": dirty(Path("/home/20262202788/SPARK")),
            },
        },
        "protocol": json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf-8")),
        "files": files,
    }
    output = ROOT / "reports/stage1/manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
