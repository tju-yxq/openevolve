#!/usr/bin/env python
"""Preflight and materialize the visible formal-V1 experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl import (
    Compiler,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    core_registry,
    equiformer_v1_capability_profile,
    import_equiformer_v1,
    reference_motif_registry,
    validate_region_transition,
    validate_unique_factor_ownership,
    v1_region_registry,
)
from equivariant_nas.dsl.serialization import dumps_program
from equivariant_nas.spec import baseline_spec


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_ok(command):
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return completed.returncode == 0, completed.stdout.strip() or completed.stderr.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    project = Path(config["project_root"]).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    checks = []
    def check(name, passed, details=""):
        checks.append({"name": name, "passed": bool(passed), "details": str(details)})

    for key in ("equiformer_root", "openevolve_root", "data_path", "quarter_subset_file", "task_contract", "llm_config"):
        path = Path(config[key])
        check("path:{}".format(key), path.exists(), path)
    for key in ("openevolve_python", "equiformer_python"):
        path = Path(config[key])
        check("executable:{}".format(key), path.is_file(), path)

    ok, status = command_ok(["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"])
    check("project_git_available", ok, status)
    check("project_git_clean", args.allow_dirty or (ok and not status), status)
    check("cuda_visible", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "missing")
    if shutil.which("nvidia-smi"):
        ok, gpu = command_ok(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
        check("gpu_query", ok and "A100" in gpu, gpu)

    profile = equiformer_v1_capability_profile()
    validate_unique_factor_ownership(profile.enabled_factors)
    parent = import_equiformer_v1(baseline_spec())
    compiler = Compiler(core_registry(), reference_motif_registry())
    parent_id = compiler.analyze(parent).architecture_id
    regions = v1_region_registry(parent)
    factor_results = []
    for factor in profile.enabled_factors:
        if not factor.parameter_paths:
            factor_results.append({"factor_id": factor.factor_id, "region_id": factor.region_id, "mode": "exact_hybrid", "status": "existing-certified-region"})
            continue
        for option_index, option in enumerate(factor.alternative_options):
            edits = tuple(
                PatchEdit("change_parameters", path, {"value": option[path.rsplit(".", 1)[1]]})
                for path in factor.parameter_paths
            )
            patch = TypedPatch(
                "1.0",
                parent_id,
                parent.language_version,
                {"factor_id": factor.factor_id, "claim": "preflight constructor mutation"},
                factor.parameter_paths,
                edits,
            )
            child = apply_typed_patch(
                parent,
                patch,
                compiler.primitives,
                expected_parent_id=parent_id,
                validate_child_with_core_registry=False,
            )
            region = next(item for item in regions if item.region_id == factor.region_id)
            audit = validate_region_transition(parent, child, region)
            plan = compiler.plan_lowering(child)
            check(
                "factor:{}:option{}".format(factor.factor_id, option_index),
                plan.mode == "exact_constructor",
                plan.to_dict(),
            )
            factor_results.append({
                "factor_id": factor.factor_id,
                "region_id": factor.region_id,
                "option_index": option_index,
                "option": dict(option),
                "mode": plan.mode,
                "audit": audit,
            })

    initial_program = run_root / "initial_program.dsl.json"
    initial_program.write_text(dumps_program(parent), encoding="utf-8")
    frozen_config = run_root / "formal_v1_config.json"
    frozen_config.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    protocol = {
        "version": config["version"],
        "config_sha256": sha256(frozen_config),
        "task_contract_sha256": sha256(config["task_contract"]),
        "quarter_subset_sha256": sha256(config["quarter_subset_file"]),
        "initial_program_sha256": sha256(initial_program),
        "test_during_search": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_root / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    ready = {
        "ready": all(item["passed"] for item in checks),
        "formal_v1_version": config["version"],
        "checks": checks,
        "capability_profile": profile.to_dict(),
        "factor_compile_results": factor_results,
        "initial_program": str(initial_program),
        "protocol": protocol,
    }
    ready_path = run_root / "READY_TO_LAUNCH.json"
    ready_path.write_text(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "STATUS.md").write_text(
        "# 等变DSL正式V1\n\n- 版本：`{}`\n- Ready：`{}`\n- 开放因子：`{}`\n- Test参与搜索：`false`\n".format(
            config["version"], str(ready["ready"]).lower(), ", ".join(item.factor_id for item in profile.enabled_factors)
        ),
        encoding="utf-8",
    )
    print(json.dumps({"ready": ready["ready"], "ready_file": str(ready_path), "run_root": str(run_root)}, ensure_ascii=False, sort_keys=True))
    if not ready["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
