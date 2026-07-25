#!/usr/bin/env python
"""Preflight and materialize the visible formal-V1 experiment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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


def required_environment_variables(config_path):
    text = Path(config_path).read_text(encoding="utf-8")
    return sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)))


def _training_started(run_root):
    root = Path(run_root)
    if any(root.rglob("checkpoint_last.pth")):
        return True
    for evolution in root.rglob("evolution.jsonl"):
        for line in evolution.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            metrics = (json.loads(line).get("metrics") or {})
            if float(metrics.get("charged_gpu_seconds", 0.0)) > 0.0 or Path(str(metrics.get("checkpoint_last", ""))).is_file():
                return True
    return False


def write_frozen_material(path, text, run_root, label):
    path = Path(path)
    encoded = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return "unchanged"
    existed = path.exists()
    if existed and _training_started(run_root):
        raise RuntimeError("{} changed after training started; refuse protocol drift".format(label))
    if existed:
        previous = path.read_bytes()
        migrations = Path(run_root) / "protocol_migrations"
        migrations.mkdir(parents=True, exist_ok=True)
        old_hash = hashlib.sha256(previous).hexdigest()
        new_hash = hashlib.sha256(encoded).hexdigest()
        archive = migrations / "{}_{}.previous".format(label, old_hash[:12])
        if not archive.exists():
            archive.write_bytes(previous)
        with (migrations / "migrations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "label": label,
                "old_sha256": old_hash,
                "new_sha256": new_hash,
                "reason": "pretraining formal-V1 protocol correction",
                "migrated_at": datetime.now(timezone.utc).isoformat(),
            }, ensure_ascii=False, sort_keys=True) + "\n")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return "migrated" if existed else "created"


def validate_acceptance_evidence(project, evidence_path):
    evidence_path = Path(evidence_path).resolve()
    checks = []

    def add(name, passed, details=""):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    if not evidence_path.is_file():
        add("acceptance_evidence_exists", False, str(evidence_path))
        return checks, {}
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    add("acceptance_evidence_exists", True, str(evidence_path))
    add("acceptance_evidence_ready", evidence.get("ready") is True, evidence.get("checks", []))
    add("acceptance_test_isolation", evidence.get("test_evaluated") is False, evidence.get("test_evaluated"))
    ok, commit = command_ok(["git", "-C", str(project), "rev-parse", "HEAD"])
    add(
        "acceptance_commit",
        ok and evidence.get("project_commit") == commit,
        {"expected": evidence.get("project_commit"), "actual": commit},
    )
    for relative, expected in (evidence.get("critical_files") or {}).items():
        path = project / relative
        actual = sha256(path) if path.is_file() else "missing"
        add(
            "acceptance_critical_file:{}".format(relative),
            actual == expected,
            {"expected": expected, "actual": actual},
        )
    for raw_path, expected in (evidence.get("artifacts") or {}).items():
        path = Path(raw_path)
        actual = sha256(path) if path.is_file() else "missing"
        add(
            "acceptance_artifact:{}".format(path.name),
            actual == expected,
            {"path": str(path), "expected": expected, "actual": actual},
        )
    return checks, evidence


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
    llm_config = Path(config["llm_config"])
    if llm_config.is_file():
        required_variables = required_environment_variables(llm_config)
        check("llm_environment_contract", bool(required_variables), required_variables)
        for variable in required_variables:
            check(
                "llm_environment:{}".format(variable),
                bool(os.environ.get(variable)),
                "set" if os.environ.get(variable) else "missing",
            )
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

    acceptance_checks, acceptance = validate_acceptance_evidence(project, config["acceptance_evidence"])
    checks.extend(acceptance_checks)
    configured_budget = float(config["gpu_budget_hours"])
    configured_fallback = float(config["fallback_seconds_per_step"])
    check(
        "formal_gpu_budget",
        abs(float(os.environ.get("NAS_GPU_BUDGET_HOURS", "nan")) - configured_budget) < 1.0e-12,
        {"configured_hours": configured_budget, "environment": os.environ.get("NAS_GPU_BUDGET_HOURS", "missing")},
    )
    check(
        "formal_step_cost_fallback",
        abs(float(os.environ.get("NAS_FALLBACK_SECONDS_PER_STEP", "nan")) - configured_fallback) < 1.0e-12,
        {"configured_seconds": configured_fallback, "environment": os.environ.get("NAS_FALLBACK_SECONDS_PER_STEP", "missing")},
    )
    expected_ledger = run_root / "budget_ledger.jsonl"
    check(
        "formal_run_budget_ledger",
        Path(os.environ.get("NAS_BUDGET_LEDGER", "")).resolve() == expected_ledger.resolve(),
        {"expected": str(expected_ledger), "actual": os.environ.get("NAS_BUDGET_LEDGER", "missing")},
    )

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
    write_frozen_material(initial_program, dumps_program(parent), run_root, "initial_program")
    frozen_config = run_root / "formal_v1_config.json"
    write_frozen_material(
        frozen_config,
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        run_root,
        "formal_v1_config",
    )
    existing_protocol = json.loads((run_root / "protocol.json").read_text(encoding="utf-8")) if (run_root / "protocol.json").is_file() else {}
    protocol = {
        "version": config["version"],
        "config_sha256": sha256(frozen_config),
        "task_contract_sha256": sha256(config["task_contract"]),
        "quarter_subset_sha256": sha256(config["quarter_subset_file"]),
        "initial_program_sha256": sha256(initial_program),
        "test_during_search": False,
        "gpu_budget_hours": configured_budget,
        "fallback_seconds_per_step": configured_fallback,
        "budget_ledger": str(expected_ledger),
        "created_at": existing_protocol.get("created_at", datetime.now(timezone.utc).isoformat()),
    }
    write_frozen_material(
        run_root / "protocol.json",
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True),
        run_root,
        "protocol",
    )
    ready = {
        "ready": all(item["passed"] for item in checks),
        "formal_v1_version": config["version"],
        "checks": checks,
        "capability_profile": profile.to_dict(),
        "factor_compile_results": factor_results,
        "initial_program": str(initial_program),
        "protocol": protocol,
        "acceptance_evidence": {
            "path": config["acceptance_evidence"],
            "sha256": sha256(config["acceptance_evidence"])
            if Path(config["acceptance_evidence"]).is_file()
            else "",
            "project_commit": acceptance.get("project_commit", ""),
            "test_summary": acceptance.get("test_summary", {}),
            "calibration": acceptance.get("calibration", {}),
            "smoke_factors": acceptance.get("smoke_factors", []),
        },
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
