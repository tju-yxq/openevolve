#!/usr/bin/env python
"""Collect and verify the A100 evidence required to launch formal DSL V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def project_commit(project):
    return subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip()


def tracked_dirty(project):
    output = subprocess.check_output(
        ["git", "-C", str(project), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    return bool(output.strip())


def candidate_rows(summary):
    return [(candidate, candidate.get("result") or {}) for candidate in summary.get("candidates", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sources_path = Path(args.sources).resolve()
    sources = read_json(sources_path)
    project = Path(sources["project_root"]).resolve()
    checks = []
    artifacts = {}

    def check(name, passed, details=""):
        checks.append({"name": name, "passed": bool(passed), "details": details})

    def bind(path):
        path = Path(path).resolve()
        if path.is_file():
            artifacts[str(path)] = sha256(path)
            return True
        return False

    commit = project_commit(project)
    bind(sources_path)
    check("project_git_clean", not tracked_dirty(project), commit)
    critical_files = {}
    for relative in sources["critical_files"]:
        path = project / relative
        exists = bind(path)
        check("critical_file:{}".format(relative), exists, str(path))
        if exists:
            critical_files[relative] = sha256(path)

    pytest_log = Path(sources["pytest_log"])
    pytest_text = pytest_log.read_text(encoding="utf-8", errors="replace") if bind(pytest_log) else ""
    match = re.search(r"(\d+) passed(?:, (\d+) skipped)?", pytest_text)
    passed = int(match.group(1)) if match else 0
    skipped = int(match.group(2) or 0) if match else 0
    check(
        "a100_pytest",
        bool(match) and passed >= int(sources["minimum_pytest_passed"]) and skipped <= int(sources["maximum_pytest_skipped"]),
        {"passed": passed, "skipped": skipped, "log": str(pytest_log)},
    )

    protocol = read_json(sources["symmetry_protocol"])
    bind(sources["symmetry_protocol"])
    observed_relative = []
    observed_absolute = []
    calibration_seeds = []
    for path in sources["calibration_summaries"]:
        summary = read_json(path)
        bind(path)
        parents = [(candidate, result) for candidate, result in candidate_rows(summary) if candidate.get("key") == "parent"]
        valid = len(parents) == 1
        if valid:
            _candidate, result = parents[0]
            report = result.get("pretrain_symmetry_report") or {}
            valid = (
                summary.get("status") == "completed"
                and summary.get("test_evaluated") is False
                and result.get("valid") is True
                and result.get("test_evaluated") is False
                and report.get("protocol_version") in sources["accepted_calibration_protocol_versions"]
                and int(report.get("molecule_count", 0)) >= int(protocol["molecule_count"])
                and len(report.get("errors") or {})
                >= int(protocol["random_rotations"])
                + len(protocol["translations"])
                + len(protocol["permutations"])
            )
            if valid:
                calibration_seeds.append(int(summary["seed"]))
                observed_relative.append(float(result["pretrain_max_symmetry_error"]))
                observed_absolute.append(float(result["pretrain_max_absolute_symmetry_error"]))
        check("calibration:{}".format(Path(path).parent.name), valid, str(path))
    check("calibration_seed_set", sorted(calibration_seeds) == sorted(protocol["calibration"]["seeds"]), calibration_seeds)
    check(
        "calibration_relative_bound",
        observed_relative and max(observed_relative) <= float(protocol["calibration"]["observed_max_relative_error"]),
        max(observed_relative) if observed_relative else None,
    )
    check(
        "calibration_absolute_bound",
        observed_absolute and max(observed_absolute) <= float(protocol["calibration"]["observed_max_absolute_error"]),
        max(observed_absolute) if observed_absolute else None,
    )

    covered = set()
    smoke_rows = []
    for path in sources["smoke_training_summaries"]:
        summary = read_json(path)
        bind(path)
        summary_valid = summary.get("status") == "completed" and summary.get("test_evaluated") is False
        for candidate, result in candidate_rows(summary):
            factor = candidate.get("factor_id") or "parent"
            checkpoint = Path(str(result.get("checkpoint_last", "")))
            row_valid = (
                summary_valid
                and result.get("valid") is True
                and result.get("test_evaluated") is False
                and int(result.get("endpoint_step", 0)) >= int(sources["minimum_smoke_steps"])
                and checkpoint.is_file()
            )
            smoke_rows.append({"factor": factor, "valid": row_valid, "summary": str(path)})
            if row_valid:
                covered.add(factor)
        check("smoke_summary:{}".format(Path(path).parent.name), summary_valid, str(path))
    check("smoke_factor_coverage", set(sources["required_smoke_factors"]) <= covered, sorted(covered))
    check("smoke_all_rows_valid", bool(smoke_rows) and all(row["valid"] for row in smoke_rows), smoke_rows)

    option_admission = sources.get("option_admission") or {}
    option_summaries = {}

    def option_result(spec):
        path = str(spec["summary"])
        if path not in option_summaries:
            option_summaries[path] = read_json(path)
            bind(path)
        return (option_summaries[path].get("results") or {}).get(str(spec["key"])) or {}

    admitted_options = []
    for spec in option_admission.get("accepted", []):
        result = option_result(spec)
        checkpoint = Path(str(result.get("checkpoint_last", "")))
        valid = (
            result.get("valid") is True
            and result.get("test_evaluated") is False
            and int(result.get("endpoint_step", 0)) >= int(sources["minimum_smoke_steps"])
            and checkpoint.is_file()
            and float(result.get("max_symmetry_error", float("inf")))
            <= float(protocol["relative_error_threshold"])
            and float(result.get("max_absolute_symmetry_error", float("inf")))
            <= float(protocol["absolute_error_threshold"])
        )
        details = dict(spec, valid=valid, checkpoint=str(checkpoint))
        admitted_options.append(details)
        check("option_admission:accepted:{}".format(spec["key"]), valid, details)

    rejected_options = []
    for spec in option_admission.get("rejected", []):
        result = option_result(spec)
        error = str(result.get("error", ""))
        valid = (
            result.get("valid") is False
            and result.get("test_evaluated") is False
            and str(spec.get("error_contains", "")) in error
            and not result.get("checkpoint_last")
        )
        details = dict(spec, valid=valid, error=error)
        rejected_options.append(details)
        check("option_admission:rejected:{}".format(spec["key"]), valid, details)

    recovery = sources["recovery"]
    recovery_summary = read_json(recovery["summary"])
    recovery_progress = read_json(recovery["progress"])
    for path in recovery.values():
        bind(path)
    pre_hashes = Path(recovery["pre_hashes"]).read_text(encoding="utf-8").splitlines()
    post_hashes = Path(recovery["post_hashes"]).read_text(encoding="utf-8").splitlines()
    pre = {Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0] for line in pre_hashes if line.strip()}
    post = {Path(line.split(maxsplit=1)[1]).name: line.split(maxsplit=1)[0] for line in post_hashes if line.strip()}
    identities = ("runtime_manifest.json", "protocol.json", "architecture.dsl.json")
    recovery_valid = (
        recovery_summary.get("status") == "completed"
        and recovery_summary.get("test_evaluated") is False
        and int(recovery_progress.get("start_global_step", -1)) == int(sources["recovery_start_step"])
        and int(recovery_progress.get("global_step", -1)) == int(sources["recovery_end_step"])
        and int(recovery_progress.get("steps_executed_current_job", -1))
        == int(sources["recovery_end_step"]) - int(sources["recovery_start_step"])
        and all(pre.get(name) == post.get(name) for name in identities)
        and pre.get("checkpoint_last.pth") != post.get("checkpoint_last.pth")
    )
    check("checkpoint_recovery", recovery_valid, {"pre": pre, "post": post, "progress": recovery_progress})

    supervisor_log = Path(sources["supervisor_log"])
    supervisor_ready = read_json(sources["supervisor_ready"])
    supervisor_text = supervisor_log.read_text(encoding="utf-8", errors="replace") if bind(supervisor_log) else ""
    bind(sources["supervisor_ready"])
    check(
        "supervisor_empty_run",
        supervisor_ready.get("ready") is True and "launcher exited 0" in supervisor_text,
        {"ready": supervisor_ready.get("ready"), "log": str(supervisor_log)},
    )

    evidence = {
        "schema_version": "formal-v1-acceptance@1",
        "created_at": now(),
        "ready": all(item["passed"] for item in checks),
        "project_root": str(project),
        "project_commit": commit,
        "sources_sha256": sha256(sources_path),
        "critical_files": critical_files,
        "artifacts": artifacts,
        "test_summary": {"passed": passed, "skipped": skipped},
        "calibration": {
            "seeds": sorted(calibration_seeds),
            "maximum_relative_error": max(observed_relative) if observed_relative else None,
            "maximum_absolute_error": max(observed_absolute) if observed_absolute else None,
        },
        "smoke_factors": sorted(covered),
        "option_admission": {
            "accepted": admitted_options,
            "rejected": rejected_options,
        },
        "test_evaluated": False,
        "checks": checks,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"ready": evidence["ready"], "output": str(output), "commit": commit}, ensure_ascii=False))
    if not evidence["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
