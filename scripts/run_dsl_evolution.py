#!/usr/bin/env python
"""Run resumable OpenEvolve population search over typed DSL programs."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl import (
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    Compiler,
    DSLGenerationEngine,
    EvidenceItem,
    EvidenceStore,
    LanguageVersion,
    LanguageEvolutionPreregistration,
    MotifDiscoveryPolicy,
    core_registry,
    reference_motif_registry,
    select_active_vocabulary,
    strict_rewrite_registry_hash,
    validate_region_transition,
    v1_region_registry,
)
from equivariant_nas.dsl.serialization import (
    dumps_program,
    load_program,
    load_task_contract,
    loads_program,
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _append_jsonl(path, payload):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _valid_candidate_ids(evolution_path):
    path = Path(evolution_path)
    if not path.exists():
        return set()
    identifiers = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("iteration", 0)) <= 0:
            continue
        metrics = record.get("metrics") or {}
        if not metrics.get("valid") or metrics.get("test_evaluated") is not False:
            continue
        architecture_id = str(metrics.get("architecture_id", record.get("architecture_id", "")))
        if architecture_id:
            identifiers.add(architecture_id)
    return identifiers


def _valid_factor_counts(evolution_path):
    path = Path(evolution_path)
    counts = {}
    seen = set()
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("iteration", 0)) <= 0:
            continue
        metrics = record.get("metrics") or {}
        architecture_id = str(metrics.get("architecture_id", record.get("architecture_id", "")))
        factor_id = str((record.get("region_audit") or {}).get("factor_id", ""))
        if (
            not architecture_id
            or architecture_id in seen
            or not factor_id
            or not metrics.get("valid")
            or metrics.get("test_evaluated") is not False
        ):
            continue
        seen.add(architecture_id)
        counts[factor_id] = counts.get(factor_id, 0) + 1
    return counts


def _used_factor_patch_signatures(evolution_path, factor_id):
    """Return valid root-isolated constructor choices already charged to one factor quota."""

    path = Path(evolution_path)
    signatures = []
    seen = set()
    if not path.exists() or not factor_id:
        return signatures
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics") or {}
        if (
            int(record.get("iteration", 0)) <= 0
            or not metrics.get("valid")
            or metrics.get("test_evaluated") is not False
            or str((record.get("region_audit") or {}).get("factor_id", "")) != str(factor_id)
        ):
            continue
        edits = []
        for edit in (record.get("patch") or {}).get("edits", []):
            if edit.get("kind") != "change_parameters":
                continue
            payload = edit.get("payload") or {}
            if set(payload) != {"value"}:
                continue
            edits.append({"target": str(edit.get("target", "")), "value": payload["value"]})
        edits.sort(key=lambda item: (item["target"], json.dumps(item["value"], sort_keys=True)))
        if not edits:
            continue
        canonical = json.dumps(edits, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if canonical in seen:
            continue
        seen.add(canonical)
        signatures.append({
            "architecture_id": str(metrics.get("architecture_id", record.get("architecture_id", ""))),
            "edits": edits,
        })
    return signatures


def _next_forced_factor(forced_factors, factor_counts, per_factor_target, iteration):
    ordered = tuple(dict.fromkeys(forced_factors))
    if not ordered:
        return ""
    if per_factor_target <= 0:
        return forced_factors[(iteration - 1) % len(forced_factors)]
    start = (iteration - 1) % len(ordered)
    for offset in range(len(ordered)):
        factor_id = ordered[(start + offset) % len(ordered)]
        if int(factor_counts.get(factor_id, 0)) < per_factor_target:
            return factor_id
    return ""


def _unique_initial_parent(programs):
    values = programs.values() if hasattr(programs, "values") else programs
    parents = [program for program in values if int(program.iteration_found) == 0]
    if len(parents) != 1:
        raise RuntimeError(
            "formal root-isolated coverage requires exactly one iteration-0 parent; found {}".format(len(parents))
        )
    return parents[0]


def _validate_root_isolated_factor_transition(initial_program, child_program, factor_id):
    regions = [region for region in v1_region_registry(initial_program) if region.factor_id == factor_id]
    if len(regions) != 1:
        raise RuntimeError(
            "formal root-isolated coverage requires exactly one region for factor {}; found {}".format(
                factor_id,
                len(regions),
            )
        )
    return validate_region_transition(initial_program, child_program, regions[0])


def _load_verified_initial_metrics(path, *, architecture_id, seed, max_steps):
    metrics = json.loads(Path(path).read_text(encoding="utf-8"))
    required_identity = ("program_id", "executable_id", "protocol_id", "runtime_manifest_sha256")
    missing = [key for key in required_identity if not metrics.get(key)]
    if missing:
        raise ValueError("initial metrics lack formal V1 identity fields: {}".format(missing))
    if metrics.get("test_evaluated") is not False:
        raise ValueError("initial metrics must explicitly prove test_evaluated=false")
    if metrics["program_id"] != architecture_id:
        raise ValueError("initial metrics program_id does not match the imported parent")
    if int(metrics.get("seed", -1)) != int(seed):
        raise ValueError("initial metrics seed does not match the search protocol")
    endpoint = int(metrics.get("endpoint_step", metrics.get("fidelity_steps", -1)))
    if endpoint != int(max_steps):
        raise ValueError("initial metrics fidelity does not match the search protocol")
    return metrics


def _ensure_compiler_manifest(output, payload, *, database_has_programs):
    path = Path(output) / "compiler_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            upgraded = json.loads(json.dumps(existing))
            existing_protocol = upgraded.get("formal_v1_search_protocol") or {}
            requested_protocol = payload.get("formal_v1_search_protocol") or {}
            if "valid_per_factor_target" not in existing_protocol:
                existing_protocol["valid_per_factor_target"] = requested_protocol.get("valid_per_factor_target", 0)
            for key in ("train_subset_sha256", "eval_interval_data_epochs"):
                if key not in existing_protocol:
                    existing_protocol[key] = requested_protocol.get(key)
            if upgraded == payload and not _valid_candidate_ids(Path(output) / "evolution.jsonl"):
                _write_json(path, payload)
                return path
            if _is_exact_constructor_capability_label_fix(existing, payload):
                valid_records = _valid_candidate_records(Path(output) / "evolution.jsonl")
                incompatible = [
                    record
                    for record in valid_records
                    if (record.get("region_audit") or {}).get("factor_id") != "F6.3"
                    or (record.get("metrics") or {}).get("lowering_mode") != "exact_hybrid"
                ]
                if incompatible:
                    raise RuntimeError(
                        "exact-constructor capability-label migration found a valid candidate "
                        "whose admission semantics could have changed"
                    )
                old_bytes = path.read_bytes()
                old_hash = hashlib.sha256(old_bytes).hexdigest()
                new_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                new_hash = hashlib.sha256(new_bytes).hexdigest()
                archive = Path(output) / "compiler_manifest_migrations"
                archive.mkdir(exist_ok=True)
                previous = archive / "compiler_manifest_{}.previous.json".format(old_hash[:12])
                if not previous.exists():
                    previous.write_bytes(old_bytes)
                _append_jsonl(
                    archive / "migrations.jsonl",
                    {
                        "kind": "exact_constructor_capability_label_fix",
                        "old_sha256": old_hash,
                        "new_sha256": new_hash,
                        "valid_candidates_preserved": [
                            (record.get("metrics") or {}).get("architecture_id", "")
                            for record in valid_records
                        ],
                        "reason": (
                            "Region capability used the stale label exact_v1_constructor while "
                            "the certified compiler lowering mode is exact_constructor"
                        ),
                        "migrated_at": _now(),
                    },
                )
                _write_json(path, payload)
                return path
            if _is_formal_root_parent_policy_fix(existing, payload):
                valid_records = _valid_candidate_records(Path(output) / "evolution.jsonl")
                root_architecture_id = _initial_architecture_id(Path(output) / "evolution.jsonl")
                incompatible = [
                    record
                    for record in valid_records
                    if str(record.get("parent_architecture_id", "")) != root_architecture_id
                ]
                if not root_architecture_id or incompatible:
                    raise RuntimeError(
                        "root-isolated parent-policy migration found a valid candidate that was not generated "
                        "directly from the iteration-0 parent"
                    )
                old_bytes = path.read_bytes()
                old_hash = hashlib.sha256(old_bytes).hexdigest()
                new_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                new_hash = hashlib.sha256(new_bytes).hexdigest()
                archive = Path(output) / "compiler_manifest_migrations"
                archive.mkdir(exist_ok=True)
                previous = archive / "compiler_manifest_{}.previous.json".format(old_hash[:12])
                if not previous.exists():
                    previous.write_bytes(old_bytes)
                _append_jsonl(
                    archive / "migrations.jsonl",
                    {
                        "kind": "formal_root_isolated_parent_policy_fix",
                        "old_sha256": old_hash,
                        "new_sha256": new_hash,
                        "root_architecture_id": root_architecture_id,
                        "valid_candidates_preserved": [
                            (record.get("metrics") or {}).get("architecture_id", "")
                            for record in valid_records
                        ],
                        "reason": (
                            "The forced four-factor coverage cohort must contain candidates that differ from "
                            "the official iteration-0 parent in exactly the routed factor; sampling an evolved "
                            "parent silently admitted multi-factor candidates into single-factor quotas"
                        ),
                        "migrated_at": _now(),
                    },
                )
                _write_json(path, payload)
                return path
            if _is_certified_option_expansion(existing, payload):
                valid_records = _valid_candidate_records(Path(output) / "evolution.jsonl")
                old_bytes = path.read_bytes()
                old_hash = hashlib.sha256(old_bytes).hexdigest()
                new_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                new_hash = hashlib.sha256(new_bytes).hexdigest()
                archive = Path(output) / "compiler_manifest_migrations"
                archive.mkdir(exist_ok=True)
                previous = archive / "compiler_manifest_{}.previous.json".format(old_hash[:12])
                if not previous.exists():
                    previous.write_bytes(old_bytes)
                _append_jsonl(
                    archive / "migrations.jsonl",
                    {
                        "kind": "certified_formal_option_expansion",
                        "old_sha256": old_hash,
                        "new_sha256": new_hash,
                        "valid_candidates_preserved": [
                            (record.get("metrics") or {}).get("architecture_id", "")
                            for record in valid_records
                        ],
                        "reason": (
                            "Additional root-isolated F2.2 and F5.3 constructor options passed the frozen "
                            "300-step A100 admission protocol and are required to make the preregistered "
                            "two-unique-candidates-per-factor cohort reachable"
                        ),
                        "migrated_at": _now(),
                    },
                )
                _write_json(path, payload)
                return path
            if _is_generation_attempt_extension(existing, payload):
                evolution_path = Path(output) / "evolution.jsonl"
                summary_path = Path(output) / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
                old_attempts = int(existing["formal_v1_search_protocol"]["maximum_generation_attempts"])
                new_attempts = int(payload["formal_v1_search_protocol"]["maximum_generation_attempts"])
                if summary.get("status") != "generation_attempts_exhausted":
                    raise RuntimeError("generation-attempt extension requires an exhausted search summary")
                if int(summary.get("last_completed_iteration", -1)) < old_attempts:
                    raise RuntimeError("generation-attempt extension requires all original attempts to be recorded")
                valid_records = _valid_candidate_records(evolution_path)
                target = int(payload["formal_v1_search_protocol"].get("valid_candidate_target", 0))
                if len(valid_records) >= target:
                    raise RuntimeError("generation-attempt extension is unnecessary because the cohort is already complete")
                old_bytes = path.read_bytes()
                old_hash = hashlib.sha256(old_bytes).hexdigest()
                new_bytes = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
                new_hash = hashlib.sha256(new_bytes).hexdigest()
                archive = Path(output) / "compiler_manifest_migrations"
                archive.mkdir(exist_ok=True)
                previous = archive / "compiler_manifest_{}.previous.json".format(old_hash[:12])
                if not previous.exists():
                    previous.write_bytes(old_bytes)
                failures = {}
                if evolution_path.exists():
                    for line in evolution_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        code = str(record.get("error", "")).split(":", 1)[0]
                        if code:
                            failures[code] = failures.get(code, 0) + 1
                _append_jsonl(
                    archive / "migrations.jsonl",
                    {
                        "kind": "formal_generation_attempt_extension",
                        "old_sha256": old_hash,
                        "new_sha256": new_hash,
                        "old_maximum_generation_attempts": old_attempts,
                        "new_maximum_generation_attempts": new_attempts,
                        "valid_candidate_count": len(valid_records),
                        "valid_candidate_target": target,
                        "failure_code_counts": failures,
                        "training_protocol_changed": False,
                        "selection_protocol_changed": False,
                        "test_evaluated": False,
                        "reason": (
                            "The preregistered generation attempts were exhausted before the immutable "
                            "four-factor coverage target was reached; additional attempts use the same root "
                            "parent, factors, options, training budget, and validation-only admission rules"
                        ),
                        "migrated_at": _now(),
                    },
                )
                _write_json(path, payload)
                return path
            raise RuntimeError(
                "compiler manifest mismatch; resume requires the exact task, language, rewrite registry, compiler, and backend semantics"
            )
        return path
    if database_has_programs:
        raise RuntimeError(
            "existing OpenEvolve database has no compiler_manifest.json; refuse to reinterpret a legacy search under new semantics"
        )
    _write_json(path, payload)
    return path


def _valid_candidate_records(evolution_path):
    path = Path(evolution_path)
    if not path.exists():
        return []
    records = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("iteration", 0)) <= 0:
            continue
        metrics = record.get("metrics") or {}
        architecture_id = str(metrics.get("architecture_id", record.get("architecture_id", "")))
        if (
            not architecture_id
            or architecture_id in seen
            or metrics.get("valid") is not True
            or metrics.get("test_evaluated") is not False
        ):
            continue
        seen.add(architecture_id)
        records.append(record)
    return records


def _initial_architecture_id(evolution_path):
    path = Path(evolution_path)
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if int(record.get("iteration", -1)) != 0:
            continue
        metrics = record.get("metrics") or {}
        architecture_id = str(metrics.get("architecture_id", ""))
        if architecture_id:
            return architecture_id
    return ""


def _is_formal_root_parent_policy_fix(existing, requested):
    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(requested))
    old_protocol = dict(old.get("formal_v1_search_protocol") or {})
    new_protocol = dict(new.get("formal_v1_search_protocol") or {})
    if old_protocol.get("forced_factor_parent_policy") is not None:
        return False
    if new_protocol.pop("forced_factor_parent_policy", None) != "root_isolated":
        return False
    old["formal_v1_search_protocol"] = old_protocol
    new["formal_v1_search_protocol"] = new_protocol
    return old == new


def _is_generation_attempt_extension(existing, requested):
    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(requested))
    old_protocol = dict(old.get("formal_v1_search_protocol") or {})
    new_protocol = dict(new.get("formal_v1_search_protocol") or {})
    old_attempts = old_protocol.pop("maximum_generation_attempts", None)
    new_attempts = new_protocol.pop("maximum_generation_attempts", None)
    if not isinstance(old_attempts, int) or not isinstance(new_attempts, int):
        return False
    if new_attempts <= old_attempts:
        return False
    old["formal_v1_search_protocol"] = old_protocol
    new["formal_v1_search_protocol"] = new_protocol
    return old == new


def _canonical_items(values):
    return {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values}


def _is_certified_option_expansion(existing, requested):
    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(requested))
    old_regions = old.pop("region_registry", None)
    new_regions = new.pop("region_registry", None)
    if old != new or not isinstance(old_regions, list) or not isinstance(new_regions, list):
        return False
    old_by_factor = {str(item.get("factor_id", "")): item for item in old_regions}
    new_by_factor = {str(item.get("factor_id", "")): item for item in new_regions}
    if set(old_by_factor) != set(new_by_factor):
        return False
    changed = set()
    for factor_id in old_by_factor:
        old_region = dict(old_by_factor[factor_id])
        new_region = dict(new_by_factor[factor_id])
        old_values = old_region.pop("allowed_parameter_values", {})
        new_values = new_region.pop("allowed_parameter_values", {})
        old_combinations = old_region.pop("allowed_parameter_combinations", [])
        new_combinations = new_region.pop("allowed_parameter_combinations", [])
        if old_region != new_region:
            return False
        if factor_id not in {"F2.2", "F5.3"}:
            if old_values != new_values or old_combinations != new_combinations:
                return False
            continue
        if set(old_values) != set(new_values):
            return False
        if any(not _canonical_items(values) <= _canonical_items(new_values[path]) for path, values in old_values.items()):
            return False
        if not _canonical_items(old_combinations) <= _canonical_items(new_combinations):
            return False
        if old_values != new_values or old_combinations != new_combinations:
            changed.add(factor_id)
    return changed == {"F2.2", "F5.3"}


def _is_exact_constructor_capability_label_fix(existing, requested):
    old = json.loads(json.dumps(existing))
    new = json.loads(json.dumps(requested))
    old_regions = old.pop("region_registry", None)
    new_regions = new.pop("region_registry", None)
    if old != new or not isinstance(old_regions, list) or not isinstance(new_regions, list):
        return False
    if len(old_regions) != len(new_regions):
        return False
    changed = set()
    for old_region, new_region in zip(old_regions, new_regions):
        old_item = dict(old_region)
        new_item = dict(new_region)
        old_capability = old_item.pop("backend_capability", None)
        new_capability = new_item.pop("backend_capability", None)
        if old_item != new_item:
            return False
        factor_id = str(new_item.get("factor_id", ""))
        if old_capability == new_capability:
            continue
        if (
            factor_id not in {"F2.2", "F4.4", "F5.3"}
            or old_capability != "exact_v1_constructor"
            or new_capability != "exact_constructor"
        ):
            return False
        changed.add(factor_id)
    return changed == {"F2.2", "F4.4", "F5.3"}


def _ensure_language_preregistration(output, preregistration, *, database_has_programs):
    path = Path(output) / "language_boundary_preregistration.json"
    payload = dict(preregistration.to_dict(), preregistration_hash=preregistration.content_hash())
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("language-boundary preregistration mismatch; a running cycle cannot change its language policy")
        return path
    if database_has_programs:
        raise RuntimeError("existing OpenEvolve cycle has no language-boundary preregistration")
    _write_json(path, payload)
    return path


def _lineage_root(program, programs):
    current = program
    seen = set()
    while getattr(current, "parent_id", None) and current.parent_id in programs and current.id not in seen:
        seen.add(current.id)
        current = programs[current.parent_id]
    metadata = dict(getattr(program, "metadata", {}) or {})
    island = metadata.get("island_id", metadata.get("island"))
    return "island:{}".format(island) if island is not None else "root:{}".format(current.id)


def _write_language_cycle_snapshot(database, output, compiler, task, language, preregistration, holdout_modulus):
    target = Path(output) / "language_cycle_candidates"
    target.mkdir(exist_ok=True)
    records = {}
    programs = database.programs
    for candidate in programs.values():
        metrics = dict(candidate.metrics or {})
        if not metrics.get("valid") or metrics.get("test_evaluated"):
            continue
        try:
            source = loads_program(candidate.code)
            artifact = compiler.analyze(source, task)
        except Exception as exc:
            _append_jsonl(Path(output) / "language_cycle_snapshot_errors.jsonl", {
                "program_id": candidate.id,
                "error_type": type(exc).__name__,
                "error": str(exc)[:4000],
            })
            continue
        reported = str(metrics.get("architecture_id", ""))
        if reported and reported != artifact.architecture_id:
            _append_jsonl(Path(output) / "language_cycle_snapshot_errors.jsonl", {
                "program_id": candidate.id,
                "reported_architecture_id": reported,
                "compiled_architecture_id": artifact.architecture_id,
                "error": "semantic identity mismatch",
            })
            continue
        if artifact.architecture_id in records:
            continue
        assignment = int(hashlib.sha256(
            "{}:{}".format(preregistration.boundary_id, artifact.architecture_id).encode("utf-8")
        ).hexdigest(), 16) % holdout_modulus
        partition = "heldout_replay" if assignment == 0 else "support"
        path = target / "{}.dsl.json".format(artifact.architecture_id)
        path.write_text(dumps_program(source), encoding="utf-8")
        records[artifact.architecture_id] = {
            "architecture_id": artifact.architecture_id,
            "program_id": candidate.id,
            "program_path": str(path),
            "lineage_id": _lineage_root(candidate, programs),
            "task_id": task.task_id,
            "visible_splits": ["train", "validation"],
            "test_evaluated": False,
            "discovery_partition": partition,
            "language_registry_hash": language.registry_hash(),
            "rewrite_registry_hash": artifact.rewrite_registry_hash,
        }
    snapshot = {
        "boundary_preregistration": dict(
            preregistration.to_dict(),
            preregistration_hash=preregistration.content_hash(),
        ),
        "closed_at": _now(),
        "selection_rule": preregistration.selection_rule,
        "partition_rule": preregistration.partition_rule,
        "candidate_count": len(records),
        "candidates": [records[key] for key in sorted(records)],
    }
    _write_json(Path(output) / "language_cycle_snapshot.json", snapshot)
    return snapshot


def _measured_evidence(program, inspirations):
    items = []
    for candidate in (program,) + tuple(inspirations):
        metrics = dict(candidate.metrics or {})
        if metrics.get("test_evaluated"):
            continue
        items.append(
            EvidenceItem(
                "openevolve:{}".format(candidate.id),
                "validation",
                "candidate_evaluation",
                {
                    "architecture_id": metrics.get("architecture_id"),
                    "fidelity_steps": metrics.get("fidelity_steps", 0),
                    "validation_alpha_mae": metrics.get("validation_alpha_mae"),
                    "parameter_count": metrics.get("parameter_count"),
                    "step_time_ms": metrics.get("step_time_ms"),
                    "valid": bool(metrics.get("valid")),
                },
            )
        )
    return tuple(items)


def _evaluate(evaluator, candidate_path, expected_id):
    metrics = dict(evaluator.evaluate(str(candidate_path)))
    reported = metrics.get("architecture_id")
    if reported and reported != expected_id:
        return {
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "identity_audit",
            "architecture_id": expected_id,
            "reported_architecture_id": reported,
            "error": "evaluator and compiler architecture IDs disagree",
            "test_evaluated": bool(metrics.get("test_evaluated", False)),
        }
    metrics["architecture_id"] = expected_id
    if metrics.get("test_evaluated"):
        metrics.update({
            "valid": False,
            "combined_score": -1.0e9,
            "failure_stage": "test_leakage",
            "error": "candidate-generation evaluation accessed the test split",
        })
    return metrics


async def run(args):
    sys.path.insert(0, args.openevolve_root)
    sys.path.insert(0, str(Path(args.evaluator_file).resolve().parent))
    from openevolve.config import load_config
    from openevolve.database import Program, ProgramDatabase
    from openevolve.llm.ensemble import LLMEnsemble
    import evaluator as evaluator_adapter

    output = Path(args.output).resolve()
    forced_factors = tuple(item.strip() for item in args.forced_factor_sequence.split(",") if item.strip())
    candidates = output / "candidates"
    output.mkdir(parents=True, exist_ok=True)
    candidates.mkdir(exist_ok=True)
    task_path = Path(args.task_contract).resolve()
    task = load_task_contract(str(task_path))
    initial_program = load_program(args.initial_program)
    if initial_program.task_contract != task.task_id:
        raise ValueError("initial program and task contract IDs disagree")

    primitives = core_registry()
    motifs = reference_motif_registry()
    compiler = Compiler(primitives, motifs)
    language = LanguageVersion.from_registries(
        initial_program.language_version,
        "",
        primitives,
        motifs,
        _now(),
        {"search": "open-evolve-typed-dsl"},
    )
    vocabulary = select_active_vocabulary(language, task.group, primitives, motifs)
    store = EvidenceStore(str(output / "evidence.sqlite"))
    store.register_language(language, motifs)

    config = load_config(args.config)
    config.database.db_path = str(output / "database")
    config.database.in_memory = True
    config.database.random_seed = args.seed
    for model in config.llm.models:
        model.random_seed = args.seed
    database = ProgramDatabase(config.database)
    language_preregistration = None
    if args.language_cycle_index:
        if args.language_holdout_modulus < 2:
            raise ValueError("language holdout modulus must be at least two")
        policy = MotifDiscoveryPolicy()
        boundary_id = args.language_boundary_id or "dsl-cycle-{}".format(args.language_cycle_index)
        preregistration_path = output / "language_boundary_preregistration.json"
        preregistered_at = (
            str(json.loads(preregistration_path.read_text(encoding="utf-8"))["preregistered_at"])
            if preregistration_path.exists()
            else _now()
        )
        language_preregistration = LanguageEvolutionPreregistration(
            boundary_id,
            language.version,
            args.language_cycle_index,
            "all valid compiled test-hidden programs present in the OpenEvolve database at cycle close",
            "sha256(boundary_id:architecture_id) mod {} equals zero is heldout_replay; all others are support".format(args.language_holdout_modulus),
            policy.content_hash(),
            preregistered_at,
        )
        _ensure_language_preregistration(output, language_preregistration, database_has_programs=bool(database.programs))
    compiler_manifest = {
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_semantics_version": BACKEND_SEMANTICS_VERSION,
        "rewrite_registry_hash": strict_rewrite_registry_hash(),
        "language_registry_hash": language.registry_hash(),
        "task_contract_hash": task.content_hash(),
        "language_preregistration_hash": language_preregistration.content_hash() if language_preregistration else "",
        "initial_lowering_plan": compiler.plan_lowering(initial_program, task).to_dict(),
        "region_registry": [item.to_dict() for item in v1_region_registry(initial_program)],
        "formal_v1_search_protocol": {
            "forced_factor_sequence": list(forced_factors),
            "max_steps": int(args.max_steps),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "maximum_generation_attempts": int(args.iterations),
            "valid_candidate_target": int(args.valid_candidate_target),
            "valid_per_factor_target": int(args.valid_per_factor_target),
            "forced_factor_parent_policy": "root_isolated" if forced_factors and args.valid_per_factor_target > 0 else "evolutionary",
            "train_subset_sha256": hashlib.sha256(Path(args.train_subset_file).read_bytes()).hexdigest()
            if args.train_subset_file
            else "",
            "eval_interval_data_epochs": int(args.eval_interval_epochs),
            "test_during_search": False,
        },
    }
    _ensure_compiler_manifest(output, compiler_manifest, database_has_programs=bool(database.programs))
    ensemble = LLMEnsemble(config.llm.models)
    engine = DSLGenerationEngine(
        compiler,
        task,
        vocabulary,
        store,
        model_name=args.model_label,
        repair_attempts=args.repair_attempts,
    )

    os.environ["DSL_TASK_CONTRACT"] = str(task_path)
    os.environ["NAS_MAX_STEPS"] = str(args.max_steps)
    os.environ["NAS_BATCH_SIZE"] = str(args.batch_size)
    os.environ["NAS_SEED"] = str(args.seed)
    os.environ["NAS_SKIP_SYMMETRY"] = "1" if args.skip_symmetry else "0"
    os.environ["NAS_TRAIN_SUBSET_FILE"] = str(Path(args.train_subset_file).resolve()) if args.train_subset_file else ""
    os.environ["NAS_EVAL_INTERVAL_EPOCHS"] = str(args.eval_interval_epochs)
    if args.equiformer_v2_root:
        os.environ["EQUIFORMER_V2_ROOT"] = args.equiformer_v2_root

    if not database.programs:
        artifact = compiler.analyze(initial_program, task)
        store.add_compiled_candidate(artifact, task)
        store.add_compiler_run(
            artifact.architecture_id,
            COMPILER_SEMANTICS_VERSION,
            "success",
            inference=artifact.inference,
            rewrite_trace=artifact.rewrite_trace,
            rewrite_registry_hash=artifact.rewrite_registry_hash,
        )
        initial_path = candidates / "iteration_0000.dsl.json"
        initial_path.write_text(dumps_program(initial_program), encoding="utf-8")
        metrics = (
            _load_verified_initial_metrics(
                args.initial_metrics,
                architecture_id=artifact.architecture_id,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            if args.initial_metrics
            else _evaluate(evaluator_adapter, initial_path, artifact.architecture_id)
        )
        metrics["architecture_id"] = artifact.architecture_id
        initial = Program(
            id=str(uuid.uuid4()),
            code=dumps_program(initial_program),
            language="json",
            metrics=metrics,
            iteration_found=0,
            metadata={"architecture_id": artifact.architecture_id, "language_version": initial_program.language_version},
        )
        database.add(initial, iteration=0)
        database.save(str(output / "database"), iteration=0)
        store.add_evaluation(
            artifact.architecture_id,
            split="validation",
            fidelity_steps=int(metrics.get("fidelity_steps", args.max_steps)),
            seed=args.seed,
            metrics=metrics,
            resources={"charged_gpu_seconds": metrics.get("charged_gpu_seconds", 0.0)},
        )
        _append_jsonl(output / "evolution.jsonl", {"iteration": 0, "kind": "initial", "metrics": metrics})
    else:
        parents = [program for program in database.programs.values() if int(program.iteration_found) == 0]
        if len(parents) == 1 and (parents[0].metrics or {}).get("error_type") == "BudgetExceeded":
            artifact = compiler.analyze(initial_program, task)
            initial_path = candidates / "iteration_0000.dsl.json"
            metrics = _evaluate(evaluator_adapter, initial_path, artifact.architecture_id)
            metrics["architecture_id"] = artifact.architecture_id
            if not metrics.get("valid"):
                raise RuntimeError("initial parent recovery failed: {}".format(metrics.get("error", "unknown error")))
            parents[0].metrics = metrics
            database.save(str(output / "database"), iteration=database.last_iteration)
            store.add_evaluation(
                artifact.architecture_id,
                split="validation",
                fidelity_steps=int(metrics.get("fidelity_steps", args.max_steps)),
                seed=args.seed,
                metrics=metrics,
                resources={"charged_gpu_seconds": metrics.get("charged_gpu_seconds", 0.0)},
            )
            _append_jsonl(
                output / "evolution.jsonl",
                {"iteration": 0, "kind": "initial_recovery", "recovered_from": "BudgetExceeded", "metrics": metrics},
            )

    start = int(database.last_iteration) + 1
    stop_file = Path(args.stop_file).resolve() if args.stop_file else output / "STOP"
    completed = start - 1
    attempt_stop = args.iterations + 1 if args.valid_candidate_target else start + args.iterations
    for iteration in range(start, attempt_stop):
        if stop_file.exists():
            break
        valid_ids = _valid_candidate_ids(output / "evolution.jsonl")
        factor_counts = _valid_factor_counts(output / "evolution.jsonl")
        factor_coverage_reached = (
            not forced_factors
            or args.valid_per_factor_target <= 0
            or all(factor_counts.get(factor_id, 0) >= args.valid_per_factor_target for factor_id in set(forced_factors))
        )
        if args.valid_candidate_target and len(valid_ids) >= args.valid_candidate_target and factor_coverage_reached:
            break
        completed = iteration
        _write_json(output / "heartbeat.json", {
            "status": "proposing",
            "updated_at": _now(),
            "iteration": iteration,
            "last_completed_iteration": iteration - 1,
            "valid_candidate_count": len(valid_ids),
            "valid_candidate_target": int(args.valid_candidate_target),
            "valid_factor_counts": factor_counts,
            "valid_per_factor_target": int(args.valid_per_factor_target),
            "stop_file": str(stop_file),
        })
        island_index = (iteration - 1) % len(database.islands)
        database.set_current_island(island_index)
        current_path = output / "current_candidate.json"
        current_payload = json.loads(current_path.read_text(encoding="utf-8")) if current_path.exists() else None
        record = {"iteration": iteration, "target_island": island_index}
        try:
            if current_payload is not None:
                if int(current_payload.get("iteration", -1)) != iteration:
                    raise RuntimeError("current_candidate iteration does not match the resumable database boundary")
                child_id = str(current_payload["architecture_id"])
                child_path = Path(current_payload["candidate_path"])
                expected_sha = str(current_payload.get("candidate_sha256", ""))
                actual_sha = hashlib.sha256(child_path.read_bytes()).hexdigest()
                if expected_sha and actual_sha != expected_sha:
                    raise RuntimeError("current candidate program hash changed; refuse identity drift")
                child_code = child_path.read_text(encoding="utf-8")
                parent_db_id = str(current_payload["parent_program_id"])
                parent_generation = int(current_payload["parent_generation"])
                generation_metadata = dict(current_payload["generation_metadata"])
                record.update(dict(current_payload["record"]))
                record["recovered_without_llm_call"] = True
                _write_json(output / "heartbeat.json", {
                    "status": "recovering_candidate",
                    "updated_at": _now(),
                    "iteration": iteration,
                    "architecture_id": child_id,
                    "candidate_path": str(child_path),
                })
            else:
                sampled_parent, inspirations = database.sample(num_inspirations=args.inspirations)
                forced_factor_id = _next_forced_factor(
                    forced_factors,
                    factor_counts,
                    args.valid_per_factor_target,
                    iteration,
                )
                root_isolated = bool(forced_factor_id and args.valid_per_factor_target > 0)
                parent = _unique_initial_parent(database.programs) if root_isolated else sampled_parent
                parent_program = loads_program(parent.code)
                regions = v1_region_registry(parent_program)
                record.update({
                    "parent_program_id": parent.id,
                    "parent_architecture_id": parent.metrics.get("architecture_id"),
                    "inspiration_ids": [item.id for item in inspirations],
                    "forced_factor_id": forced_factor_id,
                    "parent_policy": "root_isolated" if root_isolated else "evolutionary",
                })
                generated = await engine.generate_region_candidate(
                    ensemble,
                    parent_program,
                    _measured_evidence(parent, inspirations),
                    regions,
                    forced_factor_id=forced_factor_id,
                    excluded_patch_signatures=_used_factor_patch_signatures(
                        output / "evolution.jsonl",
                        forced_factor_id,
                    ),
                )
                child_id = generated.child.architecture_id
                child_path = candidates / "iteration_{:04d}_{}.dsl.json".format(iteration, child_id)
                child_code = dumps_program(generated.child.source_program)
                child_path.write_text(child_code, encoding="utf-8")
                parent_db_id = parent.id
                parent_generation = parent.generation
                generation_metadata = {
                    "architecture_id": child_id,
                    "patch": generated.patch.to_dict(),
                    "router_response": dict(generated.router_response),
                    "critic_response": dict(generated.critic_response),
                    "region_audit": dict(generated.region_audit),
                    "repair_count": generated.repair_count,
                    "planner_repair_count": generated.planner_repair_count,
                    "critic_repair_count": generated.critic_repair_count,
                }
                record.update(dict(generation_metadata, planner_response=dict(generated.planner_response)))
                _write_json(current_path, {
                    "status": "evaluating",
                    "updated_at": _now(),
                    "iteration": iteration,
                    "architecture_id": child_id,
                    "candidate_path": str(child_path),
                    "candidate_sha256": hashlib.sha256(child_path.read_bytes()).hexdigest(),
                    "parent_program_id": parent_db_id,
                    "parent_generation": parent_generation,
                    "generation_metadata": generation_metadata,
                    "record": record,
                })
            forced_factor_id = str(record.get("forced_factor_id", ""))
            if forced_factor_id and args.valid_per_factor_target > 0:
                root_audit = _validate_root_isolated_factor_transition(
                    initial_program,
                    loads_program(child_code),
                    forced_factor_id,
                )
                record["root_isolation_audit"] = dict(root_audit)
            metrics = _evaluate(evaluator_adapter, child_path, child_id)
            partial_checkpoint = Path(str(metrics.get("checkpoint_last", "")))
            if not metrics.get("valid") and partial_checkpoint.is_file():
                record["same_candidate_retry_from_checkpoint"] = True
                metrics = _evaluate(evaluator_adapter, child_path, child_id)
            record.update({"architecture_id": child_id, "metrics": metrics})
            store.add_evaluation(
                child_id,
                split="validation",
                fidelity_steps=int(metrics.get("fidelity_steps", args.max_steps)),
                seed=args.seed,
                metrics=metrics,
                resources={"charged_gpu_seconds": metrics.get("charged_gpu_seconds", 0.0)},
            )
            if metrics.get("valid"):
                child = Program(
                    id=str(uuid.uuid4()),
                    code=child_code,
                    language="json",
                    parent_id=parent_db_id,
                    generation=parent_generation + 1,
                    metrics=metrics,
                    iteration_found=iteration,
                    metadata=generation_metadata,
                )
                database.add(child, iteration=iteration, target_island=island_index)
        except Exception as exc:
            record.update({"error_type": type(exc).__name__, "error": str(exc)[:4000]})
        _append_jsonl(output / "evolution.jsonl", record)
        database.save(str(output / "database"), iteration=iteration)
        if current_path.exists():
            current_path.unlink()
        best = database.get_best_program()
        valid_ids = _valid_candidate_ids(output / "evolution.jsonl")
        factor_counts = _valid_factor_counts(output / "evolution.jsonl")
        summary = {
            "last_completed_iteration": iteration,
            "program_count": len(database.programs),
            "best_program_id": best.id if best else None,
            "best_metrics": best.metrics if best else None,
            "valid_candidate_count": len(valid_ids),
            "valid_candidate_target": int(args.valid_candidate_target),
            "valid_factor_counts": factor_counts,
            "valid_per_factor_target": int(args.valid_per_factor_target),
            "task_contract_hash": task.content_hash(),
            "language_registry_hash": language.registry_hash(),
            "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
            "rewrite_registry_hash": strict_rewrite_registry_hash(),
            "updated_at": _now(),
        }
        _write_json(output / "summary.json", summary)
        _write_json(output / "heartbeat.json", dict(summary, status="running", stop_file=str(stop_file)))

    best = database.get_best_program()
    valid_ids = _valid_candidate_ids(output / "evolution.jsonl")
    factor_counts = _valid_factor_counts(output / "evolution.jsonl")
    database.save(str(output / "database"), iteration=completed)
    language_snapshot = None
    if language_preregistration is not None:
        language_snapshot = _write_language_cycle_snapshot(
            database,
            output,
            compiler,
            task,
            language,
            language_preregistration,
            args.language_holdout_modulus,
        )
    factor_coverage_reached = (
        not forced_factors
        or args.valid_per_factor_target <= 0
        or all(factor_counts.get(factor_id, 0) >= args.valid_per_factor_target for factor_id in set(forced_factors))
    )
    target_reached = (not args.valid_candidate_target or len(valid_ids) >= args.valid_candidate_target) and factor_coverage_reached
    final = {
        "last_completed_iteration": completed,
        "program_count": len(database.programs),
        "best_program_id": best.id if best else None,
        "best_metrics": best.metrics if best else None,
        "valid_candidate_count": len(valid_ids),
        "valid_candidate_target": int(args.valid_candidate_target),
        "valid_factor_counts": factor_counts,
        "valid_per_factor_target": int(args.valid_per_factor_target),
        "task_contract_hash": task.content_hash(),
        "language_registry_hash": language.registry_hash(),
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "rewrite_registry_hash": strict_rewrite_registry_hash(),
        "status": (
            "stopped"
            if stop_file.exists()
            else "completed"
            if target_reached
            else "generation_attempts_exhausted"
        ),
        "updated_at": _now(),
        "language_cycle_snapshot": str(output / "language_cycle_snapshot.json") if language_snapshot is not None else "",
    }
    _write_json(output / "summary.json", final)
    _write_json(output / "heartbeat.json", dict(final, stop_file=str(stop_file)))
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-program", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--evaluator-file", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--openevolve-root", default="/home/20262202788/openevolve")
    parser.add_argument("--equiformer-v2-root", default="")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Generation attempt limit; with --valid-candidate-target this is an absolute resumable run limit.",
    )
    parser.add_argument(
        "--valid-candidate-target",
        type=int,
        default=0,
        help="Stop early after this many unique valid non-parent candidates; zero preserves fixed-attempt behavior.",
    )
    parser.add_argument(
        "--valid-per-factor-target",
        type=int,
        default=0,
        help="Require this many unique valid candidates for every forced factor before stopping.",
    )
    parser.add_argument("--inspirations", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-subset-file", default="")
    parser.add_argument("--eval-interval-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--model-label", default="openevolve-ensemble")
    parser.add_argument(
        "--forced-factor-sequence",
        default="",
        help="Comma-separated pre-registered factor ids; skips the Router LLM call and cycles by iteration.",
    )
    parser.add_argument("--initial-metrics", default="")
    parser.add_argument("--skip-symmetry", action="store_true")
    parser.add_argument("--stop-file", default="")
    parser.add_argument("--language-cycle-index", type=int, default=0)
    parser.add_argument("--language-boundary-id", default="")
    parser.add_argument("--language-holdout-modulus", type=int, default=5)
    return parser


def main():
    asyncio.run(run(get_parser().parse_args()))


if __name__ == "__main__":
    main()
