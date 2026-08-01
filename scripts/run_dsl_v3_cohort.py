#!/usr/bin/env python
"""Generate one fixed-parent cohort of eight unique V3 Typed DSL candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_V3_ROOT = PROJECT_ROOT.parent / "equiformer_v3_official"

from equivariant_nas.dsl import (
    BACKEND_SEMANTICS_VERSION,
    COMPILER_SEMANTICS_VERSION,
    V3_FIRST_ROUND_MUTATION_VERSION,
    TypeChecker,
    V3MultiFidelityProtocol,
    architecture_id,
    canonicalize,
    core_registry,
    v3_evolution_state,
    v3_program_spec,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend, V3_REFERENCE_COMMIT
from equivariant_nas.dsl.serialization import dumps_program, load_program
from scripts.run_dsl_v3_evolution import (
    _append_jsonl,
    _available_actions,
    _load_mapping,
    _load_seed,
    _now,
    _run_three_stage_round,
    _validate_candidate,
    _write_json,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_protocol_path(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return (config_path.parent / candidate).resolve()


def load_frozen_protocol(path: Path) -> V3MultiFidelityProtocol:
    path = Path(path).resolve()
    source = _load_mapping(path)
    dataset_manifest = _resolve_protocol_path(path, str(source.pop("dataset_manifest")))
    quarter_subset = _resolve_protocol_path(path, str(source.pop("quarter_subset")))
    equivariance_contract = _resolve_protocol_path(path, str(source.pop("equivariance_contract")))
    for required in (dataset_manifest, quarter_subset, equivariance_contract):
        if not required.is_file():
            raise FileNotFoundError(required)
    frozen = {
        **source,
        "dataset_manifest_sha256": _sha256(dataset_manifest),
        "quarter_subset_sha256": _sha256(quarter_subset),
        "equivariance_contract_sha256": _sha256(equivariance_contract),
    }
    protocol = V3MultiFidelityProtocol.from_mapping(frozen)
    if protocol.dataset_id != str(_load_mapping(dataset_manifest).get("dataset_id", "")):
        raise ValueError("protocol dataset_id disagrees with the frozen dataset manifest")
    return protocol


def _candidate_dir(output: Path, candidate_index: int, *, resume: bool) -> Path:
    candidate_dir = output / "candidate_{:03d}".format(candidate_index)
    if candidate_dir.exists():
        if not resume:
            raise RuntimeError("candidate evidence directory already exists: {}".format(candidate_dir))
        failed_root = output / "failed_attempts"
        failed_root.mkdir(parents=True, exist_ok=True)
        attempt = 1
        while True:
            archived = failed_root / "candidate_{:03d}_attempt_{:03d}".format(candidate_index, attempt)
            if not archived.exists():
                candidate_dir.replace(archived)
                break
            attempt += 1
    candidate_dir.mkdir(parents=True, exist_ok=False)
    return candidate_dir


def _read_records(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args):
    output = Path(args.output).resolve()
    protocol = load_frozen_protocol(Path(args.protocol_config))
    if args.cohort_size != protocol.cohort_size:
        raise ValueError("cohort size is frozen by the 8->4->2 protocol")
    parent, parent_source = _load_seed(args)
    registry = core_registry()
    TypeChecker(registry).check(parent)
    parent_id = architecture_id(parent, registry)
    if set(item.name for item in parent.outputs) not in ({"energy"}, {"energy", "forces"}):
        raise ValueError("V3 cohort parent must expose energy or direct Energy+Force outputs")

    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError("output directory is nonempty; pass --resume for the same frozen cohort")
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "cohort.jsonl"
    manifest_path = output / "cohort_manifest.json"
    manifest = {
        "kind": "fixed_parent_v3_candidate_cohort",
        "cycle_index": args.cycle_index,
        "cohort_size": protocol.cohort_size,
        "fixed_parent_architecture_id": parent_id,
        "parent_source": parent_source,
        "protocol": protocol.to_dict(),
        "protocol_hash": protocol.content_hash(),
        "mutation_version": V3_FIRST_ROUND_MUTATION_VERSION,
        "compiler_semantics_version": COMPILER_SEMANTICS_VERSION,
        "backend_semantics_version": BACKEND_SEMANTICS_VERSION,
        "official_v3_commit": V3_REFERENCE_COMMIT,
        "selection_mode": args.selection_mode,
        "model": args.model if args.selection_mode == "glm" else "",
        "credential_serialized": False,
        "validation_level": args.validation_level,
        "equiformer_v3_root": str(Path(args.equiformer_v3_root).resolve()),
        "workflow": [
            "fixed_best_parent",
            "generate_8_unique_siblings",
            "quarter_8k_all_8",
            "quarter_80k_top_4",
            "full_250k_top_2",
            "validation_winner_becomes_next_cycle_parent",
        ],
        "created_at": _now(),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        frozen_keys = (
            "cycle_index",
            "cohort_size",
            "fixed_parent_architecture_id",
            "protocol_hash",
            "mutation_version",
            "compiler_semantics_version",
            "backend_semantics_version",
            "official_v3_commit",
            "selection_mode",
            "model",
            "validation_level",
            "equiformer_v3_root",
        )
        changed = {key: (existing.get(key), manifest.get(key)) for key in frozen_keys if existing.get(key) != manifest.get(key)}
        if changed:
            raise RuntimeError("resume changed the frozen cohort identity: {}".format(changed))
        manifest = existing
    _write_json(manifest_path, manifest)
    parent_path = output / "parent.dsl.json"
    if not parent_path.exists():
        parent_path.write_text(dumps_program(parent), encoding="utf-8")

    records = _read_records(records_path)
    completed = len(records)
    if completed > protocol.cohort_size:
        raise RuntimeError("cohort contains more records than its frozen size")
    seen_states = [v3_evolution_state(parent)] + [str(item["state"]) for item in records]
    seen_ids = {parent_id} | {str(item["architecture_id"]) for item in records}
    backend = E3NNGraphBackend(registry, equiformer_v3_root=args.equiformer_v3_root)

    for candidate_index in range(completed + 1, protocol.cohort_size + 1):
        actions = _available_actions(parent, seen_states)
        if not actions:
            raise RuntimeError("fixed parent cannot produce eight unseen certified V3 siblings")
        candidate_dir = _candidate_dir(output, candidate_index, resume=args.resume)
        workflow = _run_three_stage_round(
            args,
            parent,
            actions,
            records,
            seen_states,
            candidate_index,
            candidate_dir,
            registry,
        )
        child = workflow["child"]
        child_id = architecture_id(child, registry)
        if child_id in seen_ids:
            raise RuntimeError("cohort generator produced a duplicate architecture")
        validation = _validate_candidate(
            child,
            registry,
            backend,
            validation_level=args.validation_level,
            seed=protocol.seed + args.cycle_index * 1000 + candidate_index,
        )
        candidate_path = candidate_dir / "candidate.dsl.json"
        canonical_path = candidate_dir / "candidate.canonical.dsl.json"
        candidate_path.write_text(dumps_program(child), encoding="utf-8")
        canonical_path.write_text(dumps_program(canonicalize(child, registry)), encoding="utf-8")
        _write_json(candidate_dir / "patch.json", workflow["patch"].to_dict())
        _write_json(candidate_dir / "lowering_manifest.json", validation)
        record = {
            "cycle_index": args.cycle_index,
            "candidate_index": candidate_index,
            "parent_architecture_id": parent_id,
            "architecture_id": child_id,
            "official_spec_id": workflow["child_spec"].architecture_id(),
            "action_id": workflow["action"].action_id,
            "mutation_family": workflow["action"].field,
            "factor_id": workflow["router"]["factor_id"],
            "region_id": workflow["router"]["region_id"],
            "hypothesis": dict(workflow["patch"].hypothesis),
            "router_response": workflow["router_response"],
            "critic_response": workflow["critic_response"],
            "synthesizer_response": workflow["synthesizer_response"],
            "critic_repair_count": workflow["critic_repair_count"],
            "compiler_repair_count": workflow["compiler_repair_count"],
            "state": workflow["state"],
            "candidate_path": str(candidate_path),
            "canonical_path": str(canonical_path),
            "patch_path": str(candidate_dir / "patch.json"),
            "lowering_manifest_path": str(candidate_dir / "lowering_manifest.json"),
            "protocol_hash": protocol.content_hash(),
            "dataset_manifest_sha256": protocol.dataset_manifest_sha256,
            "quarter_subset_sha256": protocol.quarter_subset_sha256,
            "equivariance_contract_sha256": protocol.equivariance_contract_sha256,
            "selected_by": args.selection_mode,
            "validation_level": args.validation_level,
            "created_at": _now(),
        }
        _append_jsonl(records_path, record)
        records.append(record)
        seen_states.append(workflow["state"])
        seen_ids.add(child_id)
        _write_json(output / "heartbeat.json", {
            "status": "generation_complete" if candidate_index == protocol.cohort_size else "generating",
            "cycle_index": args.cycle_index,
            "completed_candidates": candidate_index,
            "cohort_size": protocol.cohort_size,
            "fixed_parent_architecture_id": parent_id,
            "updated_at": _now(),
        })

    summary = {
        "status": "cohort_generation_complete",
        "cycle_index": args.cycle_index,
        "fixed_parent_architecture_id": parent_id,
        "candidate_count": len(records),
        "unique_candidate_count": len({item["architecture_id"] for item in records}),
        "all_candidates_are_direct_children_of_fixed_parent": all(
            item["parent_architecture_id"] == parent_id for item in records
        ),
        "protocol_hash": protocol.content_hash(),
        "next_required_stage": "quarter_8k_all_8",
        "next_parent_not_selected_before_250k": True,
        "candidate_records": records,
        "completed_at": _now(),
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "candidate_records"}, ensure_ascii=False, sort_keys=True))
    return summary


def get_parser():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config")
    source.add_argument("--seed-program")
    parser.add_argument("--protocol-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cycle-index", type=int, default=1)
    parser.add_argument("--cohort-size", type=int, default=8)
    parser.add_argument("--selection-mode", choices=("glm", "deterministic"), default="glm")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--api-base", default=os.environ.get("GLM_API_BASE", "https://glm.llm.autos/v1"))
    parser.add_argument("--api-key-env", default="GLM_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--llm-attempts", type=int, default=3)
    parser.add_argument("--allow-deterministic-fallback", action="store_true")
    parser.add_argument("--critic-repair-attempts", type=int, default=2)
    parser.add_argument("--compiler-repair-attempts", type=int, default=2)
    parser.add_argument("--validation-level", choices=("static", "build", "full"), default="static")
    parser.add_argument(
        "--equiformer-v3-root",
        default=os.environ.get("EQUIFORMER_V3_ROOT", "") or (str(DEFAULT_V3_ROOT) if DEFAULT_V3_ROOT.exists() else ""),
    )
    parser.add_argument("--resume", action="store_true")
    return parser


def main():
    args = get_parser().parse_args()
    if args.cycle_index <= 0:
        raise ValueError("--cycle-index must be positive")
    if not args.equiformer_v3_root:
        raise ValueError("V3 cohort validation requires --equiformer-v3-root")
    run(args)


if __name__ == "__main__":
    main()
