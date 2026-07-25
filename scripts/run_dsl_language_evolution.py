#!/usr/bin/env python
"""Discover or admit learned motifs at a closed OpenEvolve language boundary."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from equivariant_nas.dsl import (
    CandidateLineageEvidence,
    Compiler,
    DSLValidationError,
    EvidenceStore,
    LanguageEvolutionPreregistration,
    MotifDiscoveryPolicy,
    core_registry,
    discover_motif_proposals,
    replay_motif_proposal,
    run_language_evolution_boundary,
)
from equivariant_nas.dsl.serialization import load_program, load_task_contract


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _load_preregistration(snapshot):
    record = snapshot["boundary_preregistration"]
    preregistration = LanguageEvolutionPreregistration(
        str(record["boundary_id"]),
        str(record["parent_language_version"]),
        int(record["cycle_index"]),
        str(record["selection_rule"]),
        str(record["partition_rule"]),
        str(record["discovery_policy_hash"]),
        str(record["preregistered_at"]),
        int(record.get("max_publications", 1)),
    )
    if preregistration.content_hash() != record.get("preregistration_hash"):
        raise RuntimeError("language cycle snapshot does not match its preregistration hash")
    if snapshot.get("selection_rule") != preregistration.selection_rule or snapshot.get("partition_rule") != preregistration.partition_rule:
        raise RuntimeError("language cycle snapshot silently changed its pre-registered selection or partition rule")
    return preregistration


def _load_candidates(snapshot, compiler, task, language):
    candidates = []
    for record in snapshot.get("candidates", ()):
        if record.get("test_evaluated") or "test" in {str(item).lower() for item in record.get("visible_splits", ())}:
            raise RuntimeError("test-visible candidate found in language cycle snapshot")
        source = load_program(str(record["program_path"]))
        artifact = compiler.analyze(source, task)
        if artifact.architecture_id != record.get("architecture_id"):
            raise RuntimeError("candidate program changed after the OpenEvolve cycle snapshot")
        if record.get("language_registry_hash") != language.registry_hash():
            raise RuntimeError("candidate was compiled under a different frozen language")
        if record.get("rewrite_registry_hash") != artifact.rewrite_registry_hash:
            raise RuntimeError("candidate rewrite registry differs from the current compiler")
        candidates.append(
            CandidateLineageEvidence(
                artifact,
                str(record["lineage_id"]),
                str(record["task_id"]),
                tuple(str(item) for item in record.get("visible_splits", ())),
                str(record["language_registry_hash"]),
                str(record["rewrite_registry_hash"]),
                bool(record.get("test_evaluated", False)),
                str(record["discovery_partition"]),
            )
        )
    if not candidates:
        raise RuntimeError("closed language cycle contains no eligible candidates")
    return tuple(candidates)


def _discover(candidates, parent, primitives, motifs, policy, store):
    report = discover_motif_proposals(candidates, primitives, policy)
    language_hash = parent.registry_hash()
    rewrite_hash = candidates[0].rewrite_registry_hash
    for occurrence in report.occurrences:
        store.add_motif_occurrence(occurrence)
    replay = {}
    for proposal in report.proposals:
        store.add_motif_proposal(
            proposal,
            parent_language_version=parent.version,
            language_registry_hash=language_hash,
            rewrite_registry_hash=rewrite_hash,
        )
        results = replay_motif_proposal(proposal, candidates, primitives, motifs)
        replay[proposal.proposal_id] = results
        for result in results:
            store.add_language_replay(result)
    return report, replay


def run(args):
    snapshot_path = Path(args.cycle_snapshot).resolve()
    snapshot = _read_json(snapshot_path)
    preregistration = _load_preregistration(snapshot)
    policy = MotifDiscoveryPolicy()
    if preregistration.discovery_policy_hash != policy.content_hash():
        raise RuntimeError("runtime motif discovery policy differs from the cycle preregistration")
    store = EvidenceStore(args.evidence_store)
    restored = store.get_language_snapshot(preregistration.parent_language_version)
    if restored is None:
        raise RuntimeError("parent language snapshot is absent from the evidence database")
    parent, motifs = restored
    primitives = core_registry()
    compiler = Compiler(primitives, motifs)
    task = load_task_contract(args.task_contract)
    candidates = _load_candidates(snapshot, compiler, task, parent)
    expected_ids = tuple(item.architecture_id for item in candidates)
    boundary = preregistration.close(expected_ids, str(snapshot["closed_at"]), test_hidden=True)
    output_path = Path(args.output).resolve() if args.output else snapshot_path.with_name(
        "language_discovery.json" if args.mode == "discover" else "language_admission.json"
    )
    if args.mode == "discover":
        report, replay = _discover(candidates, parent, primitives, motifs, policy, store)
        payload = {
            "status": "discovered",
            "boundary": boundary.to_dict(),
            "parent_language_registry_hash": parent.registry_hash(),
            "discovery": report.to_dict(),
            "replay_results": {
                key: [item.to_dict() for item in values]
                for key, values in replay.items()
            },
            "test_evaluated": False,
            "created_at": _now(),
        }
    else:
        if not args.regression_evidence or not args.new_version:
            raise RuntimeError("admit mode requires --regression-evidence and --new-version")
        regression = _read_json(args.regression_evidence)
        generation = _read_json(args.generation_evidence) if args.generation_evidence else {}
        deltas = {
            proposal_id: float(record.get("valid_generation_rate_delta", 0.0))
            for proposal_id, record in generation.items()
        }
        experiment_ids = {
            proposal_id: tuple(str(item) for item in record.get("artifact_ids", ()))
            for proposal_id, record in generation.items()
        }
        result = run_language_evolution_boundary(
            candidates,
            parent,
            primitives,
            motifs,
            boundary,
            new_version=args.new_version,
            frozen_at=args.frozen_at or _now(),
            discovery_policy=policy,
            regression_passed=bool(regression.get("passed", False)),
            regression_artifact_ids=tuple(str(item) for item in regression.get("artifact_ids", ())),
            valid_generation_rate_deltas=deltas,
            generation_experiment_ids=experiment_ids,
            heldout_task_ids=tuple(str(item) for item in args.heldout_task_id),
            evidence_store=store,
        )
        payload = dict(result.to_dict(), status="published" if result.published else "no_admissible_motif", test_evaluated=False, created_at=_now())
    _write_json(output_path, payload)
    print(json.dumps({"status": payload["status"], "output": str(output_path)}, ensure_ascii=False, sort_keys=True))
    return payload


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("discover", "admit"), default="discover")
    parser.add_argument("--cycle-snapshot", required=True)
    parser.add_argument("--task-contract", required=True)
    parser.add_argument("--evidence-store", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--regression-evidence", default="")
    parser.add_argument("--generation-evidence", default="")
    parser.add_argument("--new-version", default="")
    parser.add_argument("--frozen-at", default="")
    parser.add_argument("--heldout-task-id", action="append", default=[])
    return parser


def main():
    run(get_parser().parse_args())


if __name__ == "__main__":
    main()
