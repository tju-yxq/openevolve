"""Slow-timescale, evidence-gated evolution of the learned motif vocabulary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .admission import AdmissionDecision, MotifAdmissionEvidence, evaluate_motif_admission, publish_motif_version
from .diagnostics import DSLValidationError, Diagnostic
from .evidence_store import EvidenceStore
from .language import LanguageVersion
from .motif_discovery import (
    CandidateLineageEvidence,
    LanguageReplayResult,
    MotifDiscoveryPolicy,
    MotifDiscoveryReport,
    MotifProposal,
    discover_motif_proposals,
    replay_motif_proposal,
)
from .motifs import MotifRegistry
from .registry import PrimitiveRegistry


@dataclass(frozen=True)
class LanguageEvolutionPreregistration:
    boundary_id: str
    parent_language_version: str
    cycle_index: int
    selection_rule: str
    partition_rule: str
    discovery_policy_hash: str
    preregistered_at: str
    max_publications: int = 1

    def __post_init__(self) -> None:
        if not self.boundary_id or not self.parent_language_version or not self.preregistered_at:
            raise DSLValidationError([Diagnostic("E_LANGUAGE_EVOLUTION_001", "language preregistration identity is incomplete")])
        if self.cycle_index < 1 or self.max_publications != 1:
            raise DSLValidationError([
                Diagnostic("E_LANGUAGE_EVOLUTION_002", "each language cycle must be positive and publish at most one motif")
            ])
        if not self.selection_rule or not self.partition_rule or not self.discovery_policy_hash:
            raise DSLValidationError([
                Diagnostic("E_LANGUAGE_EVOLUTION_003", "selection, partition, and discovery policies must be pre-registered")
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "parent_language_version": self.parent_language_version,
            "cycle_index": self.cycle_index,
            "selection_rule": self.selection_rule,
            "partition_rule": self.partition_rule,
            "discovery_policy_hash": self.discovery_policy_hash,
            "preregistered_at": self.preregistered_at,
            "max_publications": self.max_publications,
        }

    def content_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def close(self, candidate_ids: Sequence[str], closed_at: str, *, test_hidden: bool = True) -> "LanguageEvolutionBoundary":
        return LanguageEvolutionBoundary(
            self.boundary_id,
            self.parent_language_version,
            self.cycle_index,
            tuple(candidate_ids),
            self.preregistered_at,
            closed_at,
            test_hidden,
            self.selection_rule,
            self.partition_rule,
            self.discovery_policy_hash,
            self.content_hash(),
        )


@dataclass(frozen=True)
class LanguageEvolutionBoundary:
    boundary_id: str
    parent_language_version: str
    cycle_index: int
    expected_candidate_ids: Tuple[str, ...]
    preregistered_at: str
    closed_at: str
    test_hidden: bool = True
    selection_rule: str = ""
    partition_rule: str = ""
    discovery_policy_hash: str = ""
    preregistration_hash: str = ""

    def __post_init__(self) -> None:
        if not self.boundary_id or not self.parent_language_version or not self.preregistered_at:
            raise DSLValidationError([Diagnostic("E_LANGUAGE_EVOLUTION_004", "language boundary identity is incomplete")])
        if self.cycle_index < 1:
            raise DSLValidationError([Diagnostic("E_LANGUAGE_EVOLUTION_005", "language cycle index must be positive")])
        if not self.expected_candidate_ids or len(set(self.expected_candidate_ids)) != len(self.expected_candidate_ids):
            raise DSLValidationError([
                Diagnostic("E_LANGUAGE_EVOLUTION_006", "closed language boundary requires a unique materialized candidate set")
            ])
        if not self.selection_rule or not self.partition_rule or not self.discovery_policy_hash or not self.preregistration_hash:
            raise DSLValidationError([
                Diagnostic("E_LANGUAGE_EVOLUTION_007", "closed boundary is not bound to a complete preregistration")
            ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "parent_language_version": self.parent_language_version,
            "cycle_index": self.cycle_index,
            "expected_candidate_ids": list(self.expected_candidate_ids),
            "preregistered_at": self.preregistered_at,
            "closed_at": self.closed_at,
            "test_hidden": self.test_hidden,
            "selection_rule": self.selection_rule,
            "partition_rule": self.partition_rule,
            "discovery_policy_hash": self.discovery_policy_hash,
            "preregistration_hash": self.preregistration_hash,
        }


@dataclass(frozen=True)
class MotifAdmissionRecord:
    proposal: MotifProposal
    replay_results: Tuple[LanguageReplayResult, ...]
    evidence: MotifAdmissionEvidence
    decision: AdmissionDecision

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "replay_results": [item.to_dict() for item in self.replay_results],
            "evidence": self.evidence.to_dict(),
            "decision": self.decision.to_dict(),
        }


@dataclass(frozen=True)
class LanguageEvolutionResult:
    boundary: LanguageEvolutionBoundary
    discovery: MotifDiscoveryReport
    admissions: Tuple[MotifAdmissionRecord, ...]
    selected_proposal_id: str
    child_language: Optional[LanguageVersion]

    @property
    def published(self) -> bool:
        return self.child_language is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary": self.boundary.to_dict(),
            "discovery": self.discovery.to_dict(),
            "admissions": [item.to_dict() for item in self.admissions],
            "selected_proposal_id": self.selected_proposal_id,
            "child_language": {
                "version": self.child_language.version,
                "parent_version": self.child_language.parent_version,
                "registry_hash": self.child_language.registry_hash(),
                "motif_names": list(self.child_language.motif_names),
                "metadata": dict(self.child_language.metadata),
            } if self.child_language is not None else None,
        }


def _validate_boundary(
    boundary: LanguageEvolutionBoundary,
    candidates: Sequence[CandidateLineageEvidence],
    parent: LanguageVersion,
    discovery_policy: MotifDiscoveryPolicy,
) -> Tuple[str, str]:
    if boundary.parent_language_version != parent.version:
        raise DSLValidationError([
            Diagnostic(
                "E_LANGUAGE_EVOLUTION_008",
                "language boundary targets a different parent version",
                expected=parent.version,
                actual=boundary.parent_language_version,
            )
        ])
    if not boundary.closed_at:
        raise DSLValidationError([
            Diagnostic("E_LANGUAGE_EVOLUTION_009", "language updates are forbidden before the pre-registered cycle closes")
        ])
    if not boundary.test_hidden:
        raise DSLValidationError([
            Diagnostic("E_LANGUAGE_EVOLUTION_010", "test-visible boundaries cannot update the language")
        ])
    actual_ids = {item.architecture_id for item in candidates}
    expected_ids = set(boundary.expected_candidate_ids)
    if actual_ids != expected_ids:
        raise DSLValidationError([
            Diagnostic(
                "E_LANGUAGE_EVOLUTION_011",
                "language update candidate set differs from its pre-registration",
                details={"missing": sorted(expected_ids - actual_ids), "unexpected": sorted(actual_ids - expected_ids)},
            )
        ])
    wrong_versions = sorted({item.artifact.source_program.language_version for item in candidates} - {parent.version})
    if wrong_versions:
        raise DSLValidationError([
            Diagnostic("E_LANGUAGE_EVOLUTION_012", "candidate programs mix language versions", details={"versions": wrong_versions})
        ])
    language_hashes = {item.language_registry_hash for item in candidates}
    rewrite_hashes = {item.rewrite_registry_hash for item in candidates}
    if language_hashes != {parent.registry_hash()}:
        raise DSLValidationError([
            Diagnostic(
                "E_LANGUAGE_EVOLUTION_013",
                "candidate language registry does not match the frozen parent language",
                expected=parent.registry_hash(),
                actual=",".join(sorted(language_hashes)),
            )
        ])
    if len(rewrite_hashes) != 1:
        raise DSLValidationError([
            Diagnostic("E_LANGUAGE_EVOLUTION_014", "candidate window mixes rewrite registries")
        ])
    if boundary.discovery_policy_hash != discovery_policy.content_hash():
        raise DSLValidationError([
            Diagnostic(
                "E_LANGUAGE_EVOLUTION_015",
                "runtime discovery policy differs from the pre-registered policy",
                expected=boundary.discovery_policy_hash,
                actual=discovery_policy.content_hash(),
            )
        ])
    return next(iter(language_hashes)), next(iter(rewrite_hashes))


def run_language_evolution_boundary(
    candidates: Sequence[CandidateLineageEvidence],
    parent: LanguageVersion,
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
    boundary: LanguageEvolutionBoundary,
    *,
    new_version: str,
    frozen_at: str,
    discovery_policy: Optional[MotifDiscoveryPolicy] = None,
    regression_passed: bool,
    regression_artifact_ids: Sequence[str],
    valid_generation_rate_deltas: Optional[Mapping[str, float]] = None,
    generation_experiment_ids: Optional[Mapping[str, Sequence[str]]] = None,
    heldout_task_ids: Sequence[str] = (),
    evidence_store: Optional[EvidenceStore] = None,
) -> LanguageEvolutionResult:
    """Run one immutable language update after a pre-registered search cycle.

    At most one motif is published at a boundary. All discovered proposals and
    admission decisions are retained, including rejected proposals.
    """

    selected_policy = discovery_policy or MotifDiscoveryPolicy()
    language_hash, rewrite_hash = _validate_boundary(boundary, candidates, parent, selected_policy)
    report = discover_motif_proposals(candidates, primitives, selected_policy)
    generation_deltas = valid_generation_rate_deltas or {}
    generation_ids = generation_experiment_ids or {}
    if evidence_store is not None:
        for candidate in candidates:
            if not evidence_store.candidate_exists(candidate.architecture_id):
                raise DSLValidationError([
                    Diagnostic(
                        "E_LANGUAGE_EVOLUTION_016",
                        "candidate must be registered before motif evidence is appended",
                        actual=candidate.architecture_id,
                    )
                ])
        for occurrence in report.occurrences:
            evidence_store.add_motif_occurrence(occurrence)
    heldout = set(heldout_task_ids)
    admissions = []
    for proposal in report.proposals:
        if evidence_store is not None:
            evidence_store.add_motif_proposal(
                proposal,
                parent_language_version=parent.version,
                language_registry_hash=language_hash,
                rewrite_registry_hash=rewrite_hash,
                status="proposed",
            )
        replay = replay_motif_proposal(proposal, candidates, primitives, motifs)
        if evidence_store is not None:
            for result in replay:
                evidence_store.add_language_replay(result)
        replay_passed = bool(replay) and all(item.passed for item in replay)
        novelty = proposal.motif.content_hash() not in {
            motifs.resolve(name).content_hash() for name in motifs.names()
        }
        delta = float(generation_deltas.get(proposal.proposal_id, 0.0))
        experiment_ids = tuple(generation_ids.get(proposal.proposal_id, ()))
        evidence = MotifAdmissionEvidence(
            independent_lineages=proposal.independent_lineages,
            heldout_tasks=tuple(sorted(set(proposal.task_ids) & heldout)),
            description_length_gain=proposal.description_length_gain,
            valid_generation_rate_delta=delta,
            regression_passed=bool(regression_passed),
            novelty_checked=novelty,
            proof_artifact_ids=proposal.proof_artifact_ids + tuple(item.replay_id for item in replay if item.passed),
            notes={
                "occurrence_count": len(proposal.occurrences),
                "task_ids": list(proposal.task_ids),
                "boundary_cycle_index": boundary.cycle_index,
            },
            source_architecture_ids=proposal.source_architecture_ids,
            replay_artifact_ids=tuple(item.replay_id for item in replay),
            heldout_replay_artifact_ids=tuple(
                item.replay_id for item in replay
                if item.discovery_partition == "heldout_replay" and item.passed
            ),
            replay_passed=replay_passed,
            test_hidden=boundary.test_hidden,
            language_registry_hash=language_hash,
            rewrite_registry_hash=rewrite_hash,
            discovery_policy_hash=proposal.discovery_policy_hash,
            boundary_id=boundary.boundary_id,
            proposal_id=proposal.proposal_id,
            regression_artifact_ids=tuple(regression_artifact_ids),
            generation_experiment_ids=experiment_ids,
        )
        decision = evaluate_motif_admission(proposal.motif, evidence, motifs)
        record = MotifAdmissionRecord(proposal, replay, evidence, decision)
        admissions.append(record)
        if evidence_store is not None:
            evidence_store.add_motif_admission(proposal.proposal_id, boundary.boundary_id, decision, evidence)
            evidence_store.add_motif_proposal(
                proposal,
                parent_language_version=parent.version,
                language_registry_hash=language_hash,
                rewrite_registry_hash=rewrite_hash,
                status="admissible" if decision.accepted else "rejected",
            )
    accepted = [item for item in admissions if item.decision.accepted]
    accepted.sort(
        key=lambda item: (
            -item.proposal.description_length_gain,
            -len(item.proposal.independent_lineages),
            -len(item.proposal.occurrences),
            item.proposal.proposal_id,
        )
    )
    child = None
    selected_id = ""
    if accepted:
        selected = accepted[0]
        selected_id = selected.proposal.proposal_id
        child = publish_motif_version(
            parent,
            selected.proposal.motif,
            selected.evidence,
            motifs,
            new_version=new_version,
            frozen_at=frozen_at,
        )
        if evidence_store is not None:
            evidence_store.add_motif_proposal(
                selected.proposal,
                parent_language_version=parent.version,
                language_registry_hash=language_hash,
                rewrite_registry_hash=rewrite_hash,
                status="published",
            )
            evidence_store.register_language(child, motifs)
    return LanguageEvolutionResult(boundary, report, tuple(admissions), selected_id, child)
