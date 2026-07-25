"""Evidence-gated motif admission and language-version publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .language import LanguageVersion
from .motifs import MotifDefinition, MotifRegistry


ADMISSION_POLICY_VERSION = "motif-admission-v2"


@dataclass(frozen=True)
class MotifAdmissionEvidence:
    independent_lineages: Tuple[str, ...]
    heldout_tasks: Tuple[str, ...]
    description_length_gain: float
    valid_generation_rate_delta: float
    regression_passed: bool
    novelty_checked: bool
    proof_artifact_ids: Tuple[str, ...]
    notes: Mapping[str, Any]
    source_architecture_ids: Tuple[str, ...] = ()
    replay_artifact_ids: Tuple[str, ...] = ()
    heldout_replay_artifact_ids: Tuple[str, ...] = ()
    replay_passed: bool = False
    test_hidden: bool = False
    language_registry_hash: str = ""
    rewrite_registry_hash: str = ""
    admission_policy_version: str = ADMISSION_POLICY_VERSION
    discovery_policy_hash: str = ""
    boundary_id: str = ""
    proposal_id: str = ""
    regression_artifact_ids: Tuple[str, ...] = ()
    generation_experiment_ids: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "independent_lineages": list(self.independent_lineages),
            "heldout_tasks": list(self.heldout_tasks),
            "description_length_gain": self.description_length_gain,
            "valid_generation_rate_delta": self.valid_generation_rate_delta,
            "regression_passed": self.regression_passed,
            "novelty_checked": self.novelty_checked,
            "proof_artifact_ids": list(self.proof_artifact_ids),
            "notes": dict(self.notes),
            "source_architecture_ids": list(self.source_architecture_ids),
            "replay_artifact_ids": list(self.replay_artifact_ids),
            "heldout_replay_artifact_ids": list(self.heldout_replay_artifact_ids),
            "replay_passed": self.replay_passed,
            "test_hidden": self.test_hidden,
            "language_registry_hash": self.language_registry_hash,
            "rewrite_registry_hash": self.rewrite_registry_hash,
            "admission_policy_version": self.admission_policy_version,
            "discovery_policy_hash": self.discovery_policy_hash,
            "boundary_id": self.boundary_id,
            "proposal_id": self.proposal_id,
            "regression_artifact_ids": list(self.regression_artifact_ids),
            "generation_experiment_ids": list(self.generation_experiment_ids),
        }


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reasons: Tuple[str, ...]
    motif_hash: str
    policy_version: str = ADMISSION_POLICY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "motif_hash": self.motif_hash,
            "policy_version": self.policy_version,
        }


def evaluate_motif_admission(
    motif: MotifDefinition,
    evidence: MotifAdmissionEvidence,
    registry: MotifRegistry,
) -> AdmissionDecision:
    reasons = []
    if motif.certification not in ("constructive", "core-certified"):
        reasons.append("motif is not constructively certified")
    if len(set(evidence.independent_lineages)) < 2 and not evidence.heldout_tasks:
        reasons.append("motif lacks two independent lineages or held-out transfer evidence")
    if evidence.description_length_gain <= 0 and evidence.valid_generation_rate_delta <= 0:
        reasons.append("motif provides neither description compression nor generation-validity gain")
    if not evidence.regression_passed:
        reasons.append("language regression suite did not pass")
    if not evidence.novelty_checked:
        reasons.append("motif novelty was not checked")
    if not evidence.proof_artifact_ids:
        reasons.append("motif has no proof artifacts")
    if not evidence.source_architecture_ids:
        reasons.append("motif has no source architecture identities")
    if not evidence.replay_artifact_ids or not evidence.replay_passed:
        reasons.append("motif did not pass semantic fold-expand replay")
    if not evidence.heldout_replay_artifact_ids:
        reasons.append("motif has no successful held-out program replay")
    if not evidence.test_hidden:
        reasons.append("language-learning evidence is not certified test-hidden")
    if not evidence.language_registry_hash or not evidence.rewrite_registry_hash:
        reasons.append("language or rewrite registry identity is missing")
    if evidence.admission_policy_version != ADMISSION_POLICY_VERSION:
        reasons.append("motif admission policy version is unsupported")
    if not evidence.discovery_policy_hash:
        reasons.append("motif discovery policy identity is missing")
    if not evidence.boundary_id:
        reasons.append("motif was not evaluated at a pre-registered language boundary")
    if not evidence.proposal_id:
        reasons.append("motif proposal identity is missing")
    if not evidence.regression_artifact_ids:
        reasons.append("language regression result has no artifact identity")
    if evidence.valid_generation_rate_delta > 0 and not evidence.generation_experiment_ids:
        reasons.append("generation-validity gain has no matched experiment artifact")
    known_hashes = {registry.resolve(name).content_hash() for name in registry.names()}
    motif_hash = motif.content_hash()
    if motif_hash in known_hashes:
        reasons.append("motif is content-equivalent to an existing registry entry")
    if motif.qualified_name in registry.names():
        reasons.append("motif qualified name already exists in the registry")
    return AdmissionDecision(not reasons, tuple(reasons), motif_hash, evidence.admission_policy_version)


def publish_motif_version(
    parent: LanguageVersion,
    motif: MotifDefinition,
    evidence: MotifAdmissionEvidence,
    registry: MotifRegistry,
    *,
    new_version: str,
    frozen_at: str,
) -> LanguageVersion:
    if not new_version or new_version == parent.version:
        raise DSLValidationError([Diagnostic("E_ADMISSION_002", "published language requires a distinct nonempty version")])
    decision = evaluate_motif_admission(motif, evidence, registry)
    if not decision.accepted:
        raise DSLValidationError([Diagnostic("E_ADMISSION_001", "motif admission rejected", details={"reasons": list(decision.reasons)})])
    metadata = dict(parent.metadata)
    metadata["admitted_motif"] = motif.qualified_name
    metadata["admission_motif_hash"] = decision.motif_hash
    metadata["admission_policy_version"] = evidence.admission_policy_version
    metadata["language_boundary_id"] = evidence.boundary_id
    metadata["motif_proposal_id"] = evidence.proposal_id
    motif_hashes = {
        name: registry.resolve(name).content_hash()
        for name in parent.motif_names
    }
    motif_hashes[motif.qualified_name] = motif.content_hash()
    child = LanguageVersion(
        new_version,
        parent.version,
        parent.primitive_names,
        tuple(sorted(set(parent.motif_names) | {motif.qualified_name})),
        frozen_at,
        metadata,
        dict(parent.primitive_hashes),
        motif_hashes,
    )
    registry.register(motif)
    return child
