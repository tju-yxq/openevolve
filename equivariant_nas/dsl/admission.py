"""Evidence-gated motif admission and language-version publication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .language import LanguageVersion
from .motifs import MotifDefinition, MotifRegistry


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


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reasons: Tuple[str, ...]
    motif_hash: str


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
    known_hashes = {registry.resolve(name).content_hash() for name in registry.names()}
    motif_hash = motif.content_hash()
    if motif_hash in known_hashes:
        reasons.append("motif is content-equivalent to an existing registry entry")
    return AdmissionDecision(not reasons, tuple(reasons), motif_hash)


def publish_motif_version(
    parent: LanguageVersion,
    motif: MotifDefinition,
    evidence: MotifAdmissionEvidence,
    registry: MotifRegistry,
    *,
    new_version: str,
    frozen_at: str,
) -> LanguageVersion:
    decision = evaluate_motif_admission(motif, evidence, registry)
    if not decision.accepted:
        raise DSLValidationError([Diagnostic("E_ADMISSION_001", "motif admission rejected", details={"reasons": list(decision.reasons)})])
    registry.register(motif)
    metadata = dict(parent.metadata)
    metadata["admitted_motif"] = motif.qualified_name
    metadata["admission_motif_hash"] = decision.motif_hash
    return LanguageVersion(
        new_version,
        parent.version,
        parent.primitive_names,
        tuple(sorted(set(parent.motif_names) | {motif.qualified_name})),
        frozen_at,
        metadata,
        dict(parent.primitive_hashes),
        {name: registry.resolve(name).content_hash() for name in registry.names()},
    )
