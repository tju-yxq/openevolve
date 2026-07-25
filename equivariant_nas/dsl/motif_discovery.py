"""Typed subgraph discovery, conservative anti-unification, and motif replay."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .ast import ArchitectureProgram, Node
from .compiler import CompilationArtifact, Compiler
from .diagnostics import DSLValidationError, Diagnostic
from .inference import InferenceResult
from .motifs import MotifDefinition, MotifRegistry
from .registry import PrimitiveRegistry
from .types import EquivariantType


DISCOVERY_POLICY_VERSION = "typed-motif-discovery-v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(prefix: str, value: Any, length: int = 20) -> str:
    return "{}:{}".format(prefix, hashlib.sha256(_json(value).encode("utf-8")).hexdigest()[:length])


def _qualified(op: str) -> str:
    return op if "@" in op else "{}@1".format(op)


def _split_reference(reference: str, nodes: Mapping[str, Node]) -> Tuple[str, str]:
    root, separator, port = reference.partition(":")
    if root not in nodes:
        return root, port if separator else ""
    if separator:
        return root, port
    outputs = nodes[root].outputs
    if len(outputs) != 1:
        raise DSLValidationError([
            Diagnostic("E_DISCOVERY_001", "bare reference to a multi-output node is ambiguous", actual=reference)
        ])
    return root, outputs[0]


def _normalized_reference(reference: str, nodes: Mapping[str, Node]) -> str:
    root, port = _split_reference(reference, nodes)
    return "{}:{}".format(root, port) if root in nodes else reference


def _type_for(reference: str, inference: InferenceResult, nodes: Mapping[str, Node]) -> EquivariantType:
    normalized = _normalized_reference(reference, nodes)
    if normalized in inference.value_types:
        return inference.value_types[normalized]
    if reference in inference.value_types:
        return inference.value_types[reference]
    raise DSLValidationError([
        Diagnostic("E_DISCOVERY_002", "subgraph boundary value has no inferred type", actual=reference)
    ])


@dataclass(frozen=True)
class MotifDiscoveryPolicy:
    min_nodes: int = 2
    max_nodes: int = 4
    min_occurrences: int = 2
    max_subgraphs_per_program: int = 10000
    max_proposals: int = 64
    policy_version: str = DISCOVERY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.min_nodes < 2 or self.max_nodes < self.min_nodes:
            raise DSLValidationError([Diagnostic("E_DISCOVERY_003", "invalid motif node bounds")])
        if self.max_nodes > 7:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_004", "canonical permutation search is bounded at seven nodes")
            ])
        if self.min_occurrences < 2 or self.max_subgraphs_per_program <= 0 or self.max_proposals <= 0:
            raise DSLValidationError([Diagnostic("E_DISCOVERY_005", "invalid motif discovery budget")])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_nodes": self.min_nodes,
            "max_nodes": self.max_nodes,
            "min_occurrences": self.min_occurrences,
            "max_subgraphs_per_program": self.max_subgraphs_per_program,
            "max_proposals": self.max_proposals,
            "policy_version": self.policy_version,
        }

    def content_hash(self) -> str:
        return hashlib.sha256(_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateLineageEvidence:
    artifact: CompilationArtifact
    lineage_id: str
    task_id: str
    visible_splits: Tuple[str, ...]
    language_registry_hash: str
    rewrite_registry_hash: str
    test_evaluated: bool = False
    discovery_partition: str = "support"

    def __post_init__(self) -> None:
        visible = {item.lower() for item in self.visible_splits}
        if self.test_evaluated or "test" in visible:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_006", "test-visible candidates cannot teach the language")
            ])
        if not self.lineage_id or not self.task_id:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_007", "candidate lineage and task identities are required")
            ])
        if not self.language_registry_hash or not self.rewrite_registry_hash:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_008", "candidate compiler registry hashes are required")
            ])
        if self.artifact.inference.open_obligations:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_009", "candidate has open proof obligations")
            ])
        if self.discovery_partition not in ("support", "heldout_replay"):
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_017", "unknown motif discovery partition", actual=self.discovery_partition)
            ])

    @property
    def architecture_id(self) -> str:
        return self.artifact.architecture_id


@dataclass(frozen=True)
class BoundaryValue:
    name: str
    reference: str
    value_type: EquivariantType

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "reference": self.reference, "type": self.value_type.to_dict()}


@dataclass(frozen=True)
class TypedSubgraphOccurrence:
    occurrence_id: str
    architecture_id: str
    lineage_id: str
    task_id: str
    node_ids: Tuple[str, ...]
    canonical_node_ids: Tuple[str, ...]
    boundary_inputs: Tuple[BoundaryValue, ...]
    boundary_outputs: Tuple[BoundaryValue, ...]
    canonical_form: str
    topology_hash: str
    boundary_signature_hash: str
    language_registry_hash: str
    rewrite_registry_hash: str
    discovery_partition: str = "support"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "architecture_id": self.architecture_id,
            "lineage_id": self.lineage_id,
            "task_id": self.task_id,
            "node_ids": list(self.node_ids),
            "canonical_node_ids": list(self.canonical_node_ids),
            "boundary_inputs": [item.to_dict() for item in self.boundary_inputs],
            "boundary_outputs": [item.to_dict() for item in self.boundary_outputs],
            "canonical_form": self.canonical_form,
            "topology_hash": self.topology_hash,
            "boundary_signature_hash": self.boundary_signature_hash,
            "language_registry_hash": self.language_registry_hash,
            "rewrite_registry_hash": self.rewrite_registry_hash,
            "discovery_partition": self.discovery_partition,
        }


@dataclass(frozen=True)
class MotifProposal:
    proposal_id: str
    motif: MotifDefinition
    occurrences: Tuple[TypedSubgraphOccurrence, ...]
    parameter_bindings: Mapping[str, Mapping[str, Any]]
    description_length_gain: float
    uncompressed_length: int
    compressed_length: int
    independent_lineages: Tuple[str, ...]
    task_ids: Tuple[str, ...]
    source_architecture_ids: Tuple[str, ...]
    proof_artifact_ids: Tuple[str, ...]
    discovery_policy_hash: str
    canonical_form_hash: str

    def to_dict(self) -> Dict[str, Any]:
        motif_record = self.motif.to_dict()
        return {
            "proposal_id": self.proposal_id,
            "motif": motif_record,
            "motif_hash": self.motif.content_hash(),
            "occurrence_ids": [item.occurrence_id for item in self.occurrences],
            "parameter_bindings": {key: dict(value) for key, value in self.parameter_bindings.items()},
            "description_length_gain": self.description_length_gain,
            "uncompressed_length": self.uncompressed_length,
            "compressed_length": self.compressed_length,
            "independent_lineages": list(self.independent_lineages),
            "task_ids": list(self.task_ids),
            "source_architecture_ids": list(self.source_architecture_ids),
            "proof_artifact_ids": list(self.proof_artifact_ids),
            "discovery_policy_hash": self.discovery_policy_hash,
            "canonical_form_hash": self.canonical_form_hash,
        }


@dataclass(frozen=True)
class MotifDiscoveryReport:
    policy: MotifDiscoveryPolicy
    occurrences: Tuple[TypedSubgraphOccurrence, ...]
    proposals: Tuple[MotifProposal, ...]
    rejected_clusters: Mapping[str, Tuple[str, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "occurrences": [item.to_dict() for item in self.occurrences],
            "proposals": [item.to_dict() for item in self.proposals],
            "rejected_clusters": {key: list(value) for key, value in self.rejected_clusters.items()},
        }


@dataclass(frozen=True)
class LanguageReplayResult:
    replay_id: str
    proposal_id: str
    occurrence_id: str
    architecture_id: str
    before_semantic_id: str
    after_semantic_id: str
    passed: bool
    diagnostics: Tuple[Mapping[str, Any], ...] = ()
    discovery_partition: str = "support"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "proposal_id": self.proposal_id,
            "occurrence_id": self.occurrence_id,
            "architecture_id": self.architecture_id,
            "before_semantic_id": self.before_semantic_id,
            "after_semantic_id": self.after_semantic_id,
            "passed": self.passed,
            "diagnostics": [dict(item) for item in self.diagnostics],
            "discovery_partition": self.discovery_partition,
        }


def _connected_node_sets(program: ArchitectureProgram, policy: MotifDiscoveryPolicy) -> Tuple[Tuple[str, ...], ...]:
    nodes = {node.id: node for node in program.nodes}
    neighbors: Dict[str, Set[str]] = {node_id: set() for node_id in nodes}
    for node in program.nodes:
        for references in node.inputs.values():
            for reference in references:
                root = reference.split(":", 1)[0]
                if root in nodes:
                    neighbors[node.id].add(root)
                    neighbors[root].add(node.id)
    frontier = {frozenset((node_id,)) for node_id in nodes}
    discovered: Set[frozenset] = set()
    for size in range(1, policy.max_nodes + 1):
        if size >= policy.min_nodes:
            discovered.update(frontier)
            if len(discovered) > policy.max_subgraphs_per_program:
                raise DSLValidationError([
                    Diagnostic(
                        "E_DISCOVERY_010",
                        "motif enumeration exceeded its pre-registered per-program budget",
                        details={"limit": policy.max_subgraphs_per_program},
                    )
                ])
        next_frontier: Set[frozenset] = set()
        for subset in frontier:
            adjacent = set()
            for node_id in subset:
                adjacent.update(neighbors[node_id])
            for node_id in adjacent - set(subset):
                next_frontier.add(frozenset(set(subset) | {node_id}))
        frontier = next_frontier
        if not frontier:
            break
    return tuple(sorted((tuple(sorted(item)) for item in discovered), key=lambda item: (len(item), item)))


def _boundary_references(program: ArchitectureProgram, subset: Set[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    nodes = {node.id: node for node in program.nodes}
    incoming = []
    outgoing = []
    for node in program.nodes:
        for references in node.inputs.values():
            for reference in references:
                root = reference.split(":", 1)[0]
                normalized = _normalized_reference(reference, nodes)
                if node.id in subset and root not in subset:
                    incoming.append(normalized)
                if node.id not in subset and root in subset:
                    outgoing.append(normalized)
    for output in program.outputs:
        root = output.source.split(":", 1)[0]
        if root in subset:
            outgoing.append(_normalized_reference(output.source, nodes))
    return tuple(dict.fromkeys(incoming)), tuple(dict.fromkeys(outgoing))


def _canonical_occurrence(
    candidate: CandidateLineageEvidence,
    node_ids: Sequence[str],
) -> Optional[TypedSubgraphOccurrence]:
    program = candidate.artifact.expanded_program
    inference = candidate.artifact.inference
    nodes = {node.id: node for node in program.nodes}
    subset = set(node_ids)
    incoming, outgoing = _boundary_references(program, subset)
    if not incoming or not outgoing:
        return None
    best_text = None
    best_order: Tuple[str, ...] = ()
    best_inputs: Tuple[str, ...] = ()
    best_outputs: Tuple[str, ...] = ()
    best_payload: Mapping[str, Any] = {}
    for order in itertools.permutations(sorted(subset)):
        local = {node_id: index for index, node_id in enumerate(order)}
        external: Dict[str, int] = {}
        encoded_nodes = []
        internal_types = []
        for node_id in order:
            node = nodes[node_id]
            encoded_inputs = {}
            for port, references in sorted(node.inputs.items()):
                encoded = []
                for reference in references:
                    normalized = _normalized_reference(reference, nodes)
                    root, value_port = _split_reference(reference, nodes)
                    if root in subset:
                        encoded.append("n{}:{}".format(local[root], value_port))
                    else:
                        if normalized not in external:
                            external[normalized] = len(external)
                        encoded.append("in{}".format(external[normalized]))
                encoded_inputs[port] = encoded
            encoded_nodes.append({"op": _qualified(node.op), "inputs": encoded_inputs, "outputs": list(node.outputs)})
            internal_types.append({
                output: _type_for("{}:{}".format(node_id, output), inference, nodes).to_dict()
                for output in node.outputs
            })
        encoded_outputs = []
        output_pairs = []
        for reference in outgoing:
            root, port = _split_reference(reference, nodes)
            encoded = "n{}:{}".format(local[root], port)
            output_pairs.append((encoded, reference))
        for encoded, _reference in sorted(output_pairs):
            if encoded not in encoded_outputs:
                encoded_outputs.append(encoded)
        ordered_external = tuple(reference for reference, _ in sorted(external.items(), key=lambda item: item[1]))
        payload = {
            "nodes": encoded_nodes,
            "internal_types": internal_types,
            "boundary_inputs": [_type_for(ref, inference, nodes).to_dict() for ref in ordered_external],
            "boundary_outputs": [
                _type_for(reference, inference, nodes).to_dict()
                for encoded in encoded_outputs
                for encoded_value, reference in output_pairs
                if encoded_value == encoded
            ],
            "output_sources": encoded_outputs,
        }
        text = _json(payload)
        if best_text is None or text < best_text:
            best_text = text
            best_order = tuple(order)
            best_inputs = ordered_external
            best_outputs = tuple(
                next(reference for encoded_value, reference in output_pairs if encoded_value == encoded)
                for encoded in encoded_outputs
            )
            best_payload = payload
    if best_text is None:
        return None
    input_values = tuple(
        BoundaryValue("in{}".format(index), reference, _type_for(reference, inference, nodes))
        for index, reference in enumerate(best_inputs)
    )
    output_values = tuple(
        BoundaryValue("out{}".format(index), reference, _type_for(reference, inference, nodes))
        for index, reference in enumerate(best_outputs)
    )
    boundary_signature = {
        "inputs": [item.value_type.to_dict() for item in input_values],
        "outputs": [item.value_type.to_dict() for item in output_values],
    }
    topology_hash = hashlib.sha256(best_text.encode("utf-8")).hexdigest()
    occurrence_payload = {
        "architecture_id": candidate.architecture_id,
        "nodes": sorted(node_ids),
        "topology_hash": topology_hash,
    }
    return TypedSubgraphOccurrence(
        _digest("occurrence", occurrence_payload),
        candidate.architecture_id,
        candidate.lineage_id,
        candidate.task_id,
        tuple(sorted(node_ids)),
        best_order,
        input_values,
        output_values,
        best_text,
        topology_hash,
        hashlib.sha256(_json(boundary_signature).encode("utf-8")).hexdigest(),
        candidate.language_registry_hash,
        candidate.rewrite_registry_hash,
        candidate.discovery_partition,
    )


def enumerate_typed_subgraphs(
    candidate: CandidateLineageEvidence,
    policy: Optional[MotifDiscoveryPolicy] = None,
) -> Tuple[TypedSubgraphOccurrence, ...]:
    selected = policy or MotifDiscoveryPolicy()
    output = []
    for node_ids in _connected_node_sets(candidate.artifact.expanded_program, selected):
        occurrence = _canonical_occurrence(candidate, node_ids)
        if occurrence is not None:
            output.append(occurrence)
    return tuple(output)


def _template_reference(reference: str, occurrence: TypedSubgraphOccurrence, nodes: Mapping[str, Node]) -> str:
    root, port = _split_reference(reference, nodes)
    local = {node_id: index for index, node_id in enumerate(occurrence.canonical_node_ids)}
    if root in local:
        return "n{}:{}".format(local[root], port)
    normalized = _normalized_reference(reference, nodes)
    boundary = {item.reference: item.name for item in occurrence.boundary_inputs}
    if normalized not in boundary:
        raise DSLValidationError([
            Diagnostic("E_DISCOVERY_011", "template references an undeclared boundary input", actual=reference)
        ])
    return "$input:{}".format(boundary[normalized])


def _build_proposal(
    occurrences: Sequence[TypedSubgraphOccurrence],
    candidates: Mapping[str, CandidateLineageEvidence],
    primitives: PrimitiveRegistry,
    policy: MotifDiscoveryPolicy,
) -> MotifProposal:
    ordered = tuple(sorted(occurrences, key=lambda item: item.occurrence_id))
    representative = ordered[0]
    rep_candidate = candidates[representative.architecture_id]
    rep_program = rep_candidate.artifact.expanded_program
    rep_nodes = {node.id: node for node in rep_program.nodes}
    aligned_nodes = []
    for occurrence in ordered:
        program_nodes = {node.id: node for node in candidates[occurrence.architecture_id].artifact.expanded_program.nodes}
        aligned_nodes.append(tuple(program_nodes[node_id] for node_id in occurrence.canonical_node_ids))
    parameter_bindings: Dict[str, Dict[str, Any]] = {item.occurrence_id: {} for item in ordered}
    template_nodes = []
    required_attrs = []
    for node_index, rep_node_id in enumerate(representative.canonical_node_ids):
        versions = [items[node_index] for items in aligned_nodes]
        operations = {_qualified(item.op) for item in versions}
        if len(operations) != 1:
            raise DSLValidationError([Diagnostic("E_DISCOVERY_012", "anti-unification aligned different operations")])
        attr_keys = {tuple(sorted(item.attrs)) for item in versions}
        if len(attr_keys) != 1:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_013", "attribute presence differs across motif occurrences")
            ])
        definition = primitives.resolve(versions[0].op)
        attrs = dict(versions[0].attrs)
        for attr in sorted(attrs):
            values = [item.attrs[attr] for item in versions]
            if all(value == values[0] for value in values[1:]):
                continue
            if attr not in definition.motif_parameter_attrs:
                raise DSLValidationError([
                    Diagnostic(
                        "E_DISCOVERY_014",
                        "attribute difference is not declared safe for motif parameterization",
                        actual="{}.{}".format(definition.qualified_name, attr),
                    )
                ])
            parameter = "n{}_{}".format(node_index, attr)
            attrs[attr] = "$attr:{}".format(parameter)
            required_attrs.append(parameter)
            for occurrence, value in zip(ordered, values):
                parameter_bindings[occurrence.occurrence_id][parameter] = value
        rep_node = rep_nodes[rep_node_id]
        inputs = {
            port: tuple(_template_reference(ref, representative, rep_nodes) for ref in references)
            for port, references in rep_node.inputs.items()
        }
        template_nodes.append(
            Node(
                "n{}".format(node_index),
                _qualified(rep_node.op),
                inputs,
                attrs,
                tuple(rep_node.outputs),
                dict(rep_node.declared_types),
                {"learned_motif_node": node_index},
            )
        )
    local = {node_id: index for index, node_id in enumerate(representative.canonical_node_ids)}
    output_bindings = {}
    for item in representative.boundary_outputs:
        root, port = _split_reference(item.reference, rep_nodes)
        output_bindings[item.name] = "n{}:{}".format(local[root], port)
    group_families = tuple(sorted({item.value_type.group.family for occurrence in ordered for item in occurrence.boundary_inputs + occurrence.boundary_outputs}))
    canonical_form_hash = hashlib.sha256(representative.canonical_form.encode("utf-8")).hexdigest()
    motif = MotifDefinition(
        "motif.learned_{}".format(canonical_form_hash[:12]),
        1,
        tuple(item.name for item in representative.boundary_inputs),
        output_bindings,
        tuple(template_nodes),
        tuple(required_attrs),
        group_families,
        provenance={
            "discovery_policy": policy.policy_version,
            "discovery_policy_hash": policy.content_hash(),
            "source_architecture_ids": sorted({item.architecture_id for item in ordered}),
            "source_occurrence_ids": [item.occurrence_id for item in ordered],
        },
        certification="constructive",
        semantic_constraints=(
            "Every expansion consists only of registered typed primitives.",
            "Boundary group, irrep, parity, frame, carrier, axes, dtype, and measure are fixed by discovery.",
            "Every source occurrence must replay to the same semantic architecture identity.",
        ),
        edit_guidance=("Instantiate only through the recorded typed signature and admitted attribute parameters.",),
    )
    raw_length = 0
    call_length = 0
    for occurrence in ordered:
        program_nodes = {node.id: node for node in candidates[occurrence.architecture_id].artifact.expanded_program.nodes}
        raw_length += sum(len(_json(program_nodes[node_id].to_dict())) for node_id in occurrence.node_ids)
        call_length += len(_json({
            "op": motif.qualified_name,
            "inputs": {item.name: item.reference for item in occurrence.boundary_inputs},
            "attrs": parameter_bindings[occurrence.occurrence_id],
            "outputs": list(output_bindings),
        }))
    definition_length = len(_json({
        "inputs": motif.input_ports,
        "outputs": dict(motif.output_bindings),
        "nodes": [item.to_dict() for item in motif.template_nodes],
        "required_attrs": motif.required_attrs,
    }))
    compressed_length = definition_length + call_length
    proof_ids = tuple(
        _digest("proof", {"architecture_id": item.architecture_id, "occurrence_id": item.occurrence_id, "kind": "compiled-constructive"})
        for item in ordered
    )
    proposal_payload = {
        "motif_hash": motif.content_hash(),
        "occurrences": [item.occurrence_id for item in ordered],
        "policy_hash": policy.content_hash(),
        "canonical_form_hash": canonical_form_hash,
    }
    return MotifProposal(
        _digest("motif-proposal", proposal_payload),
        motif,
        ordered,
        parameter_bindings,
        float(raw_length - compressed_length),
        raw_length,
        compressed_length,
        tuple(sorted({item.lineage_id for item in ordered})),
        tuple(sorted({item.task_id for item in ordered})),
        tuple(sorted({item.architecture_id for item in ordered})),
        proof_ids,
        policy.content_hash(),
        canonical_form_hash,
    )


def discover_motif_proposals(
    candidates: Sequence[CandidateLineageEvidence],
    primitives: PrimitiveRegistry,
    policy: Optional[MotifDiscoveryPolicy] = None,
) -> MotifDiscoveryReport:
    selected = policy or MotifDiscoveryPolicy()
    by_architecture: Dict[str, CandidateLineageEvidence] = {}
    occurrences = []
    registry_hashes = set()
    rewrite_hashes = set()
    for candidate in candidates:
        if candidate.architecture_id in by_architecture:
            raise DSLValidationError([
                Diagnostic("E_DISCOVERY_015", "duplicate candidate architecture evidence", actual=candidate.architecture_id)
            ])
        by_architecture[candidate.architecture_id] = candidate
        registry_hashes.add(candidate.language_registry_hash)
        rewrite_hashes.add(candidate.rewrite_registry_hash)
        occurrences.extend(enumerate_typed_subgraphs(candidate, selected))
    if len(registry_hashes) > 1 or len(rewrite_hashes) > 1:
        raise DSLValidationError([
            Diagnostic("E_DISCOVERY_016", "motif discovery window mixes compiler or language semantics")
        ])
    clusters: Dict[str, List[TypedSubgraphOccurrence]] = {}
    for occurrence in occurrences:
        if occurrence.discovery_partition != "support":
            continue
        clusters.setdefault(occurrence.topology_hash, []).append(occurrence)
    proposals = []
    rejected: Dict[str, Tuple[str, ...]] = {}
    for cluster_hash, items in sorted(clusters.items()):
        if len(items) < selected.min_occurrences:
            continue
        non_overlapping = []
        occupied_by_architecture: Dict[str, Set[str]] = {}
        for occurrence in sorted(items, key=lambda item: item.occurrence_id):
            occupied = occupied_by_architecture.setdefault(occurrence.architecture_id, set())
            if occupied.intersection(occurrence.node_ids):
                continue
            non_overlapping.append(occurrence)
            occupied.update(occurrence.node_ids)
        if len(non_overlapping) < selected.min_occurrences:
            rejected[cluster_hash] = ("cluster has fewer than two non-overlapping occurrences",)
            continue
        try:
            proposal = _build_proposal(non_overlapping, by_architecture, primitives, selected)
        except DSLValidationError as error:
            rejected[cluster_hash] = tuple(item.message for item in error.diagnostics)
            continue
        proposals.append(proposal)
    proposals.sort(key=lambda item: (-item.description_length_gain, item.proposal_id))
    if len(proposals) > selected.max_proposals:
        for item in proposals[selected.max_proposals:]:
            rejected[item.canonical_form_hash] = ("proposal exceeded the pre-registered proposal budget",)
        proposals = proposals[:selected.max_proposals]
    return MotifDiscoveryReport(selected, tuple(occurrences), tuple(proposals), rejected)


def fold_occurrence(
    program: ArchitectureProgram,
    occurrence: TypedSubgraphOccurrence,
    proposal: MotifProposal,
    *,
    call_id: Optional[str] = None,
) -> ArchitectureProgram:
    if occurrence.occurrence_id not in proposal.parameter_bindings:
        raise DSLValidationError([
            Diagnostic("E_REPLAY_001", "occurrence is not part of the motif proposal", actual=occurrence.occurrence_id)
        ])
    nodes = {node.id: node for node in program.nodes}
    subset = set(occurrence.node_ids)
    missing = subset - set(nodes)
    if missing:
        raise DSLValidationError([
            Diagnostic("E_REPLAY_002", "occurrence nodes are absent from replay program", details={"missing": sorted(missing)})
        ])
    resolved_call_id = call_id or "learned_{}".format(occurrence.occurrence_id.split(":", 1)[-1][:12])
    if resolved_call_id in nodes:
        raise DSLValidationError([Diagnostic("E_REPLAY_003", "motif call id collides with an existing node", actual=resolved_call_id)])
    output_aliases = {}
    declared_types = {}
    for item in occurrence.boundary_outputs:
        output_aliases[item.reference] = "{}:{}".format(resolved_call_id, item.name)
        root, _port = _split_reference(item.reference, nodes)
        if len(nodes[root].outputs) == 1:
            output_aliases[root] = "{}:{}".format(resolved_call_id, item.name)
        declared_types[item.name] = item.value_type
    call = Node(
        resolved_call_id,
        proposal.motif.qualified_name,
        {item.name: (item.reference,) for item in occurrence.boundary_inputs},
        dict(proposal.parameter_bindings[occurrence.occurrence_id]),
        tuple(item.name for item in occurrence.boundary_outputs),
        declared_types,
        {"folded_occurrence": occurrence.occurrence_id, "proposal_id": proposal.proposal_id},
    )

    def rewrite(reference: str) -> str:
        normalized = _normalized_reference(reference, nodes)
        return output_aliases.get(normalized, output_aliases.get(reference, reference))

    kept = []
    insertion = min(index for index, node in enumerate(program.nodes) if node.id in subset)
    for node in program.nodes:
        if node.id not in subset:
            kept.append(replace(node, inputs={port: tuple(rewrite(ref) for ref in refs) for port, refs in node.inputs.items()}))
    kept.insert(min(insertion, len(kept)), call)
    outputs = tuple(replace(item, source=rewrite(item.source)) for item in program.outputs)
    annotations = dict(program.annotations)
    annotations["folded_motif_proposal"] = proposal.proposal_id
    return replace(program, nodes=tuple(kept), outputs=outputs, annotations=annotations)


def replay_motif_proposal(
    proposal: MotifProposal,
    candidates: Sequence[CandidateLineageEvidence],
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
) -> Tuple[LanguageReplayResult, ...]:
    by_architecture = {item.architecture_id: item for item in candidates}
    replay_registry = MotifRegistry()
    for name in motifs.names():
        replay_registry.register(motifs.resolve(name))
    replay_registry.register(proposal.motif)
    base_compiler = Compiler(primitives, motifs)
    replay_compiler = Compiler(primitives, replay_registry)
    replay_occurrences = list(proposal.occurrences)
    motif_size = len(proposal.motif.template_nodes)
    heldout_policy = MotifDiscoveryPolicy(
        min_nodes=motif_size,
        max_nodes=motif_size,
        min_occurrences=2,
    )
    for candidate in candidates:
        if candidate.discovery_partition != "heldout_replay":
            continue
        replay_occurrences.extend(
            occurrence
            for occurrence in enumerate_typed_subgraphs(candidate, heldout_policy)
            if occurrence.topology_hash == proposal.canonical_form_hash
        )
    replay_occurrences = list({item.occurrence_id: item for item in replay_occurrences}.values())
    bindings = {key: dict(value) for key, value in proposal.parameter_bindings.items()}
    for occurrence in replay_occurrences:
        if occurrence.occurrence_id in bindings:
            continue
        candidate = by_architecture[occurrence.architecture_id]
        program_nodes = {node.id: node for node in candidate.artifact.expanded_program.nodes}
        occurrence_bindings = {}
        compatible = True
        for node_index, template in enumerate(proposal.motif.template_nodes):
            actual = program_nodes[occurrence.canonical_node_ids[node_index]]
            if _qualified(actual.op) != _qualified(template.op) or set(actual.attrs) != set(template.attrs):
                compatible = False
                break
            for attr, template_value in template.attrs.items():
                if isinstance(template_value, str) and template_value.startswith("$attr:"):
                    occurrence_bindings[template_value.split(":", 1)[1]] = actual.attrs[attr]
                elif actual.attrs[attr] != template_value:
                    compatible = False
                    break
            if not compatible:
                break
        if compatible:
            bindings[occurrence.occurrence_id] = occurrence_bindings
    replay_occurrences = [item for item in replay_occurrences if item.occurrence_id in bindings]
    replay_proposal = replace(proposal, parameter_bindings=bindings)
    results = []
    for occurrence in replay_occurrences:
        candidate = by_architecture.get(occurrence.architecture_id)
        if candidate is None:
            raise DSLValidationError([
                Diagnostic("E_REPLAY_004", "proposal replay lacks its source candidate", actual=occurrence.architecture_id)
            ])
        program = candidate.artifact.expanded_program
        before = base_compiler.analyze(program).architecture_id
        diagnostics: Tuple[Mapping[str, Any], ...] = ()
        after = ""
        passed = False
        try:
            folded = fold_occurrence(program, occurrence, replay_proposal)
            after = replay_compiler.analyze(folded).architecture_id
            passed = before == after
            if not passed:
                diagnostics = ({"code": "E_REPLAY_005", "message": "motif folding and expansion changed semantic architecture identity"},)
        except DSLValidationError as error:
            diagnostics = tuple(item.to_dict() for item in error.diagnostics)
        payload = {
            "proposal_id": proposal.proposal_id,
            "occurrence_id": occurrence.occurrence_id,
            "before": before,
            "after": after,
        }
        results.append(
            LanguageReplayResult(
                _digest("motif-replay", payload),
                proposal.proposal_id,
                occurrence.occurrence_id,
                occurrence.architecture_id,
                before,
                after,
                passed,
                diagnostics,
                occurrence.discovery_partition,
            )
        )
    return tuple(results)
