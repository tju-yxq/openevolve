"""Compiler-enforced structural regions for local DSL evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .diagnostics import DSLValidationError, Diagnostic


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class RegionDefinition:
    """One named edit capability with a typed, auditable boundary."""

    region_id: str
    description: str
    editable_targets: Tuple[str, ...]
    boundary_sources: Tuple[str, ...]
    allowed_ops: Tuple[str, ...]
    max_new_nodes: int = 8
    backend_capability: str = "analysis_only"
    invariants: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "description": self.description,
            "editable_targets": list(self.editable_targets),
            "boundary_sources": list(self.boundary_sources),
            "allowed_ops": list(self.allowed_ops),
            "max_new_nodes": self.max_new_nodes,
            "backend_capability": self.backend_capability,
            "invariants": list(self.invariants),
        }


def v1_region_registry(program: ArchitectureProgram) -> Tuple[RegionDefinition, ...]:
    """Return the currently certified regions for an imported V1 program."""

    if program.annotations.get("legacy_backend") != "equiformer_v1":
        return ()
    block_ids = tuple(
        node.id for node in program.nodes
        if node.id.startswith("block") and node.id[5:].isdigit()
    )
    auxiliary_blocks = tuple(item for item in block_ids[:-1])
    if not auxiliary_blocks:
        return ()
    return (
        RegionDefinition(
            region_id="v1_readout",
            description=(
                "Replace the terminal graph pooling call with a constructively typed "
                "multi-level invariant readout that may tap exactly one earlier V1 block."
            ),
            editable_targets=("graph_pool", "output:prediction"),
            boundary_sources=("scalar_readout",) + auxiliary_blocks,
            allowed_ops=("motif.v1_multilevel_readout@1",),
            max_new_nodes=0,
            backend_capability="exact_hybrid",
            invariants=(
                "All V1 embedding and attention blocks remain structurally identical.",
                "scalar_readout still consumes block5, the terminal official V1 feature block.",
                "The terminal official V1 graph prediction remains one input of the runtime combiner.",
                "The auxiliary branch consumes scalar irreps from exactly one earlier block.",
                "The task output remains one graph-carried scalar.",
                "The hybrid backend adds a trainable auxiliary scalar head and a trainable two-input combiner; it is not parameter-free.",
                "max_new_nodes counts source-AST nodes; the selected motif expands into certified internal nodes.",
            ),
        ),
    )


def region_by_id(regions: Sequence[RegionDefinition], region_id: str) -> RegionDefinition:
    matches = [item for item in regions if item.region_id == region_id]
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic("E_REGION_001", "router selected an unknown or duplicate region", actual=region_id)
        ])
    return matches[0]


def _node_map(program: ArchitectureProgram) -> Mapping[str, Node]:
    return {node.id: node for node in program.nodes}


def frozen_complement_payload(
    program: ArchitectureProgram,
    region: RegionDefinition,
    *,
    original_node_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    """Canonical payload for all original structure outside an edit region."""

    editable_nodes = {item for item in region.editable_targets if not item.startswith("output:")}
    editable_outputs = {
        item.split(":", 1)[1] for item in region.editable_targets if item.startswith("output:")
    }
    original = set(original_node_ids) if original_node_ids else {node.id for node in program.nodes}
    nodes = [
        node.to_dict() for node in program.nodes
        if node.id in original and node.id not in editable_nodes
    ]
    outputs = [
        output.to_dict() for output in program.outputs
        if output.name not in editable_outputs
    ]
    return {"nodes": nodes, "outputs": outputs}


def frozen_complement_hash(
    program: ArchitectureProgram,
    region: RegionDefinition,
    *,
    original_node_ids: Sequence[str] = (),
) -> str:
    payload = frozen_complement_payload(program, region, original_node_ids=original_node_ids)
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def validate_region_transition(
    parent: ArchitectureProgram,
    child: ArchitectureProgram,
    region: RegionDefinition,
) -> Dict[str, Any]:
    """Prove that a child changes only the selected region and its declared boundary."""

    parent_ids = tuple(node.id for node in parent.nodes)
    parent_nodes = _node_map(parent)
    child_nodes = _node_map(child)
    editable_nodes = {item for item in region.editable_targets if not item.startswith("output:")}
    inserted = sorted(set(child_nodes) - set(parent_nodes))
    deleted = sorted(set(parent_nodes) - set(child_nodes))
    if len(inserted) > region.max_new_nodes:
        raise DSLValidationError([
            Diagnostic(
                "E_REGION_002",
                "patch exceeded the selected region new-node budget",
                details={"inserted": inserted, "limit": region.max_new_nodes},
            )
        ])
    illegal_deleted = sorted(set(deleted) - editable_nodes)
    if illegal_deleted:
        raise DSLValidationError([
            Diagnostic("E_REGION_003", "patch deleted frozen nodes", details={"nodes": illegal_deleted})
        ])
    parent_hash = frozen_complement_hash(parent, region, original_node_ids=parent_ids)
    child_hash = frozen_complement_hash(child, region, original_node_ids=parent_ids)
    if parent_hash != child_hash:
        raise DSLValidationError([
            Diagnostic(
                "E_REGION_004",
                "patch changed the frozen structural complement",
                expected=parent_hash,
                actual=child_hash,
            )
        ])

    region_nodes = editable_nodes | set(inserted)
    allowed_roots = set(region.boundary_sources) | region_nodes
    for node_id in sorted(region_nodes & set(child_nodes)):
        node = child_nodes[node_id]
        qualified = node.op if "@" in node.op else node.op + "@1"
        if region.allowed_ops and node_id in editable_nodes and qualified not in region.allowed_ops:
            raise DSLValidationError([
                Diagnostic(
                    "E_REGION_005",
                    "edited region root uses an unsupported operation",
                    node_id=node_id,
                    actual=qualified,
                    details={"allowed_ops": list(region.allowed_ops)},
                )
            ])
        for refs in node.inputs.values():
            for reference in refs:
                root = reference.split(":", 1)[0]
                if root.startswith("input:"):
                    continue
                if root not in allowed_roots:
                    raise DSLValidationError([
                        Diagnostic(
                            "E_REGION_006",
                            "edited region reads an undeclared boundary source",
                            node_id=node_id,
                            actual=reference,
                            details={"boundary_sources": list(region.boundary_sources)},
                        )
                    ])
    return {
        "region_id": region.region_id,
        "frozen_complement_hash": child_hash,
        "inserted_node_ids": inserted,
        "deleted_node_ids": deleted,
        "edited_target_ids": sorted(editable_nodes & set(child_nodes)),
    }
