"""Canonical serialization and version-bound architecture fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Tuple

from .ast import ArchitectureProgram, Node
from .inference import TypeChecker
from .registry import PrimitiveRegistry


_COMMUTATIVE_PORTS = {
    "core.residual_add@1": ("left", "right"),
    "core.irrep_concat@1": ("xs",),
}


def canonicalize(program: ArchitectureProgram, registry: Optional[PrimitiveRegistry] = None) -> ArchitectureProgram:
    """Return a stable graph order with normalized commutative inputs.

    Node names remain stable source locations. They are excluded from no semantic
    checks, but normalization prevents source tuple order and dictionary order
    from changing a candidate fingerprint.
    """

    normalized = []
    for node in program.nodes:
        qualified = node.op if "@" in node.op else "{}@1".format(node.op)
        inputs = {name: tuple(refs) for name, refs in node.inputs.items()}
        for port in _COMMUTATIVE_PORTS.get(qualified, ()):
            if port in inputs:
                inputs[port] = tuple(sorted(inputs[port]))
        normalized.append(replace(node, op=qualified, inputs=inputs))
    candidate = replace(program, nodes=tuple(normalized))
    if registry is not None:
        TypeChecker(registry, require_closed_obligations=False)._topological_order(candidate, {node.id: node for node in candidate.nodes})

    by_id = {node.id: node for node in candidate.nodes}
    memo = {}

    def structural_signature(node_id):
        if node_id in memo:
            return memo[node_id]
        node = by_id[node_id]
        encoded_inputs = []
        depth = 0
        for port, refs in sorted(node.inputs.items()):
            encoded_refs = []
            for reference in refs:
                root, separator, output_port = reference.partition(":")
                if root == "input":
                    encoded_refs.append("input:{}".format(output_port))
                elif root in by_id:
                    parent_depth, parent_signature = structural_signature(root)
                    depth = max(depth, parent_depth + 1)
                    encoded_refs.append("node:{}:{}".format(parent_signature, output_port or "out"))
                else:
                    encoded_refs.append(reference)
            encoded_inputs.append((port, tuple(encoded_refs)))
        semantic = {
            "op": node.op,
            "inputs": encoded_inputs,
            "attrs": dict(node.attrs),
            "outputs": tuple(node.outputs),
            "declared_types": {key: value.to_dict() for key, value in sorted(node.declared_types.items())},
        }
        signature = hashlib.sha256(
            json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        memo[node_id] = (depth, signature)
        return memo[node_id]

    ordered = sorted(candidate.nodes, key=lambda item: structural_signature(item.id) + (item.id,))
    rename = {node.id: "n{:04d}".format(index) for index, node in enumerate(ordered)}

    def rewrite(reference):
        root, separator, output_port = reference.partition(":")
        if root not in rename:
            return reference
        return "{}:{}".format(rename[root], output_port) if separator else rename[root]

    canonical_nodes = tuple(
        replace(
            node,
            id=rename[node.id],
            inputs={port: tuple(rewrite(ref) for ref in refs) for port, refs in sorted(node.inputs.items())},
        )
        for node in ordered
    )
    canonical_outputs = tuple(replace(item, source=rewrite(item.source)) for item in candidate.outputs)
    return replace(candidate, nodes=canonical_nodes, outputs=canonical_outputs)


def semantic_dict(program: ArchitectureProgram, registry: Optional[PrimitiveRegistry] = None) -> Dict[str, Any]:
    data = canonicalize(program, registry).to_dict()
    data.pop("program_id", None)
    data.pop("annotations", None)
    for node in data["nodes"]:
        node.pop("annotations", None)
    return data


def canonical_json(program: ArchitectureProgram, registry: Optional[PrimitiveRegistry] = None) -> str:
    return json.dumps(semantic_dict(program, registry), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def architecture_id(
    program: ArchitectureProgram,
    registry: Optional[PrimitiveRegistry] = None,
    *,
    kernel_registry_hash: str = "builtin-core-v1",
    compiler_version: str = "evoequilang-1",
    task_contract_hash: str = "unresolved-task-contract",
    backend_semantics_version: str = "backend-neutral-v1",
) -> str:
    payload = {
        "canonical_ast": canonical_json(program, registry),
        "language_version": program.language_version,
        "kernel_registry_hash": kernel_registry_hash,
        "compiler_version": compiler_version,
        "task_contract_hash": task_contract_hash,
        "backend_semantics_version": backend_semantics_version,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
