"""Canonical serialization and version-bound architecture fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional

from .ast import ArchitectureProgram
from .inference import TypeChecker
from .parameters import PARAMETER_CONTRACT_SCHEMA_VERSION
from .registry import PrimitiveRegistry
from .rewrites import apply_strict_rewrites, strict_rewrite_registry_hash
from .types import VALUE_TYPE_SCHEMA_VERSION


COMPILER_SEMANTICS_VERSION = "evoequilang-23"
BACKEND_SEMANTICS_VERSION = "backend-neutral-v21"


def canonicalize(program: ArchitectureProgram, registry: Optional[PrimitiveRegistry] = None) -> ArchitectureProgram:
    """Return a stable graph order with normalized commutative inputs.

    Node names remain stable source locations. They are excluded from no semantic
    checks, but normalization prevents source tuple order and dictionary order
    from changing a candidate fingerprint.
    """

    strictly_rewritten = apply_strict_rewrites(program).program
    normalized = []
    for node in strictly_rewritten.nodes:
        qualified = node.op if "@" in node.op else "{}@1".format(node.op)
        inputs = {name: tuple(refs) for name, refs in node.inputs.items()}
        attrs = dict(node.attrs)
        if registry is not None:
            attrs = registry.resolve(qualified).canonical_attrs(node.id, attrs)
        normalized.append(replace(node, op=qualified, inputs=inputs, attrs=attrs))
    candidate = replace(strictly_rewritten, nodes=tuple(normalized))
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
        outputs = by_id[root].outputs
        if len(outputs) == 1 and (not separator or output_port == outputs[0]):
            return rename[root]
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
    parameters = dict(candidate.parameters)
    lowering_contract = parameters.get("lowering_contract")
    if isinstance(lowering_contract, Mapping):
        lowering_contract = dict(lowering_contract)
        raw_order = lowering_contract.get("module_construction_order")
        if isinstance(raw_order, (list, tuple)):
            lowering_contract["module_construction_order"] = [
                rename.get(str(node_id), str(node_id)) for node_id in raw_order
            ]
        raw_schedule = lowering_contract.get("initializer_schedule")
        if isinstance(raw_schedule, (list, tuple)):
            rewritten_schedule = []
            for item in raw_schedule:
                if not isinstance(item, Mapping):
                    rewritten_schedule.append(item)
                    continue
                event = dict(item)
                if "after_node" in event:
                    event["after_node"] = rename.get(str(event["after_node"]), str(event["after_node"]))
                if isinstance(event.get("nodes"), (list, tuple)):
                    event["nodes"] = [
                        rename.get(str(node_id), str(node_id)) for node_id in event["nodes"]
                    ]
                rewritten_schedule.append(event)
            lowering_contract["initializer_schedule"] = rewritten_schedule
        raw_shared_norm = lowering_contract.get("shared_final_norm")
        if isinstance(raw_shared_norm, Mapping):
            shared_norm = dict(raw_shared_norm)
            if "node" in shared_norm:
                shared_norm["node"] = rename.get(
                    str(shared_norm["node"]),
                    str(shared_norm["node"]),
                )
            if isinstance(shared_norm.get("consumers"), (list, tuple)):
                shared_norm["consumers"] = [
                    rename.get(str(node_id), str(node_id))
                    for node_id in shared_norm["consumers"]
                ]
            lowering_contract["shared_final_norm"] = shared_norm
        parameters["lowering_contract"] = lowering_contract
    return replace(
        candidate,
        nodes=canonical_nodes,
        outputs=canonical_outputs,
        parameters=parameters,
    )


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
    kernel_registry_hash: Optional[str] = None,
    compiler_version: str = COMPILER_SEMANTICS_VERSION,
    task_contract_hash: str = "unresolved-task-contract",
    backend_semantics_version: str = BACKEND_SEMANTICS_VERSION,
    rewrite_registry_hash: str = "",
    value_type_schema_version: Optional[str] = VALUE_TYPE_SCHEMA_VERSION,
) -> str:
    canonical_program = canonicalize(program, registry)
    canonical_data = canonical_program.to_dict()
    canonical_data.pop("program_id", None)
    canonical_data.pop("annotations", None)
    for node in canonical_data["nodes"]:
        node.pop("annotations", None)
    parameter_contracts = {}
    if registry is not None:
        inference = TypeChecker(registry, require_closed_obligations=False).check(canonical_program)
        parameter_contracts = {
            node_id: [
                contract.to_dict()
                for contract in contracts
                if contract.architecture_identity != "excluded"
            ]
            for node_id, contracts in sorted(inference.parameter_contracts.items())
            if any(contract.architecture_identity != "excluded" for contract in contracts)
        }
    payload = {
        "canonical_ast": json.dumps(canonical_data, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        "language_version": program.language_version,
        "kernel_registry_hash": kernel_registry_hash or (registry.content_hash() if registry is not None else "builtin-core-v1"),
        "compiler_version": compiler_version,
        "rewrite_registry_hash": rewrite_registry_hash or strict_rewrite_registry_hash(),
        "task_contract_hash": task_contract_hash,
        "backend_semantics_version": backend_semantics_version,
        "parameter_contract_schema_version": PARAMETER_CONTRACT_SCHEMA_VERSION,
        "parameter_contracts": parameter_contracts,
    }
    if value_type_schema_version is not None:
        payload["value_type_schema_version"] = str(value_type_schema_version)
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
