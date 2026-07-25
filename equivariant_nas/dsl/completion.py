"""Typed partial-program completion and Syno-inspired completion distances."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import TypeChecker
from .obligations import ProofObligation
from .patch import PatchEdit, TypedPatch
from .registry import PrimitiveRegistry
from .types import Carrier, EquivariantType, Frame


@dataclass(frozen=True)
class AvailableValue:
    reference: str
    value_type: EquivariantType

    def to_dict(self) -> Dict[str, Any]:
        return {"reference": self.reference, "value_type": self.value_type.to_dict()}


@dataclass(frozen=True)
class CompletionAction:
    node_id: str
    op: str
    inputs: Mapping[str, Tuple[str, ...]]
    attrs: Mapping[str, Any]
    output_type: EquivariantType
    output_name: str = "out"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "op": self.op,
            "inputs": {name: list(refs) for name, refs in self.inputs.items()},
            "attrs": dict(self.attrs),
            "output_type": self.output_type.to_dict(),
            "output_name": self.output_name,
        }


@dataclass(frozen=True)
class TypedHole:
    hole_id: str
    expected_type: EquivariantType
    available_values: Tuple[AvailableValue, ...]
    allowed_ops: Tuple[str, ...] = ()
    max_steps: int = 4

    def __post_init__(self) -> None:
        if not self.hole_id or ":" in self.hole_id:
            raise ValueError("hole_id is required and cannot contain ':'")
        if not self.available_values:
            raise ValueError("a typed hole requires at least one available value")
        if self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hole_id": self.hole_id,
            "expected_type": self.expected_type.to_dict(),
            "available_values": [item.to_dict() for item in self.available_values],
            "allowed_ops": list(self.allowed_ops),
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class HoleSink:
    """A capability-scoped destination for a completed value."""

    kind: str
    target: str
    port: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("input_port", "program_output"):
            raise ValueError("hole sink kind must be input_port or program_output")
        if not self.target:
            raise ValueError("hole sink target is required")
        if self.kind == "input_port" and not self.port:
            raise ValueError("input_port sink requires a port")
        if self.kind == "program_output" and self.port:
            raise ValueError("program_output sink cannot declare a port")

    @classmethod
    def input_port(cls, node_id: str, port: str) -> "HoleSink":
        return cls("input_port", node_id, port)

    @classmethod
    def program_output(cls, output_name: str) -> "HoleSink":
        return cls("program_output", output_name)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "port": self.port}


@dataclass(frozen=True)
class CompletionDistance:
    reachable: bool
    minimum_steps: Optional[int]
    search_cost: Optional[int]
    representation_distance: Optional[int]
    frame_distance: Optional[int]
    carrier_distance: Optional[int]
    invariant_distance: Optional[int]
    backend_distance: Optional[int]
    proof_gap: int
    path: Tuple[CompletionAction, ...] = ()
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reachable": self.reachable,
            "minimum_steps": self.minimum_steps,
            "search_cost": self.search_cost,
            "representation_distance": self.representation_distance,
            "frame_distance": self.frame_distance,
            "carrier_distance": self.carrier_distance,
            "invariant_distance": self.invariant_distance,
            "backend_distance": self.backend_distance,
            "proof_gap": self.proof_gap,
            "path": [item.to_dict() for item in self.path],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _State:
    reference: str
    value_type: EquivariantType
    cost: int
    search_cost: int
    path: Tuple[CompletionAction, ...]


_OPERATION_SEARCH_COST = {
    "core.tensor_product@1": 3,
    "core.irrep_concat@1": 2,
    "core.so2_convolution@1": 3,
}


def _type_key(value_type: EquivariantType) -> str:
    return json.dumps(value_type.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _action_id(hole_id: str, op: str, inputs: Mapping[str, Tuple[str, ...]], attrs: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"op": op, "inputs": {key: list(value) for key, value in sorted(inputs.items())}, "attrs": dict(attrs)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return "{}__{}".format(hole_id, suffix)


def _merge_paths(*paths: Sequence[CompletionAction]) -> Tuple[CompletionAction, ...]:
    seen = set()
    merged = []
    for path in paths:
        for action in path:
            if action.node_id not in seen:
                seen.add(action.node_id)
                merged.append(action)
    return tuple(merged)


def _try_action(
    registry: PrimitiveRegistry,
    hole_id: str,
    op: str,
    inputs: Mapping[str, Tuple[_State, ...]],
    attrs: Mapping[str, Any],
) -> Optional[Tuple[CompletionAction, int, int, Tuple[CompletionAction, ...]]]:
    try:
        definition = registry.resolve(op)
        typed_inputs = {name: tuple(item.value_type for item in states) for name, states in inputs.items()}
        outputs, _ = definition.type_rule("{}__probe".format(hole_id), typed_inputs, attrs)
    except (DSLValidationError, KeyError, TypeError, ValueError):
        return None
    if len(outputs) != 1:
        return None
    output_name, output_type = next(iter(outputs.items()))
    refs = {name: tuple(item.reference for item in states) for name, states in inputs.items()}
    node_id = _action_id(hole_id, definition.qualified_name, refs, attrs)
    action = CompletionAction(node_id, definition.qualified_name, refs, dict(attrs), output_type, output_name)
    parents = tuple(state for states in inputs.values() for state in states)
    path = _merge_paths(*(item.path for item in parents)) + (action,)
    search_cost = sum(_OPERATION_SEARCH_COST.get(item.op, 1) for item in path)
    return action, len(path), search_cost, path


def _scalar_only(value_type: EquivariantType) -> bool:
    return bool(value_type.irreps.terms) and all(ir.degree == 0 for _, ir in value_type.irreps)


def _candidate_actions(
    hole: TypedHole,
    states: Sequence[_State],
    registry: PrimitiveRegistry,
    allowed: Sequence[str],
) -> Iterable[Tuple[CompletionAction, int, int, Tuple[CompletionAction, ...]]]:
    target = hole.expected_type
    allowed_set = set(allowed)

    def enabled(name: str) -> bool:
        qualified = name if "@" in name else "{}@1".format(name)
        return qualified in allowed_set

    for source in states:
        same_static_context = (
            source.value_type.group == target.group
            and source.value_type.axes == target.axes
            and source.value_type.dtype == target.dtype
            and source.value_type.measure == target.measure
        )
        proposals = []
        if same_static_context:
            target_in_source_context = EquivariantType(
                target.group,
                source.value_type.carrier,
                target.irreps,
                source.value_type.frame,
                target.axes,
                target.dtype,
                target.measure,
                source.value_type.level,
            )
            proposals.extend((
                ("core.irrep_linear", {"x": (source,)}, {"out_irreps": str(target.irreps)}),
                ("core.change_multiplicity", {"x": (source,)}, {"out_irreps": str(target.irreps)}),
                ("core.irrep_slice", {"x": (source,)}, {"irreps": str(target.irreps)}),
            ))
            if _scalar_only(target_in_source_context):
                scalar_multiplicity = sum(mul for mul, _ in target.irreps)
                proposals.append(("core.select_scalars", {"x": (source,)}, {"multiplicity": scalar_multiplicity}))
        if source.value_type.carrier == Carrier.NODE:
            proposals.extend((
                ("core.edge_lift", {"x": (source,)}, {}),
                ("core.global_pool", {"x": (source,)}, {}),
            ))
        if source.value_type.carrier == Carrier.EDGE:
            proposals.extend((
                ("core.segment_sum", {"x": (source,)}, {}),
                ("core.segment_mean", {"x": (source,)}, {}),
            ))
        if source.value_type.group.dimension == 3 and source.value_type.frame.kind == "global":
            frame_id = target.frame.reference if target.frame.kind == "edge" else "completion_frame"
            proposals.append(("core.to_edge_frame", {"x": (source,)}, {"frame_id": frame_id}))
        if source.value_type.frame.kind == "edge":
            proposals.append(("core.from_edge_frame", {"x": (source,)}, {"frame_id": source.value_type.frame.reference}))
        for op, inputs, attrs in proposals:
            if enabled(op):
                result = _try_action(registry, hole.hole_id, op, inputs, attrs)
                if result is not None:
                    yield result

    for index, left in enumerate(states):
        for right in states[index:]:
            if not (
                left.value_type.group == right.value_type.group == target.group
                and left.value_type.carrier == right.value_type.carrier
                and left.value_type.frame == right.value_type.frame
                and left.value_type.axes == right.value_type.axes == target.axes
                and left.value_type.measure == right.value_type.measure == target.measure
            ):
                continue
            if enabled("core.tensor_product"):
                result = _try_action(
                    registry,
                    hole.hole_id,
                    "core.tensor_product",
                    {"left": (left,), "right": (right,)},
                    {"out_irreps": str(target.irreps)},
                )
                if result is not None:
                    yield result
            if enabled("core.irrep_concat"):
                result = _try_action(
                    registry,
                    hole.hole_id,
                    "core.irrep_concat",
                    {"xs": (left, right)},
                    {},
                )
                if result is not None:
                    yield result


def _carrier_lower_bound(sources: Sequence[EquivariantType], target: EquivariantType) -> Optional[int]:
    edges = {
        Carrier.NODE: {Carrier.EDGE: 1, Carrier.GRAPH: 1},
        Carrier.EDGE: {Carrier.NODE: 1},
        Carrier.GRAPH: {},
        Carrier.GRID: {},
        Carrier.PAIR: {},
    }
    best = None
    for source in sources:
        if source.group != target.group:
            continue
        if source.carrier == target.carrier:
            value = 0
        else:
            frontier = [(source.carrier, 0)]
            visited = set()
            value = None
            while frontier:
                carrier, distance = frontier.pop(0)
                if carrier in visited:
                    continue
                visited.add(carrier)
                if carrier == target.carrier:
                    value = distance
                    break
                for next_carrier in edges.get(carrier, {}):
                    frontier.append((next_carrier, distance + 1))
        if value is not None:
            best = value if best is None else min(best, value)
    return best


def _frame_lower_bound(sources: Sequence[EquivariantType], target: EquivariantType) -> Optional[int]:
    distances = []
    for source in sources:
        if source.group != target.group:
            continue
        if source.frame == target.frame:
            distances.append(0)
        elif source.group.dimension == 3 and {source.frame.kind, target.frame.kind} <= {"global", "edge"}:
            distances.append(1)
        elif source.frame.kind == "invariant" and target.frame.kind == "invariant":
            distances.append(0)
    return min(distances) if distances else None


def complete_typed_hole(
    hole: TypedHole,
    registry: PrimitiveRegistry,
    *,
    backend_supported_ops: Sequence[str] = (),
    open_obligations: Sequence[ProofObligation] = (),
) -> CompletionDistance:
    """Find a minimum-operation legal completion path for a typed hole.

    The search space is finite because attribute synthesis is goal-directed and
    only one- and two-input trusted core operations are considered.
    """

    allowed = tuple(
        name for name in (hole.allowed_ops or registry.names())
        if not name.startswith("motif.")
    )
    initial = tuple(_State(item.reference, item.value_type, 0, 0, ()) for item in hole.available_values)
    sources = tuple(item.value_type for item in initial)
    target_key = _type_key(hole.expected_type)
    best: Dict[str, _State] = {}
    for state in initial:
        key = _type_key(state.value_type)
        if key not in best or state.reference < best[key].reference:
            best[key] = state
    if target_key in best:
        selected = best[target_key]
        return CompletionDistance(True, 0, 0, 0, 0, 0, 0, 0, len(tuple(open_obligations)), selected.path)

    changed = True
    while changed:
        changed = False
        snapshot = tuple(sorted(best.values(), key=lambda item: (item.search_cost, item.cost, _type_key(item.value_type), item.reference)))
        for action, cost, search_cost, path in _candidate_actions(hole, snapshot, registry, allowed):
            if cost > hole.max_steps:
                continue
            key = _type_key(action.output_type)
            candidate = _State(action.node_id, action.output_type, cost, search_cost, path)
            previous = best.get(key)
            if previous is None or (candidate.search_cost, candidate.cost, candidate.reference) < (
                previous.search_cost,
                previous.cost,
                previous.reference,
            ):
                best[key] = candidate
                changed = True

    selected = best.get(target_key)
    representation_distance = min(
        (0 if item.irreps == hole.expected_type.irreps else 1)
        for item in sources
        if item.group == hole.expected_type.group
    ) if any(item.group == hole.expected_type.group for item in sources) else None
    carrier_distance = _carrier_lower_bound(sources, hole.expected_type)
    frame_distance = _frame_lower_bound(sources, hole.expected_type)
    invariant_distance = 0 if any(
        item.group == hole.expected_type.group and _scalar_only(item) and item.irreps == hole.expected_type.irreps
        for item in sources
    ) else (1 if _scalar_only(hole.expected_type) else 0)
    proof_gap = len(tuple(open_obligations))
    if selected is None:
        return CompletionDistance(
            False,
            None,
            None,
            representation_distance,
            frame_distance,
            carrier_distance,
            invariant_distance,
            None,
            proof_gap,
            (),
            "no legal completion found within {} trusted operations".format(hole.max_steps),
        )
    supported = {name if "@" in name else "{}@1".format(name) for name in backend_supported_ops}
    backend_distance = sum(1 for action in selected.path if supported and action.op not in supported)
    return CompletionDistance(
        True,
        selected.cost,
        selected.search_cost,
        representation_distance,
        frame_distance,
        carrier_distance,
        invariant_distance,
        backend_distance,
        proof_gap,
        selected.path,
    )


def program_completion_frontier(
    program: ArchitectureProgram,
    target: EquivariantType,
    registry: PrimitiveRegistry,
    *,
    allowed_ops: Sequence[str] = (),
    max_steps: int = 3,
) -> Tuple[Mapping[str, Any], ...]:
    """Estimate how each declared source value can legally reach a target type."""

    records = []
    for node in program.nodes:
        for output_name, value_type in sorted(node.declared_types.items()):
            reference = node.id if len(node.outputs) == 1 else "{}:{}".format(node.id, output_name)
            result = complete_typed_hole(
                TypedHole(
                    "frontier_{}_{}".format(node.id, output_name),
                    target,
                    (AvailableValue(reference, value_type),),
                    tuple(allowed_ops),
                    max_steps,
                ),
                registry,
            )
            records.append({
                "source_node": node.id,
                "source_output": output_name,
                "source_type": value_type.to_dict(),
                "target_type": target.to_dict(),
                "distance": result.to_dict(),
            })
    return tuple(records)


def materialize_completion_patch(
    parent: ArchitectureProgram,
    hole: TypedHole,
    result: CompletionDistance,
    sink: HoleSink,
    registry: PrimitiveRegistry,
    *,
    task_contract_hash: str = "unresolved-task-contract",
) -> TypedPatch:
    """Turn a trusted completion path into an executable, auditable typed patch."""

    if not result.reachable:
        raise DSLValidationError([
            Diagnostic("E_COMPLETE_001", "cannot materialize an unreachable typed hole", actual=result.reason)
        ])
    if result.path and result.path[-1].output_type != hole.expected_type:
        raise DSLValidationError([
            Diagnostic(
                "E_COMPLETE_002",
                "completion path does not end at the hole's expected type",
                expected=str(hole.expected_type.to_dict()),
                actual=str(result.path[-1].output_type.to_dict()),
            )
        ])

    TypeChecker(registry).check(parent)
    existing_ids = {node.id for node in parent.nodes}
    action_ids = [action.node_id for action in result.path]
    if len(action_ids) != len(set(action_ids)) or existing_ids.intersection(action_ids):
        raise DSLValidationError([
            Diagnostic(
                "E_COMPLETE_003",
                "completion action ids must be unique and absent from the parent",
                details={"action_ids": action_ids, "collisions": sorted(existing_ids.intersection(action_ids))},
            )
        ])

    materialized_nodes = tuple(
        Node(
            action.node_id,
            action.op,
            action.inputs,
            action.attrs,
            (action.output_name,),
            {action.output_name: action.output_type},
            {"origin": "typed_completion", "hole_id": hole.hole_id},
        )
        for action in result.path
    )

    if result.path:
        final_action = result.path[-1]
        final_reference = (
            final_action.node_id
            if final_action.output_name == "out"
            else "{}:{}".format(final_action.node_id, final_action.output_name)
        )
    else:
        matching = sorted(
            item.reference for item in hole.available_values if item.value_type == hole.expected_type
        )
        if not matching:
            raise DSLValidationError([
                Diagnostic("E_COMPLETE_004", "zero-step completion has no source with the expected type")
            ])
        final_reference = matching[0]

    edits = []
    preconditions = []
    postconditions = []
    scope = []

    if sink.kind == "input_port":
        consumer = next((node for node in parent.nodes if node.id == sink.target), None)
        if consumer is None:
            raise DSLValidationError([
                Diagnostic("E_COMPLETE_005", "input-port sink names an unknown node", actual=sink.target)
            ])
        if sink.port not in consumer.inputs:
            raise DSLValidationError([
                Diagnostic("E_COMPLETE_006", "input-port sink names an unknown port", node_id=sink.target, port=sink.port)
            ])
        scope.append(sink.target)
        preconditions.append({
            "kind": "node_input_equals",
            "node_id": sink.target,
            "port": sink.port,
            "references": list(consumer.inputs[sink.port]),
        })
        for node in materialized_nodes:
            edits.append(PatchEdit("insert_before", sink.target, {"node": node.to_dict()}))
        edits.append(PatchEdit("rewire_port", sink.target, {"port": sink.port, "references": [final_reference]}))
        postconditions.append({
            "kind": "node_input_equals",
            "node_id": sink.target,
            "port": sink.port,
            "references": [final_reference],
        })
    else:
        output = next((item for item in parent.outputs if item.name == sink.target), None)
        if output is None:
            raise DSLValidationError([
                Diagnostic("E_COMPLETE_007", "program-output sink names an unknown output", actual=sink.target)
            ])
        if output.expected_type != hole.expected_type:
            raise DSLValidationError([
                Diagnostic(
                    "E_COMPLETE_008",
                    "program-output sink type differs from the typed-hole target",
                    expected=str(output.expected_type.to_dict()),
                    actual=str(hole.expected_type.to_dict()),
                )
            ])
        output_scope = "output:{}".format(sink.target)
        scope.append(output_scope)
        preconditions.append({
            "kind": "output_source_is",
            "output": sink.target,
            "reference": output.source,
        })
        if result.path:
            anchor = output.source.split(":", 1)[0]
            if anchor not in existing_ids:
                raise DSLValidationError([
                    Diagnostic(
                        "E_COMPLETE_009",
                        "a nonempty output completion requires an existing output-source node as insertion anchor",
                        actual=output.source,
                    )
                ])
            scope.append(anchor)
            for node in materialized_nodes:
                edits.append(PatchEdit("insert_after", anchor, {"node": node.to_dict()}))
                anchor = node.id
        edits.append(PatchEdit("rewire_output", output_scope, {"reference": final_reference}))
        postconditions.append({
            "kind": "output_source_is",
            "output": sink.target,
            "reference": final_reference,
        })

    hypothetical_nodes = list(parent.nodes) + list(materialized_nodes)
    if sink.kind == "input_port":
        hypothetical_nodes = [
            Node(
                node.id,
                node.op,
                {
                    port: ((final_reference,) if node.id == sink.target and port == sink.port else references)
                    for port, references in node.inputs.items()
                },
                node.attrs,
                node.outputs,
                node.declared_types,
                node.annotations,
            )
            for node in hypothetical_nodes
        ]
        hypothetical_outputs = tuple(parent.outputs)
    else:
        hypothetical_outputs = tuple(
            type(output)(output.name, final_reference if output.name == sink.target else output.source, output.expected_type)
            for output in parent.outputs
        )

    by_id = {node.id: node for node in hypothetical_nodes}
    live = set()
    pending = [output.source.split(":", 1)[0] for output in hypothetical_outputs]
    while pending:
        node_id = pending.pop()
        if node_id in live or node_id not in by_id:
            continue
        live.add(node_id)
        for references in by_id[node_id].inputs.values():
            pending.extend(reference.split(":", 1)[0] for reference in references)

    dead_existing = existing_ids - live
    remaining = set(dead_existing)
    deletion_order = []
    while remaining:
        referenced = {
            reference.split(":", 1)[0]
            for node in hypothetical_nodes
            if node.id in remaining
            for references in node.inputs.values()
            for reference in references
            if reference.split(":", 1)[0] in remaining
        }
        leaves = sorted(remaining - referenced)
        if not leaves:
            raise DSLValidationError([
                Diagnostic("E_COMPLETE_010", "dead-node pruning encountered a cyclic dependency")
            ])
        deletion_order.extend(leaves)
        remaining.difference_update(leaves)
        hypothetical_nodes = [node for node in hypothetical_nodes if node.id not in leaves]

    for node_id in deletion_order:
        scope.append(node_id)
        preconditions.append({"kind": "node_exists", "node_id": node_id})
        edits.append(PatchEdit("delete_if_bypassed", node_id, {"replacement_reference": final_reference}))
        postconditions.append({"kind": "node_absent", "node_id": node_id})

    for action in result.path:
        postconditions.extend((
            {"kind": "node_exists", "node_id": action.node_id},
            {"kind": "node_op_is", "node_id": action.node_id, "op": action.op},
        ))
        for port, references in sorted(action.inputs.items()):
            postconditions.append({
                "kind": "node_input_equals",
                "node_id": action.node_id,
                "port": port,
                "references": list(references),
            })
        for attr, value in sorted(action.attrs.items()):
            postconditions.append({
                "kind": "node_attr_equals",
                "node_id": action.node_id,
                "attr": attr,
                "value": value,
            })

    return TypedPatch(
        "1.0",
        architecture_id(parent, registry, task_contract_hash=task_contract_hash),
        parent.language_version,
        {
            "kind": "deterministic_typed_completion",
            "hole_id": hole.hole_id,
            "sink": sink.to_dict(),
            "minimum_steps": result.minimum_steps,
            "search_cost": result.search_cost,
        },
        tuple(dict.fromkeys(scope)),
        tuple(edits),
        tuple(preconditions),
        tuple(postconditions),
        {
            "completion_distance": result.to_dict(),
            "final_reference": final_reference,
            "test_evaluated": False,
        },
    )
