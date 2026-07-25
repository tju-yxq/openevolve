"""Deterministic typed-completion advice derived from compiler diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .completion import AvailableValue, HoleSink, TypedHole, complete_typed_hole
from .diagnostics import DSLValidationError, Diagnostic
from .inference import TypeChecker
from .registry import PrimitiveRegistry
from .types import Carrier, EquivariantType, Frame


_LOCAL_COMPLETION_CODES = {
    "E_TYPE_004",
    "E_CARRIER_001",
    "E_CARRIER_002",
    "E_CARRIER_003",
    "E_FRAME_004",
    "E_FRAME_005",
    "E_FRAME_006",
}


def _resolve(reference: str, values: Mapping[str, EquivariantType]) -> EquivariantType:
    if reference in values:
        return values[reference]
    alias = "{}:out".format(reference)
    if ":" not in reference and alias in values:
        return values[alias]
    raise KeyError(reference)


def _recover_prefix_types(program: ArchitectureProgram, registry: PrimitiveRegistry) -> Dict[str, EquivariantType]:
    """Infer every value whose dependencies and trusted type rule remain valid."""

    nodes = {node.id: node for node in program.nodes}
    try:
        order = TypeChecker._topological_order(program, nodes)
    except DSLValidationError:
        return {"input:{}".format(item.name): item.value_type for item in program.inputs}
    values: Dict[str, EquivariantType] = {
        "input:{}".format(item.name): item.value_type for item in program.inputs
    }
    for node_id in order:
        node = nodes[node_id]
        try:
            definition = registry.resolve(node.op)
            resolved = {
                port: tuple(_resolve(reference, values) for reference in references)
                for port, references in node.inputs.items()
            }
            inferred, _ = definition.type_rule(node.id, resolved, node.attrs)
        except (DSLValidationError, KeyError, TypeError, ValueError):
            continue
        for output_name, output_type in inferred.items():
            values["{}:{}".format(node.id, output_name)] = output_type
            if len(inferred) == 1:
                values[node.id] = output_type
    return values


def _context_like(base: EquivariantType, irreps_source: EquivariantType) -> EquivariantType:
    return EquivariantType(
        base.group,
        base.carrier,
        irreps_source.irreps,
        base.frame,
        base.axes,
        base.dtype,
        base.measure,
        base.level,
    )


def _single_reference(node: Node, port: str) -> str:
    references = node.inputs.get(port, ())
    if len(references) != 1:
        raise KeyError(port)
    return references[0]


def _local_request(
    diagnostic: Diagnostic,
    node: Node,
    values: Mapping[str, EquivariantType],
    registry: PrimitiveRegistry,
) -> Tuple[str, EquivariantType, HoleSink]:
    definition = registry.resolve(node.op)
    port = diagnostic.port if diagnostic.port in node.inputs else ""
    if diagnostic.code == "E_TYPE_004" and len(definition.input_ports) >= 2:
        port = definition.input_ports[1]
        base_port = definition.input_ports[0]
        base = _resolve(_single_reference(node, base_port), values)
        actual = _resolve(_single_reference(node, port), values)
        exact = definition.name == "core.residual_add"
        expected = base if exact else _context_like(base, actual)
        return _single_reference(node, port), expected, HoleSink.input_port(node.id, port)
    if not port and len(definition.input_ports) == 1:
        port = definition.input_ports[0]
    source = _single_reference(node, port)
    actual = _resolve(source, values)
    if diagnostic.code in {"E_CARRIER_001", "E_CARRIER_003"}:
        expected = actual.with_carrier(Carrier.NODE)
    elif diagnostic.code == "E_CARRIER_002":
        expected = actual.with_carrier(Carrier.EDGE)
    elif diagnostic.code in {"E_FRAME_004", "E_FRAME_005"}:
        expected = actual.with_frame(Frame("global"))
    elif diagnostic.code == "E_FRAME_006":
        expected = actual.with_frame(Frame("edge", str(node.attrs.get("frame_id", "completion_frame"))))
    else:
        raise KeyError(diagnostic.code)
    return source, expected, HoleSink.input_port(node.id, port)


def completion_repair_suggestions(
    program: ArchitectureProgram,
    diagnostics: Sequence[Diagnostic],
    registry: PrimitiveRegistry,
    *,
    allowed_ops: Sequence[str] = (),
    authorized_scope: Sequence[str] = (),
    max_steps: int = 3,
) -> Tuple[Mapping[str, Any], ...]:
    """Return trusted local paths that a Repairer may encode in its replacement patch."""

    values = _recover_prefix_types(program, registry)
    nodes = {node.id: node for node in program.nodes}
    outputs = {output.name: output for output in program.outputs}
    authorized = set(authorized_scope)
    suggestions = []
    for index, diagnostic in enumerate(diagnostics):
        try:
            if diagnostic.code == "E_OUTPUT_001":
                output = outputs[diagnostic.port]
                source = output.source
                actual = _resolve(source, values)
                expected = output.expected_type
                sink = HoleSink.program_output(output.name)
                required_scope = "output:{}".format(output.name)
            elif diagnostic.code in _LOCAL_COMPLETION_CODES:
                node = nodes[diagnostic.node_id]
                source, expected, sink = _local_request(diagnostic, node, values, registry)
                actual = _resolve(source, values)
                required_scope = node.id
            else:
                continue
        except (KeyError, DSLValidationError, TypeError, ValueError):
            continue
        hole = TypedHole(
            "repair_{}_{}_{}".format(index, diagnostic.code.lower(), sink.target).replace(".", "_"),
            expected,
            (AvailableValue(source, actual),),
            tuple(allowed_ops),
            max_steps,
        )
        result = complete_typed_hole(hole, registry)
        suggestions.append({
            "diagnostic_code": diagnostic.code,
            "source_reference": source,
            "actual_type": actual.to_dict(),
            "expected_type": expected.to_dict(),
            "sink": sink.to_dict(),
            "required_scope": required_scope,
            "scope_compatible": required_scope in authorized,
            "hole": hole.to_dict(),
            "completion": result.to_dict(),
            "usage": (
                "may encode these trusted actions in the replacement patch"
                if result.reachable and required_scope in authorized
                else "advisory only; do not widen scope or invent an unverified path"
            ),
        })
    return tuple(suggestions)
