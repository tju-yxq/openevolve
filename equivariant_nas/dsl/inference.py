"""Static graph checking, type inference, and proof-obligation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .diagnostics import DSLValidationError, Diagnostic
from .obligations import ObligationKind, ProofObligation
from .parameters import ParameterContract
from .registry import PrimitiveRegistry
from .types import ValueType, value_type_group_families


@dataclass(frozen=True)
class InferenceResult:
    value_types: Mapping[str, ValueType]
    node_order: Tuple[str, ...]
    obligations: Tuple[ProofObligation, ...]
    parameter_contracts: Mapping[str, Tuple[ParameterContract, ...]]

    @property
    def open_obligations(self) -> Tuple[ProofObligation, ...]:
        return tuple(item for item in self.obligations if item.status == "open")


class TypeChecker:
    def __init__(self, registry: PrimitiveRegistry, *, require_closed_obligations: bool = True):
        self.registry = registry
        self.require_closed_obligations = require_closed_obligations

    def check(self, program: ArchitectureProgram) -> InferenceResult:
        node_by_id = {node.id: node for node in program.nodes}
        order = self._topological_order(program, node_by_id)
        live = self._reachable_from_outputs(program, node_by_id)
        dead = sorted(set(node_by_id) - live)
        if dead:
            raise DSLValidationError([
                Diagnostic(
                    "E_GRAPH_002",
                    "architecture contains nodes that do not contribute to any program output",
                    details={"dead_nodes": dead},
                    repairs=("rewire a live consumer or program output to the intended node", "remove the dead subgraph"),
                )
            ])
        values: Dict[str, ValueType] = {
            "input:{}".format(item.name): item.value_type for item in program.inputs
        }
        obligations: List[ProofObligation] = []
        parameter_contracts: Dict[str, Tuple[ParameterContract, ...]] = {}

        for node_id in order:
            node = node_by_id[node_id]
            definition = self.registry.resolve(node.op)
            attrs = definition.canonical_attrs(node.id, node.attrs)
            resolved = {
                port: tuple(self._resolve_reference(ref, values, node) for ref in refs)
                for port, refs in node.inputs.items()
            }
            missing_ports = set(definition.input_ports) - set(resolved)
            extra_ports = set(resolved) - set(definition.input_ports)
            if missing_ports or extra_ports:
                raise DSLValidationError([
                    Diagnostic(
                        "E_PORT_003",
                        "primitive input ports do not match its signature",
                        node_id=node.id,
                        details={"missing": sorted(missing_ports), "extra": sorted(extra_ports)},
                    )
                ])
            input_families = {
                family
                for port_values in resolved.values()
                for value in port_values
                for family in value_type_group_families(value)
            }
            unsupported = input_families - set(definition.group_families)
            if unsupported:
                raise DSLValidationError([
                    Diagnostic("E_GROUP_008", "primitive does not support input group", node_id=node.id, details={"groups": sorted(unsupported)})
                ])
            inferred, created = definition.type_rule(node.id, resolved, attrs)
            if set(inferred) != set(definition.output_ports):
                raise DSLValidationError([
                    Diagnostic("E_REGISTRY_003", "primitive type rule returned incorrect output ports", node_id=node.id)
                ])
            if set(node.outputs) != set(definition.output_ports):
                raise DSLValidationError([
                    Diagnostic(
                        "E_PORT_004",
                        "node output declaration does not match primitive signature",
                        node_id=node.id,
                        expected=str(definition.output_ports),
                        actual=str(node.outputs),
                    )
                ])
            for output_name, output_type in inferred.items():
                declared = node.declared_types.get(output_name)
                if declared is not None and declared != output_type:
                    raise DSLValidationError([
                        Diagnostic(
                            "E_TYPE_005",
                            "declared output type disagrees with inferred type",
                            node_id=node.id,
                            port=output_name,
                            expected=str(declared.to_dict()),
                            actual=str(output_type.to_dict()),
                        )
                    ])
                values["{}:{}".format(node.id, output_name)] = output_type
                if len(node.outputs) == 1:
                    values[node.id] = output_type
            parameter_contracts[node.id] = definition.infer_parameter_contracts(
                node.id,
                resolved,
                inferred,
                attrs,
            )
            obligations.extend(created)

        output_obligations = []
        for output in program.outputs:
            actual = self._resolve_reference(output.source, values, None)
            if actual != output.expected_type:
                raise DSLValidationError([
                    Diagnostic(
                        "E_OUTPUT_001",
                        "program output does not satisfy the task contract type",
                        port=output.name,
                        expected=str(output.expected_type.to_dict()),
                        actual=str(actual.to_dict()),
                    )
                ])
            output_obligations.append(
                ProofObligation(
                    "output:{}".format(output.name),
                    ObligationKind.OUTPUT_CONTRACT_MATCH,
                    output.source,
                    "discharged",
                    "type-checker",
                )
            )
        obligations.extend(output_obligations)
        obligations = list(self._reconcile_frame_obligations(obligations))
        open_items = [item for item in obligations if item.status == "open"]
        if self.require_closed_obligations and open_items:
            raise DSLValidationError([
                Diagnostic(
                    "E_OBLIGATION_001",
                    "program has unresolved proof obligations",
                    details={"obligations": [item.to_dict() for item in open_items]},
                )
            ])
        return InferenceResult(values, tuple(order), tuple(obligations), parameter_contracts)

    @staticmethod
    def _resolve_reference(reference: str, values: Mapping[str, ValueType], node: Optional[Node]) -> ValueType:
        if reference in values:
            return values[reference]
        raise DSLValidationError([
            Diagnostic(
                "E_REF_001",
                "unknown or forward value reference",
                node_id=node.id if node else "",
                actual=reference,
            )
        ])

    @staticmethod
    def _topological_order(program: ArchitectureProgram, nodes: Mapping[str, Node]) -> List[str]:
        input_names = {"input:{}".format(item.name) for item in program.inputs}
        dependencies: Dict[str, set] = {node.id: set() for node in program.nodes}
        for node in program.nodes:
            for refs in node.inputs.values():
                for reference in refs:
                    root = reference.split(":", 1)[0]
                    if reference in input_names or root == "input":
                        if reference not in input_names:
                            raise DSLValidationError([Diagnostic("E_REF_002", "unknown input reference", node_id=node.id, actual=reference)])
                        continue
                    if root not in nodes:
                        raise DSLValidationError([Diagnostic("E_REF_003", "reference names an unknown node", node_id=node.id, actual=reference)])
                    dependencies[node.id].add(root)
        order: List[str] = []
        remaining = {key: set(value) for key, value in dependencies.items()}
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if not deps)
            if not ready:
                raise DSLValidationError([Diagnostic("E_GRAPH_001", "architecture graph contains a cycle", details={"remaining": sorted(remaining)})])
            order.extend(ready)
            for key in ready:
                del remaining[key]
            for deps in remaining.values():
                deps.difference_update(ready)
        return order

    @staticmethod
    def _reachable_from_outputs(program: ArchitectureProgram, nodes: Mapping[str, Node]) -> set:
        roots = []
        for output in program.outputs:
            root = output.source.split(":", 1)[0]
            if root in nodes:
                roots.append(root)
        reachable = set()
        pending = list(roots)
        while pending:
            node_id = pending.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            for refs in nodes[node_id].inputs.values():
                for reference in refs:
                    parent = reference.split(":", 1)[0]
                    if parent in nodes and parent not in reachable:
                        pending.append(parent)
        return reachable

    @staticmethod
    def _reconcile_frame_obligations(obligations: Sequence[ProofObligation]) -> Tuple[ProofObligation, ...]:
        open_frame_ids = [
            item.details.get("frame_id")
            for item in obligations
            if item.kind == ObligationKind.FRAME_BALANCE and item.status == "open"
        ]
        duplicates = sorted({item for item in open_frame_ids if open_frame_ids.count(item) > 1})
        if duplicates:
            raise DSLValidationError([
                Diagnostic(
                    "E_FRAME_007",
                    "frame tokens must uniquely identify one frame-entry value",
                    details={"duplicate_frame_ids": duplicates},
                )
            ])
        restored = {
            item.details.get("frame_id")
            for item in obligations
            if item.kind == ObligationKind.FRAME_BALANCE and item.status == "discharged"
        }
        output = []
        for item in obligations:
            if item.kind == ObligationKind.FRAME_BALANCE and item.status == "open" and item.details.get("frame_id") in restored:
                output.append(item.discharge("matching frame restoration"))
            else:
                output.append(item)
        return tuple(output)
