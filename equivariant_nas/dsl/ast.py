"""Backend-neutral abstract syntax tree for EvoEquiLang programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .types import EquivariantType


def _strict_fields(data: Mapping[str, Any], allowed, kind: str) -> None:
    unknown = set(data) - set(allowed)
    if unknown:
        raise DSLValidationError([
            Diagnostic("E_SCHEMA_003", "unknown {} fields".format(kind), details={"fields": sorted(unknown)})
        ])


@dataclass(frozen=True)
class InputPort:
    name: str
    value_type: EquivariantType

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InputPort":
        _strict_fields(data, ("name", "type"), "input")
        return cls(str(data["name"]), EquivariantType.from_dict(data["type"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.value_type.to_dict()}


@dataclass(frozen=True)
class Node:
    id: str
    op: str
    inputs: Mapping[str, Tuple[str, ...]]
    attrs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Tuple[str, ...] = ("out",)
    declared_types: Mapping[str, EquivariantType] = field(default_factory=dict)
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or ":" in self.id:
            raise DSLValidationError([Diagnostic("E_AST_001", "node id must be nonempty and cannot contain ':'", actual=self.id)])
        if not self.op:
            raise DSLValidationError([Diagnostic("E_AST_002", "node operation is required", node_id=self.id)])
        if not self.outputs or len(set(self.outputs)) != len(self.outputs):
            raise DSLValidationError([Diagnostic("E_AST_003", "node outputs must be nonempty and unique", node_id=self.id)])
        if set(self.declared_types) - set(self.outputs):
            raise DSLValidationError([Diagnostic("E_AST_004", "declared type refers to unknown output", node_id=self.id)])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Node":
        _strict_fields(data, ("id", "op", "inputs", "attrs", "outputs", "declared_types", "annotations"), "node")
        raw_inputs = data.get("inputs", {})
        inputs = {}
        for port, refs in raw_inputs.items():
            if isinstance(refs, str):
                inputs[str(port)] = (refs,)
            elif isinstance(refs, (list, tuple)) and all(isinstance(item, str) for item in refs):
                inputs[str(port)] = tuple(refs)
            else:
                raise DSLValidationError([Diagnostic("E_AST_005", "node input must be a reference or reference list", node_id=str(data.get("id", "")), port=str(port))])
        declared = {
            str(name): EquivariantType.from_dict(value)
            for name, value in data.get("declared_types", {}).items()
        }
        return cls(
            id=str(data["id"]),
            op=str(data["op"]),
            inputs=inputs,
            attrs=dict(data.get("attrs", {})),
            outputs=tuple(str(item) for item in data.get("outputs", ["out"])),
            declared_types=declared,
            annotations=dict(data.get("annotations", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": {
                name: refs[0] if len(refs) == 1 else list(refs)
                for name, refs in sorted(self.inputs.items())
            },
            "attrs": dict(self.attrs),
            "outputs": list(self.outputs),
            "declared_types": {
                name: value.to_dict() for name, value in sorted(self.declared_types.items())
            },
            "annotations": dict(self.annotations),
        }


@dataclass(frozen=True)
class OutputPort:
    name: str
    source: str
    expected_type: EquivariantType

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputPort":
        _strict_fields(data, ("name", "source", "type"), "output")
        return cls(str(data["name"]), str(data["source"]), EquivariantType.from_dict(data["type"]))

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "source": self.source, "type": self.expected_type.to_dict()}


@dataclass(frozen=True)
class ArchitectureProgram:
    language_version: str
    task_contract: str
    inputs: Tuple[InputPort, ...]
    nodes: Tuple[Node, ...]
    outputs: Tuple[OutputPort, ...]
    program_id: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    constraints: Tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.language_version:
            raise DSLValidationError([Diagnostic("E_AST_006", "language_version is required")])
        if not self.task_contract:
            raise DSLValidationError([Diagnostic("E_AST_007", "task_contract is required")])
        input_names = [item.name for item in self.inputs]
        node_ids = [item.id for item in self.nodes]
        output_names = [item.name for item in self.outputs]
        if len(input_names) != len(set(input_names)):
            raise DSLValidationError([Diagnostic("E_AST_008", "input names must be unique")])
        if len(node_ids) != len(set(node_ids)):
            raise DSLValidationError([Diagnostic("E_AST_009", "node ids must be unique")])
        if len(output_names) != len(set(output_names)):
            raise DSLValidationError([Diagnostic("E_AST_010", "output names must be unique")])
        if set(input_names) & set(node_ids):
            raise DSLValidationError([Diagnostic("E_AST_011", "input and node names must not collide")])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArchitectureProgram":
        _strict_fields(
            data,
            ("language_version", "program_id", "task_contract", "parameters", "inputs", "nodes", "outputs", "constraints", "annotations"),
            "program",
        )
        return cls(
            language_version=str(data["language_version"]),
            task_contract=str(data["task_contract"]),
            inputs=tuple(InputPort.from_dict(item) for item in data.get("inputs", [])),
            nodes=tuple(Node.from_dict(item) for item in data.get("nodes", [])),
            outputs=tuple(OutputPort.from_dict(item) for item in data.get("outputs", [])),
            program_id=str(data.get("program_id", "")),
            parameters=dict(data.get("parameters", {})),
            constraints=tuple(dict(item) for item in data.get("constraints", [])),
            annotations=dict(data.get("annotations", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language_version": self.language_version,
            "program_id": self.program_id,
            "task_contract": self.task_contract,
            "parameters": dict(self.parameters),
            "inputs": [item.to_dict() for item in self.inputs],
            "nodes": [item.to_dict() for item in self.nodes],
            "outputs": [item.to_dict() for item in self.outputs],
            "constraints": [dict(item) for item in self.constraints],
            "annotations": dict(self.annotations),
        }
