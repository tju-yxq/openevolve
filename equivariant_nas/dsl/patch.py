"""Transactional typed patches for LLM-authored architecture edits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import TypeChecker
from .registry import PrimitiveRegistry


_EDIT_KINDS = {
    "replace_node",
    "change_attrs",
    "rewire_port",
    "rewire_output",
    "insert_before",
    "insert_after",
    "delete_if_bypassed",
    "instantiate_motif",
    "change_parameters",
}

_CONDITION_FIELDS = {
    "node_exists": {"kind", "node_id"},
    "node_absent": {"kind", "node_id"},
    "node_op_is": {"kind", "node_id", "op"},
    "node_attr_equals": {"kind", "node_id", "attr", "value"},
    "node_input_equals": {"kind", "node_id", "port", "references"},
    "output_source_is": {"kind", "output", "reference"},
}


def _validate_condition(condition: Mapping[str, Any]) -> None:
    kind = str(condition.get("kind", ""))
    expected = _CONDITION_FIELDS.get(kind)
    if expected is None or set(condition) != expected:
        raise DSLValidationError([
            Diagnostic(
                "E_PATCH_011",
                "patch condition must use one executable condition contract",
                actual=kind,
                details={
                    "fields": sorted(condition),
                    "allowed_contracts": {name: sorted(fields) for name, fields in sorted(_CONDITION_FIELDS.items())},
                },
            )
        ])


def _assert_conditions(program: ArchitectureProgram, conditions: Sequence[Mapping[str, Any]], phase: str) -> None:
    nodes = {node.id: node for node in program.nodes}
    outputs = {output.name: output for output in program.outputs}
    for condition in conditions:
        _validate_condition(condition)
        kind = str(condition["kind"])
        passed = False
        if kind == "node_exists":
            passed = str(condition["node_id"]) in nodes
        elif kind == "node_absent":
            passed = str(condition["node_id"]) not in nodes
        elif kind == "node_op_is":
            node = nodes.get(str(condition["node_id"]))
            expected_op = str(condition["op"])
            expected_op = expected_op if "@" in expected_op else "{}@1".format(expected_op)
            actual_op = node.op if node is not None else ""
            actual_op = actual_op if "@" in actual_op else "{}@1".format(actual_op) if actual_op else ""
            passed = actual_op == expected_op
        elif kind == "node_attr_equals":
            node = nodes.get(str(condition["node_id"]))
            passed = node is not None and node.attrs.get(str(condition["attr"])) == condition["value"]
        elif kind == "node_input_equals":
            node = nodes.get(str(condition["node_id"]))
            references = condition["references"]
            references = (references,) if isinstance(references, str) else tuple(str(item) for item in references)
            passed = node is not None and node.inputs.get(str(condition["port"])) == tuple(references)
        elif kind == "output_source_is":
            output = outputs.get(str(condition["output"]))
            passed = output is not None and output.source == str(condition["reference"])
        if not passed:
            raise DSLValidationError([
                Diagnostic(
                    "E_PATCH_012",
                    "{} condition failed".format(phase),
                    actual=json.dumps(dict(condition), ensure_ascii=False, sort_keys=True),
                )
            ])


def patch_protocol_schema(
    parent_architecture_id: str,
    language_version: str,
    immutable_scope: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return the authoritative, LLM-facing typed-patch wire contract.

    This is intentionally more explicit than the persisted JSON Schema: language
    models need the payload shape of every edit, not a single representative edit.
    """

    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "patch_version",
            "parent_architecture_id",
            "language_version",
            "hypothesis",
            "scope",
            "edits",
            "preconditions",
            "postconditions",
            "expected_effects",
        ],
        "properties": {
            "patch_version": {"const": "1.0"},
            "parent_architecture_id": {"const": parent_architecture_id},
            "language_version": {"const": language_version},
            "hypothesis": {"type": "object"},
            "scope": (
                {"const": list(immutable_scope), "description": "immutable planner-authorized edit scope"}
                if immutable_scope
                else {"type": "array", "items": {"type": "string"}, "minItems": 1}
            ),
            "edits": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "target", "payload"],
                    "properties": {
                        "kind": {"enum": sorted(_EDIT_KINDS)},
                        "target": {
                            "type": "string",
                            "description": "exact source node id, output:<name>, or an authorized constructor.* parameter path",
                        },
                        "payload": {"type": "object"},
                    },
                },
            },
            "preconditions": {"type": "array", "items": {"type": "object"}},
            "postconditions": {"type": "array", "items": {"type": "object"}},
            "expected_effects": {"type": "object"},
        },
        "edit_payload_contracts": {
            "replace_node": {"node": "complete Node object; id must equal target"},
            "change_attrs": {"attrs": "complete replacement attrs object"},
            "rewire_port": {"port": "existing input port name", "references": "source reference string or list"},
            "rewire_output": {"reference": "new source reference for the named program output"},
            "insert_before": {"node": "complete Node object with a new unique id"},
            "insert_after": {"node": "complete Node object with a new unique id"},
            "delete_if_bypassed": {"replacement_reference": "existing source reference"},
            "instantiate_motif": {"op": "visible motif name", "attrs": "motif attrs object"},
            "change_parameters": {"updates": "mapping from authorized flat parameter path to JSON value"},
        },
        "node_contract": {
            "required": ["id", "op", "inputs"],
            "optional": ["attrs", "outputs", "declared_types", "annotations"],
            "input_reference_format": "node_id or node_id:output_port; external inputs use input:port_name",
        },
        "formal_v1_constructor_contract": {
            "edit_kind": "change_parameters",
            "target": "one exact constructor.* path present in immutable scope",
            "payload": {"value": "one JSON value admitted by the capability profile"},
        },
        "condition_contracts": {
            name: sorted(fields) for name, fields in sorted(_CONDITION_FIELDS.items())
        },
    }


@dataclass(frozen=True)
class PatchEdit:
    kind: str
    target: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _EDIT_KINDS:
            raise DSLValidationError([Diagnostic("E_PATCH_001", "unsupported patch edit", actual=self.kind)])

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PatchEdit":
        required = {"kind", "target", "payload"}
        unknown = set(data) - required
        missing = required - set(data)
        if unknown or missing:
            raise DSLValidationError([
                Diagnostic(
                    "E_PATCH_002",
                    "patch edit must contain exactly kind, target, and payload",
                    details={"unknown_fields": sorted(unknown), "missing_fields": sorted(missing)},
                )
            ])
        if not isinstance(data["payload"], Mapping):
            raise DSLValidationError([Diagnostic("E_PATCH_002", "patch edit payload must be an object")])
        return cls(str(data["kind"]), str(data["target"]), dict(data.get("payload", {})))

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "target": self.target, "payload": dict(self.payload)}


@dataclass(frozen=True)
class TypedPatch:
    patch_version: str
    parent_architecture_id: str
    language_version: str
    hypothesis: Mapping[str, Any]
    scope: Tuple[str, ...]
    edits: Tuple[PatchEdit, ...]
    preconditions: Tuple[Mapping[str, Any], ...] = ()
    postconditions: Tuple[Mapping[str, Any], ...] = ()
    expected_effects: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TypedPatch":
        allowed = {"patch_version", "parent_architecture_id", "language_version", "hypothesis", "scope", "edits", "preconditions", "postconditions", "expected_effects"}
        required = {"patch_version", "parent_architecture_id", "language_version", "hypothesis", "scope", "edits"}
        unknown = set(data) - allowed
        missing = required - set(data)
        if unknown or missing:
            raise DSLValidationError([
                Diagnostic(
                    "E_PATCH_003",
                    "patch does not match the typed-patch envelope",
                    details={"unknown_fields": sorted(unknown), "missing_fields": sorted(missing)},
                )
            ])
        if not isinstance(data["hypothesis"], Mapping) or not isinstance(data["edits"], Sequence) or isinstance(data["edits"], (str, bytes)):
            raise DSLValidationError([Diagnostic("E_PATCH_003", "patch hypothesis must be an object and edits must be an array")])
        for condition in tuple(data.get("preconditions", ())) + tuple(data.get("postconditions", ())):
            if not isinstance(condition, Mapping):
                raise DSLValidationError([Diagnostic("E_PATCH_011", "patch conditions must be objects")])
            _validate_condition(condition)
        return cls(
            patch_version=str(data["patch_version"]),
            parent_architecture_id=str(data["parent_architecture_id"]),
            language_version=str(data["language_version"]),
            hypothesis=dict(data["hypothesis"]),
            scope=tuple(str(item) for item in data["scope"]),
            edits=tuple(PatchEdit.from_dict(item) for item in data["edits"]),
            preconditions=tuple(dict(item) for item in data.get("preconditions", [])),
            postconditions=tuple(dict(item) for item in data.get("postconditions", [])),
            expected_effects=dict(data.get("expected_effects", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_version": self.patch_version,
            "parent_architecture_id": self.parent_architecture_id,
            "language_version": self.language_version,
            "hypothesis": dict(self.hypothesis),
            "scope": list(self.scope),
            "edits": [item.to_dict() for item in self.edits],
            "preconditions": [dict(item) for item in self.preconditions],
            "postconditions": [dict(item) for item in self.postconditions],
            "expected_effects": dict(self.expected_effects),
        }


def apply_typed_patch(
    parent: ArchitectureProgram,
    patch: TypedPatch,
    registry: PrimitiveRegistry,
    *,
    task_contract_hash: str = "unresolved-task-contract",
    validate_parent_id: bool = True,
    expected_parent_id: str = "",
    validate_child_with_core_registry: bool = True,
) -> ArchitectureProgram:
    """Apply all edits to a copy and commit only if the child type-checks."""

    if patch.language_version != parent.language_version:
        raise DSLValidationError([Diagnostic("E_PATCH_004", "patch and parent language versions differ")])
    expected_parent = expected_parent_id or architecture_id(parent, registry, task_contract_hash=task_contract_hash)
    if validate_parent_id and patch.parent_architecture_id != expected_parent:
        raise DSLValidationError([
            Diagnostic("E_PATCH_005", "patch targets the wrong parent architecture", expected=expected_parent, actual=patch.parent_architecture_id)
        ])
    _assert_conditions(parent, patch.preconditions, "precondition")
    nodes = list(parent.nodes)
    outputs = list(parent.outputs)
    parameters = dict(parent.parameters)
    inserted_node_ids = set()

    def scope_allows(target: str) -> bool:
        if target.startswith("output:"):
            return target in patch.scope
        if target in inserted_node_ids:
            return True
        return any(target == item or target.startswith(item + "__") or target.startswith(item + ".") for item in patch.scope)

    for edit in patch.edits:
        if edit.kind == "change_parameters":
            if not edit.target.startswith("constructor."):
                raise DSLValidationError([
                    Diagnostic("E_PATCH_016", "change_parameters target must be a constructor parameter path", actual=edit.target)
                ])
            if not scope_allows(edit.target):
                raise DSLValidationError([Diagnostic("E_PATCH_007", "patch edit escapes its declared scope", node_id=edit.target)])
            if set(edit.payload) != {"value"}:
                raise DSLValidationError([
                    Diagnostic("E_PATCH_017", "change_parameters payload must contain exactly value", actual=sorted(edit.payload))
                ])
            parameters[edit.target] = edit.payload["value"]
            continue
        if edit.kind == "rewire_output":
            if not edit.target.startswith("output:") or edit.target.count(":") != 1:
                raise DSLValidationError([
                    Diagnostic("E_PATCH_013", "rewire_output target must be output:<name>", actual=edit.target)
                ])
            if not scope_allows(edit.target):
                raise DSLValidationError([
                    Diagnostic("E_PATCH_007", "patch edit escapes its declared scope", node_id=edit.target)
                ])
            if set(edit.payload) != {"reference"}:
                raise DSLValidationError([
                    Diagnostic(
                        "E_PATCH_014",
                        "rewire_output payload must contain exactly reference",
                        actual=sorted(edit.payload),
                    )
                ])
            output_name = edit.target.split(":", 1)[1]
            output_index = next((i for i, output in enumerate(outputs) if output.name == output_name), None)
            if output_index is None:
                raise DSLValidationError([
                    Diagnostic("E_PATCH_015", "rewire_output names an unknown program output", actual=output_name)
                ])
            outputs[output_index] = replace(outputs[output_index], source=str(edit.payload["reference"]))
            continue

        index = next((i for i, node in enumerate(nodes) if node.id == edit.target), None)
        if index is None:
            raise DSLValidationError([Diagnostic("E_PATCH_006", "patch target does not exist", actual=edit.target)])
        if not scope_allows(edit.target):
            raise DSLValidationError([Diagnostic("E_PATCH_007", "patch edit escapes its declared scope", node_id=edit.target)])
        target = nodes[index]
        if edit.kind == "replace_node":
            replacement = Node.from_dict(edit.payload["node"])
            if replacement.id != target.id:
                raise DSLValidationError([Diagnostic("E_PATCH_008", "replacement must preserve target id", node_id=target.id)])
            nodes[index] = replacement
        elif edit.kind == "change_attrs":
            nodes[index] = replace(target, attrs=dict(edit.payload["attrs"]))
        elif edit.kind == "rewire_port":
            port = str(edit.payload["port"])
            if port not in target.inputs:
                raise DSLValidationError([Diagnostic("E_PATCH_009", "rewire names an unknown input port", node_id=target.id, port=port)])
            refs = edit.payload["references"]
            refs = (refs,) if isinstance(refs, str) else tuple(str(item) for item in refs)
            inputs = dict(target.inputs)
            inputs[port] = refs
            nodes[index] = replace(target, inputs=inputs)
        elif edit.kind == "instantiate_motif":
            nodes[index] = replace(target, op=str(edit.payload["op"]), attrs=dict(edit.payload.get("attrs", {})))
        elif edit.kind in ("insert_before", "insert_after"):
            inserted = Node.from_dict(edit.payload["node"])
            if any(item.id == inserted.id for item in nodes):
                raise DSLValidationError([Diagnostic("E_PATCH_010", "inserted node id already exists", actual=inserted.id)])
            offset = 0 if edit.kind == "insert_before" else 1
            nodes.insert(index + offset, inserted)
            inserted_node_ids.add(inserted.id)
        elif edit.kind == "delete_if_bypassed":
            replacement_ref = str(edit.payload["replacement_reference"])
            references = []
            for node in nodes:
                for refs in node.inputs.values():
                    references.extend(refs)
            output_refs = [item.source for item in outputs]
            target_refs = {target.id, "{}:out".format(target.id)}
            if not any(ref in target_refs for ref in references + output_refs):
                nodes.pop(index)
                continue
            rewritten = []
            for node in nodes:
                inputs = {
                    port: tuple(replacement_ref if ref in target_refs else ref for ref in refs)
                    for port, refs in node.inputs.items()
                }
                rewritten.append(replace(node, inputs=inputs))
            nodes = [node for node in rewritten if node.id != target.id]
            outputs = [replace(item, source=replacement_ref if item.source in target_refs else item.source) for item in outputs]
    child = replace(parent, nodes=tuple(nodes), outputs=tuple(outputs), parameters=parameters)
    _assert_conditions(child, patch.postconditions, "postcondition")
    if validate_child_with_core_registry:
        TypeChecker(registry).check(child)
    return child
