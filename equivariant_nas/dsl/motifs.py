"""Versioned, fully expandable motif templates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .diagnostics import DSLValidationError, Diagnostic


def _substitute(value: Any, attrs: Mapping[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$attr:"):
        name = value.split(":", 1)[1]
        if name not in attrs:
            raise DSLValidationError([Diagnostic("E_MOTIF_001", "missing motif attribute", actual=name)])
        return attrs[name]
    if isinstance(value, list):
        return [_substitute(item, attrs) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute(item, attrs) for item in value)
    if isinstance(value, Mapping):
        return {key: _substitute(item, attrs) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class MotifDefinition:
    name: str
    version: int
    input_ports: Tuple[str, ...]
    output_bindings: Mapping[str, str]
    template_nodes: Tuple[Node, ...]
    required_attrs: Tuple[str, ...] = ()
    group_families: Tuple[str, ...] = ("O3", "SO3")
    provenance: Mapping[str, Any] = field(default_factory=dict)
    certification: str = "constructive"
    semantic_constraints: Tuple[str, ...] = ()
    edit_guidance: Tuple[str, ...] = ()

    @property
    def qualified_name(self) -> str:
        return "{}@{}".format(self.name, self.version)

    def __post_init__(self) -> None:
        local_ids = {node.id for node in self.template_nodes}
        if len(local_ids) != len(self.template_nodes):
            raise DSLValidationError([Diagnostic("E_MOTIF_002", "motif template node ids must be unique")])
        for port, reference in self.output_bindings.items():
            root = reference.split(":", 1)[0]
            if root not in local_ids:
                raise DSLValidationError([Diagnostic("E_MOTIF_003", "motif output references an unknown local node", port=port, actual=reference)])

    def content_hash(self) -> str:
        payload = {
            "name": self.name,
            "version": self.version,
            "inputs": self.input_ports,
            "outputs": dict(self.output_bindings),
            "nodes": [item.to_dict() for item in self.template_nodes],
            "required_attrs": self.required_attrs,
            "groups": self.group_families,
            "certification": self.certification,
            "semantic_constraints": self.semantic_constraints,
            "edit_guidance": self.edit_guidance,
        }
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MotifRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, MotifDefinition] = {}

    def register(self, motif: MotifDefinition) -> None:
        if motif.qualified_name in self._items:
            raise DSLValidationError([Diagnostic("E_MOTIF_004", "duplicate motif", actual=motif.qualified_name)])
        self._items[motif.qualified_name] = motif

    def resolve(self, name: str) -> MotifDefinition:
        key = name if "@" in name else "{}@1".format(name)
        try:
            return self._items[key]
        except KeyError:
            raise DSLValidationError([Diagnostic("E_MOTIF_005", "unknown motif", actual=name)])

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._items))


def expand_motifs(program: ArchitectureProgram, registry: MotifRegistry) -> ArchitectureProgram:
    """Expand every `motif.*` node and rewrite references transactionally."""

    expanded = []
    aliases: Dict[str, str] = {}
    for call in program.nodes:
        if not call.op.startswith("motif."):
            expanded.append(call)
            continue
        motif = registry.resolve(call.op)
        missing_inputs = set(motif.input_ports) - set(call.inputs)
        extra_inputs = set(call.inputs) - set(motif.input_ports)
        missing_attrs = set(motif.required_attrs) - set(call.attrs)
        if missing_inputs or extra_inputs or missing_attrs:
            raise DSLValidationError([
                Diagnostic(
                    "E_MOTIF_006",
                    "motif call does not satisfy its signature",
                    node_id=call.id,
                    details={"missing_inputs": sorted(missing_inputs), "extra_inputs": sorted(extra_inputs), "missing_attrs": sorted(missing_attrs)},
                )
            ])
        if set(call.outputs) != set(motif.output_bindings):
            raise DSLValidationError([
                Diagnostic("E_MOTIF_007", "motif call outputs do not match motif signature", node_id=call.id)
            ])
        local_map = {node.id: "{}__{}".format(call.id, node.id) for node in motif.template_nodes}

        def map_reference(reference: str) -> Tuple[str, ...]:
            if reference.startswith("$input:"):
                input_name = reference.split(":", 1)[1]
                return tuple(call.inputs[input_name])
            root, separator, port = reference.partition(":")
            if root not in local_map:
                raise DSLValidationError([Diagnostic("E_MOTIF_008", "motif template contains unknown reference", node_id=call.id, actual=reference)])
            mapped = local_map[root]
            return ("{}:{}".format(mapped, port),) if separator else (mapped,)

        for template in motif.template_nodes:
            inputs = {}
            for port, references in template.inputs.items():
                mapped = []
                for reference in references:
                    mapped.extend(map_reference(reference))
                inputs[port] = tuple(mapped)
            expanded.append(
                replace(
                    template,
                    id=local_map[template.id],
                    inputs=inputs,
                    attrs=_substitute(template.attrs, call.attrs),
                    annotations={**dict(template.annotations), "expanded_from": call.id, "motif": motif.qualified_name},
                )
            )
        for output, reference in motif.output_bindings.items():
            mapped = map_reference(reference)
            if len(mapped) != 1:
                raise DSLValidationError([Diagnostic("E_MOTIF_009", "motif output must bind one value", node_id=call.id, port=output)])
            aliases["{}:{}".format(call.id, output)] = mapped[0]
            if len(call.outputs) == 1:
                aliases[call.id] = mapped[0]

    def rewrite(reference: str) -> str:
        seen = set()
        while reference in aliases:
            if reference in seen:
                raise DSLValidationError([Diagnostic("E_MOTIF_010", "motif alias cycle", actual=reference)])
            seen.add(reference)
            reference = aliases[reference]
        return reference

    rewritten_nodes = tuple(
        replace(node, inputs={port: tuple(rewrite(ref) for ref in refs) for port, refs in node.inputs.items()})
        for node in expanded
    )
    rewritten_outputs = tuple(replace(output, source=rewrite(output.source)) for output in program.outputs)
    return replace(program, nodes=rewritten_nodes, outputs=rewritten_outputs)
