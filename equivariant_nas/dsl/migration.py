"""Deterministic migrations between the legacy v1 and versioned v2 type schemas."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple

from .ast import ArchitectureProgram, InputPort
from .canonicalize import COMPILER_SEMANTICS_VERSION, architecture_id
from .motifs import MotifRegistry, expand_motifs
from .registry import PrimitiveRegistry
from .task import TaskContract
from .types import (
    VALUE_TYPE_SCHEMA_VERSION,
    AxisSpec,
    EquivariantTensorType,
    EquivariantType,
    FeatureRole,
    InvariantTensorType,
    IndexMapType,
    Carrier,
    RecordType,
    RepresentationLayout,
    TupleType,
    ValueType,
)


MIGRATION_POLICY_VERSION = "evoequilang-v1-to-v2-types@1"
LEGACY_COMPILER_SEMANTICS_VERSION = "evoequilang-4"


def _axis_role(name: str) -> FeatureRole:
    try:
        return FeatureRole.parse(name)
    except Exception:
        return FeatureRole.CHANNEL


def migrate_value_type_v1_to_v2(value: ValueType) -> ValueType:
    """Lift every representable legacy type without changing its tensor semantics."""

    if isinstance(value, (EquivariantTensorType, InvariantTensorType)):
        return value
    if isinstance(value, EquivariantType):
        axes = tuple(
            AxisSpec(name=name, role=_axis_role(name), order=index)
            for index, name in enumerate(value.axes)
        )
        return EquivariantTensorType(
            group=value.group,
            carrier=value.carrier,
            irreps=value.irreps,
            frame=value.frame,
            axes=value.axes,
            dtype=value.dtype,
            measure=value.measure,
            level=value.level,
            axis_specs=axes,
            layout=RepresentationLayout(storage="irrep_major"),
        )
    if isinstance(value, RecordType):
        return RecordType({name: migrate_value_type_v1_to_v2(item) for name, item in value.fields})
    if isinstance(value, TupleType):
        return TupleType(tuple(migrate_value_type_v1_to_v2(item) for item in value.items))
    return value


def legacy_equivariant_view(value: ValueType) -> EquivariantType:
    """Return the exact legacy tensor contract when a v2 type has one."""

    if isinstance(value, EquivariantTensorType):
        return EquivariantType(
            value.group,
            value.carrier,
            value.irreps,
            value.frame,
            value.axes,
            value.dtype,
            value.measure,
            value.level,
        )
    if isinstance(value, EquivariantType):
        return value
    raise TypeError("{} has no lossless legacy EquivariantType view".format(type(value).__name__))


@dataclass(frozen=True)
class TypeMigrationManifest:
    policy_version: str
    source_language_version: str
    target_language_version: str
    value_type_schema_version: str
    source_compiler_semantics_version: str
    target_compiler_semantics_version: str
    source_architecture_id: str
    target_architecture_id: str
    migrated_type_paths: Tuple[str, ...]
    semantic_scope: str = "type-schema-only"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "source_language_version": self.source_language_version,
            "target_language_version": self.target_language_version,
            "value_type_schema_version": self.value_type_schema_version,
            "source_compiler_semantics_version": self.source_compiler_semantics_version,
            "target_compiler_semantics_version": self.target_compiler_semantics_version,
            "source_architecture_id": self.source_architecture_id,
            "target_architecture_id": self.target_architecture_id,
            "migrated_type_paths": list(self.migrated_type_paths),
            "semantic_scope": self.semantic_scope,
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TypeMigrationResult:
    program: ArchitectureProgram
    manifest: TypeMigrationManifest


@dataclass(frozen=True)
class ExplicitTopologyMigrationManifest:
    policy_version: str
    type_migration_manifest_hash: str
    source_architecture_id: str
    target_architecture_id: str
    replaced_nodes: Tuple[Tuple[str, str, str], ...]
    added_inputs: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "type_migration_manifest_hash": self.type_migration_manifest_hash,
            "source_architecture_id": self.source_architecture_id,
            "target_architecture_id": self.target_architecture_id,
            "replaced_nodes": [list(item) for item in self.replaced_nodes],
            "added_inputs": list(self.added_inputs),
        }

    def content_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExplicitTopologyMigrationResult:
    program: ArchitectureProgram
    type_migration: TypeMigrationManifest
    manifest: ExplicitTopologyMigrationManifest


def migrate_task_contract_v1_to_v2(task: TaskContract) -> TaskContract:
    metadata = dict(task.metadata)
    metadata["type_migration"] = {
        "policy_version": MIGRATION_POLICY_VERSION,
        "value_type_schema_version": VALUE_TYPE_SCHEMA_VERSION,
        "semantic_scope": "type-schema-only",
    }
    return replace(
        task,
        output_type=migrate_value_type_v1_to_v2(task.output_type),
        metadata=metadata,
    )


def migrate_message_flow_to_explicit_topology(
    program: ArchitectureProgram,
    registry: PrimitiveRegistry,
    motifs: MotifRegistry,
    *,
    target_language_version: str = "2.1.0",
) -> ExplicitTopologyMigrationResult:
    """Replace legacy hidden-context gather/reduce operations with typed inputs."""

    typed = migrate_program_v1_to_v2(program, registry, motifs=motifs)
    expanded = expand_motifs(typed.program, motifs)
    tensor_inputs = [item.value_type for item in expanded.inputs if isinstance(item.value_type, EquivariantTensorType)]
    if not tensor_inputs:
        raise ValueError("explicit-topology migration requires at least one equivariant tensor input")
    group = tensor_inputs[0].group
    if any(item.group != group for item in tensor_inputs):
        raise ValueError("explicit-topology migration does not support mixed groups")

    qualified_ops = {
        node.id: node.op if "@" in node.op else "{}@1".format(node.op)
        for node in expanded.nodes
    }
    endpoint_roles = {
        str(node.attrs.get("endpoint", "source"))
        for node in expanded.nodes
        if qualified_ops[node.id] == "core.edge_lift@1"
    }
    needs_edge_reduce = any(
        qualified_ops[node.id] in ("core.segment_sum@1", "core.segment_mean@1")
        for node in expanded.nodes
    )
    needs_batch_reduce = any(qualified_ops[node.id] == "core.global_pool@1" for node in expanded.nodes)

    existing_names = {item.name for item in expanded.inputs}
    required_inputs = []

    def add_input(name: str, value_type: IndexMapType) -> None:
        if name in existing_names:
            raise ValueError("explicit-topology migration input {} collides with an existing input".format(name))
        existing_names.add(name)
        required_inputs.append(InputPort(name, value_type))

    if "source" in endpoint_roles:
        add_input("source_index", IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source"))
    if "target" in endpoint_roles:
        add_input("target_index", IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target"))
    if needs_edge_reduce:
        add_input("edge_to_node_index", IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment"))
    if needs_batch_reduce:
        add_input("batch_index", IndexMapType(group, Carrier.NODE, Carrier.GRAPH, "batch"))

    replaced_nodes = []
    nodes = []
    for node in expanded.nodes:
        qualified = qualified_ops[node.id]
        if qualified == "core.edge_lift@1":
            endpoint = str(node.attrs.get("endpoint", "source"))
            replacement = "core.endpoint_gather@1"
            nodes.append(replace(
                node,
                op=replacement,
                inputs={"x": tuple(node.inputs["x"]), "index": ("input:{}_index".format(endpoint),)},
                attrs={},
                annotations={**dict(node.annotations), "migrated_from": qualified},
            ))
            replaced_nodes.append((node.id, qualified, replacement))
            continue
        if qualified in ("core.segment_sum@1", "core.segment_mean@1"):
            replacement = "core.segment_reduce@1"
            nodes.append(replace(
                node,
                op=replacement,
                inputs={"x": tuple(node.inputs["x"]), "index": ("input:edge_to_node_index",)},
                attrs={"reduce": "mean" if qualified == "core.segment_mean@1" else "sum", "normalization": "none"},
                annotations={**dict(node.annotations), "migrated_from": qualified},
            ))
            replaced_nodes.append((node.id, qualified, replacement))
            continue
        if qualified == "core.global_pool@1":
            replacement = "core.segment_reduce@1"
            nodes.append(replace(
                node,
                op=replacement,
                inputs={"x": tuple(node.inputs["x"]), "index": ("input:batch_index",)},
                attrs={"reduce": str(node.attrs.get("reduce", "sum")), "normalization": "none"},
                annotations={**dict(node.annotations), "migrated_from": qualified},
            ))
            replaced_nodes.append((node.id, qualified, replacement))
            continue
        nodes.append(node)

    annotations = dict(expanded.annotations)
    annotations["topology_migration"] = {
        "policy_version": "evoequilang-explicit-topology@1",
        "hidden_context_removed": ["edge_src", "edge_dst", "batch", "num_nodes", "num_graphs"],
    }
    migrated_program = replace(
        expanded,
        language_version=str(target_language_version),
        inputs=tuple(expanded.inputs) + tuple(required_inputs),
        nodes=tuple(nodes),
        annotations=annotations,
    )
    target_id = architecture_id(migrated_program, registry)
    manifest = ExplicitTopologyMigrationManifest(
        "evoequilang-explicit-topology@1",
        typed.manifest.content_hash(),
        typed.manifest.target_architecture_id,
        target_id,
        tuple(replaced_nodes),
        tuple(item.name for item in required_inputs),
    )
    return ExplicitTopologyMigrationResult(migrated_program, typed.manifest, manifest)


def migrate_program_v1_to_v2(
    program: ArchitectureProgram,
    registry: Optional[PrimitiveRegistry] = None,
    *,
    motifs: Optional[MotifRegistry] = None,
    target_language_version: str = "2.0.0",
    task_contract_hash: str = "unresolved-task-contract",
    source_task_contract_hash: Optional[str] = None,
    target_task_contract_hash: Optional[str] = None,
) -> TypeMigrationResult:
    migrated_paths = []

    inputs = []
    for item in program.inputs:
        migrated = migrate_value_type_v1_to_v2(item.value_type)
        if migrated is not item.value_type:
            migrated_paths.append("input:{}".format(item.name))
        inputs.append(replace(item, value_type=migrated))

    nodes = []
    for node in program.nodes:
        declared = {}
        for port, value in node.declared_types.items():
            migrated = migrate_value_type_v1_to_v2(value)
            if migrated is not value:
                migrated_paths.append("node:{}:{}".format(node.id, port))
            declared[port] = migrated
        nodes.append(replace(node, declared_types=declared))

    outputs = []
    for item in program.outputs:
        migrated = migrate_value_type_v1_to_v2(item.expected_type)
        if migrated is not item.expected_type:
            migrated_paths.append("output:{}".format(item.name))
        outputs.append(replace(item, expected_type=migrated))

    annotations = dict(program.annotations)
    annotations["type_migration"] = {
        "policy_version": MIGRATION_POLICY_VERSION,
        "source_language_version": program.language_version,
        "value_type_schema_version": VALUE_TYPE_SCHEMA_VERSION,
        "semantic_scope": "type-schema-only",
    }
    migrated_program = replace(
        program,
        language_version=str(target_language_version),
        inputs=tuple(inputs),
        nodes=tuple(nodes),
        outputs=tuple(outputs),
        annotations=annotations,
    )
    source_identity_program = expand_motifs(program, motifs) if motifs is not None else program
    target_identity_program = expand_motifs(migrated_program, motifs) if motifs is not None else migrated_program
    source_id = architecture_id(
        source_identity_program,
        registry,
        task_contract_hash=source_task_contract_hash or task_contract_hash,
        compiler_version=LEGACY_COMPILER_SEMANTICS_VERSION,
        value_type_schema_version=None,
    )
    target_id = architecture_id(
        target_identity_program,
        registry,
        task_contract_hash=target_task_contract_hash or task_contract_hash,
    )
    manifest = TypeMigrationManifest(
        MIGRATION_POLICY_VERSION,
        program.language_version,
        str(target_language_version),
        VALUE_TYPE_SCHEMA_VERSION,
        LEGACY_COMPILER_SEMANTICS_VERSION,
        COMPILER_SEMANTICS_VERSION,
        source_id,
        target_id,
        tuple(sorted(migrated_paths)),
    )
    return TypeMigrationResult(migrated_program, manifest)
