"""Certified local structural rewrites for explicit Equiformer V3 Typed DSL graphs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import TypeChecker
from .patch import PatchEdit, TypedPatch, apply_typed_patch
from .registry import PrimitiveRegistry
from .v3_evolution import v3_program_spec


V3_STRUCTURAL_MUTATION_VERSION = "v3-local-structural-rewrite-2"


@dataclass(frozen=True)
class V3StructuralTarget:
    node_id: str
    current_contract: str

    def to_dict(self) -> Mapping[str, Any]:
        return {"node_id": self.node_id, "current_contract": self.current_contract}


@dataclass(frozen=True)
class V3StructuralAction:
    action_id: str
    field: str
    value: str
    current_value: str
    factor_id: str
    region_id: str
    level: str
    block_index: int
    targets: Tuple[V3StructuralTarget, ...]
    description: str
    parameters: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "action_id": self.action_id,
            "field": self.field,
            "value": self.value,
            "current_value": self.current_value,
            "factor_id": self.factor_id,
            "region_id": self.region_id,
            "level": self.level,
            "block_index": self.block_index,
            "targets": [item.to_dict() for item in self.targets],
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class V3StructuralRegion:
    factor_id: str
    region_id: str
    field: str
    level: str
    description: str

    def to_dict(self, *, actions: Sequence[V3StructuralAction] = ()) -> Mapping[str, Any]:
        selected = tuple(item for item in actions if item.field == self.field)
        return {
            "factor_id": self.factor_id,
            "region_id": self.region_id,
            "mutation_family": self.field,
            "mutation_level": self.level,
            "description": self.description,
            "editable_targets": sorted({target.node_id for action in selected for target in action.targets}),
            "allowed_action_ids": [item.action_id for item in selected],
            "allowed_values": [item.value for item in selected],
            "preserved_contracts": [
                "equivariant input/output types must remain valid",
                "the official V3 tensor dimensions remain frozen",
                "the rewritten graph must lower without an official constructor bypass",
            ],
        }


_STRUCTURAL_REGIONS = (
    V3StructuralRegion(
        "V3.S1",
        "v3_block_branch_topology",
        "block_branch_topology",
        "block",
        "在单个 V3 Block 内切换 Attention→FFN 串行计算与共享输入的并行双分支计算",
    ),
    V3StructuralRegion(
        "V3.S2",
        "v3_radial_operator_set",
        "radial_norm_presence",
        "operator",
        "在 Attention 径向 MLP 的局部阶段插入或删除可学习 LayerNorm 原语",
    ),
    V3StructuralRegion(
        "V3.S3",
        "v3_radial_operator_order",
        "radial_norm_activation_order",
        "operator",
        "重排 Attention 径向 MLP 内 LayerNorm 与标量激活原语的执行顺序",
    ),
    V3StructuralRegion(
        "V3.S4",
        "v3_logit_operator_order",
        "alpha_norm_activation_order",
        "operator",
        "重排 Attention logit 支路内 LayerNorm 与标量激活原语的执行顺序",
    ),
    V3StructuralRegion(
        "V3.S5",
        "v3_ffn_scalar_gate_reimplementation",
        "ffn_scalar_gate_orientation",
        "operator_internal",
        "在 FFN 标量门控算子内部重组激活支路，由 activation(gate)×value 改为 gate×activation(value)，或执行逆变换",
    ),
)


def v3_structural_regions(
    actions: Sequence[V3StructuralAction] = (),
) -> Tuple[V3StructuralRegion, ...]:
    if not actions:
        return _STRUCTURAL_REGIONS
    fields = {item.field for item in actions}
    return tuple(item for item in _STRUCTURAL_REGIONS if item.field in fields)


def _single_reference(node: Node, port: str) -> str:
    references = tuple(node.inputs.get(port, ()))
    if len(references) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_STRUCT_001",
                "structural rewrite requires one exact input reference",
                node_id=node.id,
                port=port,
                actual=str(references),
            )
        ])
    return references[0]


def _action(
    *,
    action_id: str,
    field: str,
    value: str,
    current_value: str,
    level: str,
    block_index: int,
    targets: Sequence[Tuple[str, str]],
    description: str,
    parameters: Mapping[str, Any],
) -> V3StructuralAction:
    region = next(item for item in _STRUCTURAL_REGIONS if item.field == field)
    return V3StructuralAction(
        action_id,
        field,
        value,
        current_value,
        region.factor_id,
        region.region_id,
        level,
        int(block_index),
        tuple(V3StructuralTarget(node_id, contract) for node_id, contract in targets),
        description,
        dict(parameters),
    )


def _block_topology_action(nodes: Mapping[str, Node], block_index: int):
    prefix = "block{}_".format(block_index)
    norm2 = nodes.get(prefix + "norm2")
    attention_residual = nodes.get(prefix + "attn_residual")
    if norm2 is None or attention_residual is None:
        return None
    current = _single_reference(norm2, "x")
    block_input = _single_reference(attention_residual, "left")
    sequential = prefix + "attn_residual"
    if current == sequential:
        value = "parallel_attention_ffn"
        description = "Block {} 的 FFN 改为读取 Attention 前的共享输入，末端仍与 Attention 结果残差合并".format(block_index)
        target_reference = block_input
    elif current == block_input:
        value = "sequential_attention_then_ffn"
        description = "Block {} 的 FFN 恢复为读取 Attention 残差输出".format(block_index)
        target_reference = sequential
    else:
        return None
    return _action(
        action_id="block{}:branch_topology={}".format(block_index, value),
        field="block_branch_topology",
        value=value,
        current_value="sequential_attention_then_ffn" if current == sequential else "parallel_attention_ffn",
        level="block",
        block_index=block_index,
        targets=((norm2.id, "x={}".format(current)),),
        description=description,
        parameters={"node_id": norm2.id, "current_reference": current, "target_reference": target_reference},
    )


def _radial_norm_presence_action(nodes: Mapping[str, Node], block_index: int, stage_index: int):
    prefix = "block{}_attn_radial_".format(block_index)
    linear_id = prefix + "linear{}".format(stage_index)
    norm_id = prefix + "norm{}".format(stage_index)
    activation_id = prefix + "act{}".format(stage_index)
    activation = nodes.get(activation_id)
    if activation is None:
        return None
    norm = nodes.get(norm_id)
    activation_input = _single_reference(activation, "x")
    if norm is not None and norm.op == "core.scalar_layer_norm@1" and activation_input == norm_id:
        value = "without_norm"
        current = "with_norm"
        parameters = {
            "direction": "remove",
            "norm_id": norm_id,
            "activation_id": activation_id,
            "replacement_reference": _single_reference(norm, "x"),
        }
    elif norm is None and activation_input == linear_id:
        value = "with_norm"
        current = "without_norm"
        parameters = {
            "direction": "insert",
            "norm_id": norm_id,
            "activation_id": activation_id,
            "source_reference": linear_id,
            "norm_attrs": {
                "affine": True,
                "axis": "edge_hidden",
                "bias": True,
                "epsilon": 1.0e-5,
            },
        }
    else:
        return None
    return _action(
        action_id="block{}:radial{}:norm={}".format(block_index, stage_index, value),
        field="radial_norm_presence",
        value=value,
        current_value=current,
        level="operator",
        block_index=block_index,
        targets=((norm_id, current), (activation_id, "x={}".format(activation_input))),
        description="Block {} 的径向 MLP 第 {} 段{} LayerNorm 原语".format(
            block_index, stage_index, "删除" if value == "without_norm" else "插入"
        ),
        parameters=parameters,
    )


def _order_action(
    nodes: Mapping[str, Node],
    *,
    block_index: int,
    field: str,
    locus: str,
    first_id: str,
    second_id: str,
    description_scope: str,
):
    first = nodes.get(first_id)
    second = nodes.get(second_id)
    if first is None or second is None:
        return None
    norm_op = "core.scalar_layer_norm@1"
    activation_op = "core.scalar_activation@1"
    if first.op == norm_op and second.op == activation_op and _single_reference(second, "x") == first.id:
        current = "norm_then_activation"
        value = "activation_then_norm"
    elif first.op == activation_op and second.op == norm_op and _single_reference(second, "x") == first.id:
        current = "activation_then_norm"
        value = "norm_then_activation"
    else:
        return None
    return _action(
        action_id="block{}:{}:{}={}".format(block_index, locus, field, value),
        field=field,
        value=value,
        current_value=current,
        level="operator",
        block_index=block_index,
        targets=((first.id, first.op), (second.id, second.op)),
        description="Block {} 的{}从 {} 重排为 {}".format(block_index, description_scope, current, value),
        parameters={
            "first_id": first.id,
            "second_id": second.id,
            "current_order": current,
            "target_order": value,
        },
    )


def _ffn_scalar_gate_orientation_action(nodes: Mapping[str, Node], block_index: int):
    prefix = "block{}_ffn_scalar_".format(block_index)
    gate_id = prefix + "gate"
    value_id = prefix + "up"
    activation_id = prefix + "gate_act"
    product_id = prefix + "product"
    activation = nodes.get(activation_id)
    product = nodes.get(product_id)
    if activation is None or product is None:
        return None
    activation_input = _single_reference(activation, "x")
    product_left = _single_reference(product, "left")
    product_right = _single_reference(product, "right")
    if (
        activation_input == gate_id
        and product_left == activation_id
        and product_right == value_id
    ):
        current = "activated_gate_times_value"
        value = "gate_times_activated_value"
        target_activation_input = value_id
        target_product_left = gate_id
        target_product_right = activation_id
    elif (
        activation_input == value_id
        and product_left == gate_id
        and product_right == activation_id
    ):
        current = "gate_times_activated_value"
        value = "activated_gate_times_value"
        target_activation_input = gate_id
        target_product_left = activation_id
        target_product_right = value_id
    else:
        return None
    return _action(
        action_id="block{}:ffn_scalar_gate_orientation={}".format(block_index, value),
        field="ffn_scalar_gate_orientation",
        value=value,
        current_value=current,
        level="operator_internal",
        block_index=block_index,
        targets=(
            (activation.id, "x={}".format(activation_input)),
            (product.id, "left={},right={}".format(product_left, product_right)),
        ),
        description=(
            "Block {} 的 FFN 标量门控算子由 {} 重实现为 {}"
            .format(block_index, current, value)
        ),
        parameters={
            "activation_id": activation.id,
            "product_id": product.id,
            "current_activation_input": activation_input,
            "current_product_left": product_left,
            "current_product_right": product_right,
            "target_activation_input": target_activation_input,
            "target_product_left": target_product_left,
            "target_product_right": target_product_right,
        },
    )


def v3_structural_mutation_catalog(program: ArchitectureProgram) -> Tuple[V3StructuralAction, ...]:
    """Enumerate reversible, type-checkable local graph rewrites for the current parent."""

    spec = v3_program_spec(program)
    nodes = {node.id: node for node in program.nodes}
    actions = []
    for block_index in range(spec.num_layers):
        topology = _block_topology_action(nodes, block_index)
        if topology is not None:
            actions.append(topology)
        for stage_index in (0, 1):
            presence = _radial_norm_presence_action(nodes, block_index, stage_index)
            if presence is not None:
                actions.append(presence)
            order = _order_action(
                nodes,
                block_index=block_index,
                field="radial_norm_activation_order",
                locus="radial{}".format(stage_index),
                first_id="block{}_attn_radial_norm{}".format(block_index, stage_index),
                second_id="block{}_attn_radial_act{}".format(block_index, stage_index),
                description_scope="径向 MLP 第 {} 段 Norm/Activation".format(stage_index),
            )
            if order is not None:
                actions.append(order)
        alpha_order = _order_action(
            nodes,
            block_index=block_index,
            field="alpha_norm_activation_order",
            locus="alpha",
            first_id="block{}_attn_alpha_norm".format(block_index),
            second_id="block{}_attn_alpha_activation".format(block_index),
            description_scope="Attention logit Norm/Activation",
        )
        if alpha_order is not None:
            actions.append(alpha_order)
        ffn_gate = _ffn_scalar_gate_orientation_action(nodes, block_index)
        if ffn_gate is not None:
            actions.append(ffn_gate)
    return tuple(actions)


def _replace_order_nodes(first: Node, second: Node) -> Tuple[Node, Node]:
    source_reference = _single_reference(first, "x")
    first_replacement = replace(
        first,
        op=second.op,
        inputs={"x": (source_reference,)},
        attrs=dict(second.attrs),
        declared_types={},
    )
    second_replacement = replace(
        second,
        op=first.op,
        inputs={"x": (first.id,)},
        attrs=dict(first.attrs),
        declared_types={},
    )
    return first_replacement, second_replacement


def build_v3_structural_patch(
    parent: ArchitectureProgram,
    action_id: str,
    registry: PrimitiveRegistry,
    *,
    hypothesis: Mapping[str, Any] = None,
) -> TypedPatch:
    actions = {item.action_id: item for item in v3_structural_mutation_catalog(parent)}
    if action_id not in actions:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_STRUCT_002",
                "selected structural action is not available for this parent",
                actual=action_id,
                details={"available": sorted(actions)},
            )
        ])
    action = actions[action_id]
    nodes = {node.id: node for node in parent.nodes}
    edits = []
    preconditions = []
    postconditions = []
    scope = [target.node_id for target in action.targets]

    if action.field == "block_branch_topology":
        node_id = str(action.parameters["node_id"])
        current = str(action.parameters["current_reference"])
        target = str(action.parameters["target_reference"])
        edits.append(PatchEdit("rewire_port", node_id, {"port": "x", "references": target}))
        preconditions.append({"kind": "node_input_equals", "node_id": node_id, "port": "x", "references": [current]})
        postconditions.append({"kind": "node_input_equals", "node_id": node_id, "port": "x", "references": [target]})
    elif action.field == "radial_norm_presence":
        norm_id = str(action.parameters["norm_id"])
        activation_id = str(action.parameters["activation_id"])
        if action.parameters["direction"] == "remove":
            replacement_reference = str(action.parameters["replacement_reference"])
            edits.append(PatchEdit("delete_if_bypassed", norm_id, {"replacement_reference": replacement_reference}))
            preconditions.extend((
                {"kind": "node_op_is", "node_id": norm_id, "op": "core.scalar_layer_norm@1"},
                {"kind": "node_input_equals", "node_id": activation_id, "port": "x", "references": [norm_id]},
            ))
            postconditions.extend((
                {"kind": "node_absent", "node_id": norm_id},
                {"kind": "node_input_equals", "node_id": activation_id, "port": "x", "references": [replacement_reference]},
            ))
        else:
            source_reference = str(action.parameters["source_reference"])
            inserted = Node(
                norm_id,
                "core.scalar_layer_norm@1",
                {"x": (source_reference,)},
                dict(action.parameters["norm_attrs"]),
                annotations=dict(nodes[activation_id].annotations),
            )
            edits.extend((
                PatchEdit("insert_before", activation_id, {"node": inserted.to_dict()}),
                PatchEdit("rewire_port", activation_id, {"port": "x", "references": norm_id}),
            ))
            preconditions.extend((
                {"kind": "node_absent", "node_id": norm_id},
                {"kind": "node_input_equals", "node_id": activation_id, "port": "x", "references": [source_reference]},
            ))
            postconditions.extend((
                {"kind": "node_op_is", "node_id": norm_id, "op": "core.scalar_layer_norm@1"},
                {"kind": "node_input_equals", "node_id": activation_id, "port": "x", "references": [norm_id]},
            ))
    elif action.field == "ffn_scalar_gate_orientation":
        activation_id = str(action.parameters["activation_id"])
        product_id = str(action.parameters["product_id"])
        current_activation_input = str(action.parameters["current_activation_input"])
        current_product_left = str(action.parameters["current_product_left"])
        current_product_right = str(action.parameters["current_product_right"])
        target_activation_input = str(action.parameters["target_activation_input"])
        target_product_left = str(action.parameters["target_product_left"])
        target_product_right = str(action.parameters["target_product_right"])
        edits.extend((
            PatchEdit(
                "rewire_port",
                activation_id,
                {"port": "x", "references": target_activation_input},
            ),
            PatchEdit(
                "rewire_port",
                product_id,
                {"port": "left", "references": target_product_left},
            ),
            PatchEdit(
                "rewire_port",
                product_id,
                {"port": "right", "references": target_product_right},
            ),
        ))
        preconditions.extend((
            {
                "kind": "node_input_equals",
                "node_id": activation_id,
                "port": "x",
                "references": [current_activation_input],
            },
            {
                "kind": "node_input_equals",
                "node_id": product_id,
                "port": "left",
                "references": [current_product_left],
            },
            {
                "kind": "node_input_equals",
                "node_id": product_id,
                "port": "right",
                "references": [current_product_right],
            },
        ))
        postconditions.extend((
            {
                "kind": "node_input_equals",
                "node_id": activation_id,
                "port": "x",
                "references": [target_activation_input],
            },
            {
                "kind": "node_input_equals",
                "node_id": product_id,
                "port": "left",
                "references": [target_product_left],
            },
            {
                "kind": "node_input_equals",
                "node_id": product_id,
                "port": "right",
                "references": [target_product_right],
            },
        ))
    else:
        first = nodes[str(action.parameters["first_id"])]
        second = nodes[str(action.parameters["second_id"])]
        first_replacement, second_replacement = _replace_order_nodes(first, second)
        edits.extend((
            PatchEdit("replace_node", first.id, {"node": first_replacement.to_dict()}),
            PatchEdit("replace_node", second.id, {"node": second_replacement.to_dict()}),
        ))
        preconditions.extend((
            {"kind": "node_op_is", "node_id": first.id, "op": first.op},
            {"kind": "node_op_is", "node_id": second.id, "op": second.op},
        ))
        postconditions.extend((
            {"kind": "node_op_is", "node_id": first.id, "op": second.op},
            {"kind": "node_op_is", "node_id": second.id, "op": first.op},
        ))

    claim = dict(hypothesis or {})
    if not str(claim.get("claim", "")).strip():
        claim["claim"] = action.description
    claim.update({
        "factor_id": action.factor_id,
        "region_id": action.region_id,
        "action_id": action.action_id,
        "mutation_family": action.field,
        "mutation_level": action.level,
        "block_index": action.block_index,
        "current_structure": action.current_value,
        "proposed_structure": action.value,
        "status": "未经训练验证的结构候选假设",
    })
    return TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        claim,
        tuple(dict.fromkeys(scope)),
        tuple(edits),
        tuple(preconditions),
        tuple(postconditions),
        {
            "equivariance_contract": "must remain type-valid and empirically audited",
            "parameter_shapes": "may change only when a parameterized primitive is inserted or deleted",
            "graph_topology": "changed",
            "training_effect": "hypothesis_only",
        },
    )


def apply_v3_structural_patch(
    parent: ArchitectureProgram,
    patch: TypedPatch,
    registry: PrimitiveRegistry,
) -> Tuple[ArchitectureProgram, Any]:
    child = apply_typed_patch(parent, patch, registry)
    parameters = dict(child.parameters)
    lowering_contract = dict(parameters.get("lowering_contract", {}))
    if "module_construction_order" in lowering_contract:
        previous_order = list(lowering_contract["module_construction_order"])
        child_ids = {node.id for node in child.nodes}
        construction_order = [node_id for node_id in previous_order if node_id in child_ids]
        new_nodes = [node for node in child.nodes if node.id not in set(previous_order)]
        for node in new_nodes:
            source_references = [reference for references in node.inputs.values() for reference in references]
            anchor = next((reference.split(":", 1)[0] for reference in source_references if reference.split(":", 1)[0] in construction_order), "")
            if anchor:
                construction_order.insert(construction_order.index(anchor) + 1, node.id)
            else:
                construction_order.append(node.id)
        lowering_contract["module_construction_order"] = construction_order
        parameters["lowering_contract"] = lowering_contract
    child = replace(child, parameters=parameters)
    TypeChecker(registry).check(child)
    parent_spec = v3_program_spec(parent)
    child_spec = v3_program_spec(child)
    if child_spec.to_dict() != parent_spec.to_dict():
        raise DSLValidationError([
            Diagnostic("E_V3_STRUCT_003", "local structural rewrite changed the frozen V3 tensor-dimension spec")
        ])
    parent_id = architecture_id(parent, registry)
    child_id = architecture_id(child, registry)
    if child_id == parent_id:
        raise DSLValidationError([
            Diagnostic("E_V3_STRUCT_004", "structural rewrite did not produce a new canonical architecture")
        ])
    return child, child_spec


def choose_deterministic_v3_structural_action(
    program: ArchitectureProgram,
    *,
    round_index: int,
    seen_states: Sequence[str] = (),
    registry: PrimitiveRegistry,
) -> V3StructuralAction:
    actions = v3_structural_mutation_catalog(program)
    seen = set(seen_states)
    if not actions:
        raise DSLValidationError([Diagnostic("E_V3_STRUCT_005", "parent exposes no certified structural rewrite")])
    offset = (int(round_index) - 1) % len(actions)
    ordered = actions[offset:] + actions[:offset]
    for action in ordered:
        patch = build_v3_structural_patch(program, action.action_id, registry)
        child, _ = apply_v3_structural_patch(program, patch, registry)
        if architecture_id(child, registry) not in seen:
            return action
    raise DSLValidationError([
        Diagnostic("E_V3_STRUCT_006", "all certified structural children are already present in the archive")
    ])


def v3_structural_novelty_report(
    parent: ArchitectureProgram,
    child: ArchitectureProgram,
    registry: PrimitiveRegistry,
) -> Mapping[str, Any]:
    """Return exact graph-edit evidence; architecture-ID uniqueness alone is insufficient."""

    parent_nodes = {node.id: node for node in parent.nodes}
    child_nodes = {node.id: node for node in child.nodes}
    shared = sorted(set(parent_nodes) & set(child_nodes))
    added = sorted(set(child_nodes) - set(parent_nodes))
    deleted = sorted(set(parent_nodes) - set(child_nodes))
    operator_changes = [
        {
            "node_id": node_id,
            "parent_op": parent_nodes[node_id].op,
            "child_op": child_nodes[node_id].op,
        }
        for node_id in shared
        if parent_nodes[node_id].op != child_nodes[node_id].op
    ]
    input_rewires = [
        {
            "node_id": node_id,
            "parent_inputs": {name: list(refs) for name, refs in parent_nodes[node_id].inputs.items()},
            "child_inputs": {name: list(refs) for name, refs in child_nodes[node_id].inputs.items()},
        }
        for node_id in shared
        if parent_nodes[node_id].inputs != child_nodes[node_id].inputs
    ]
    attribute_changes = [
        {
            "node_id": node_id,
            "parent_attrs": dict(parent_nodes[node_id].attrs),
            "child_attrs": dict(child_nodes[node_id].attrs),
        }
        for node_id in shared
        if parent_nodes[node_id].attrs != child_nodes[node_id].attrs
    ]
    graph_edit_count = len(added) + len(deleted) + len(operator_changes) + len(input_rewires)
    payload = {
        "parent_architecture_id": architecture_id(parent, registry),
        "child_architecture_id": architecture_id(child, registry),
        "parent_node_count": len(parent.nodes),
        "child_node_count": len(child.nodes),
        "node_count_delta": len(child.nodes) - len(parent.nodes),
        "added_nodes": added,
        "deleted_nodes": deleted,
        "operator_changes": operator_changes,
        "input_rewires": input_rewires,
        "attribute_changes": attribute_changes,
        "graph_edit_count": graph_edit_count,
        "is_structurally_novel": graph_edit_count > 0,
        "only_attribute_changed": graph_edit_count == 0 and bool(attribute_changes),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {**payload, "novelty_evidence_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
