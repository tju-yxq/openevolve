"""First-round, shape-preserving Typed DSL evolution for Equiformer V3."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram, Node
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import TypeChecker
from .patch import PatchEdit, TypedPatch, apply_typed_patch
from .reference_programs import equiformer_v3_direct_model_program
from .registry import PrimitiveRegistry
from .backends.equiformer_v3_spec import EquiformerV3Spec


V3_FIRST_ROUND_MUTATION_VERSION = "v3-shape-preserving-1"


@dataclass(frozen=True)
class V3MutationTarget:
    node_id: str
    attr: str
    current_value: float

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "node_id": self.node_id,
            "attr": self.attr,
            "current_value": self.current_value,
        }


@dataclass(frozen=True)
class V3MutationAction:
    action_id: str
    field: str
    value: float
    current_value: float
    targets: Tuple[V3MutationTarget, ...]
    description: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "action_id": self.action_id,
            "field": self.field,
            "value": self.value,
            "current_value": self.current_value,
            "targets": [item.to_dict() for item in self.targets],
            "description": self.description,
        }


_MUTATION_VALUES = {
    "alpha_drop": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    "attn_weights_drop": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    "value_drop": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    "drop_path_rate": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    "proj_drop": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
    "ffn_drop": (0.0, 0.02, 0.05, 0.10, 0.15, 0.20),
}

_MUTATION_DESCRIPTIONS = {
    "alpha_drop": "改变注意力 alpha 标量路径的 dropout 概率",
    "attn_weights_drop": "改变归一化注意力权重的 dropout 概率",
    "value_drop": "同时改变注意力值路径的标量与 S2 网格 dropout",
    "drop_path_rate": "同时改变每个 V3 block 的注意力与 FFN 图级随机深度",
    "proj_drop": "同时改变每个 V3 block 的注意力与 FFN 等变投影 dropout",
    "ffn_drop": "同时改变每个 V3 block 的 FFN 标量与 S2 网格 dropout",
}


def v3_program_spec(program: ArchitectureProgram) -> EquiformerV3Spec:
    payload = program.parameters.get("equiformer_v3_spec")
    if not isinstance(payload, Mapping):
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EVO_001",
                "V3 evolution requires program.parameters.equiformer_v3_spec",
            )
        ])
    try:
        return EquiformerV3Spec.from_mapping(payload, strict=True)
    except ValueError as error:
        raise DSLValidationError([
            Diagnostic("E_V3_EVO_002", "V3 program spec is invalid", actual=str(error))
        ]) from error


def _target_contract(field: str, node: Node) -> str:
    contracts = {
        "alpha_drop": (r"block\d+_attn_alpha_dropout", "core.scalar_dropout@1", "p"),
        "attn_weights_drop": (
            r"block\d+_attn_attention_weight_dropout",
            "core.scalar_dropout@1",
            "p",
        ),
        "value_drop": (
            r"block\d+_attn_gated_activation",
            "core.s2_gated_swiglu_merge@1",
            "dropout",
        ),
        "drop_path_rate": (
            r"block\d+_(?:attn|ffn)_drop_path",
            "core.graph_stochastic_depth@1",
            "p",
        ),
        "proj_drop": (
            r"block\d+_(?:attn|ffn)_proj_drop",
            "core.equivariant_dropout@1",
            "p",
        ),
        "ffn_drop": (
            r"block\d+_ffn_(?:scalar|grid)_dropout",
            "",
            "",
        ),
    }
    pattern, expected_op, attr = contracts[field]
    if re.fullmatch(pattern, node.id) is None:
        return ""
    if field == "ffn_drop":
        if node.op == "core.scalar_dropout@1":
            return "p"
        if node.op == "core.grid_dropout@1":
            return "probability"
        return ""
    if node.op != expected_op:
        return ""
    return attr


def _mutation_targets(program: ArchitectureProgram, field: str) -> Tuple[V3MutationTarget, ...]:
    spec = v3_program_spec(program)
    targets = []
    for node in program.nodes:
        attr = _target_contract(field, node)
        if not attr:
            continue
        if attr not in node.attrs:
            raise DSLValidationError([
                Diagnostic(
                    "E_V3_EVO_003",
                    "V3 mutation target lacks its contracted attribute",
                    node_id=node.id,
                    actual=attr,
                )
            ])
        targets.append(V3MutationTarget(node.id, attr, float(node.attrs[attr])))
    expected_count = spec.num_layers * (2 if field in {"drop_path_rate", "proj_drop", "ffn_drop"} else 1)
    expected_value = float(getattr(spec, field))
    if not targets and expected_value == 0.0:
        # Some official V3 branches replace a zero-probability dropout with an
        # identity and therefore contain no editable dropout node.  Enabling
        # such a branch requires an insertion patch and is intentionally left
        # for the second-round structural mutation surface.
        return ()
    if len(targets) != expected_count:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EVO_004",
                "V3 mutation family does not cover every required block path",
                expected=str(expected_count),
                actual=str(len(targets)),
                details={"field": field, "targets": [item.node_id for item in targets]},
            )
        ])
    inconsistent = [item.node_id for item in targets if item.current_value != expected_value]
    if inconsistent:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EVO_005",
                "V3 node attributes disagree with the frozen program spec",
                expected=str(expected_value),
                details={"field": field, "nodes": inconsistent},
            )
        ])
    return tuple(targets)


def v3_first_round_mutation_catalog(program: ArchitectureProgram) -> Tuple[V3MutationAction, ...]:
    """Enumerate certified first-round mutations for the current V3 parent."""

    spec = v3_program_spec(program)
    actions = []
    for field, values in _MUTATION_VALUES.items():
        targets = _mutation_targets(program, field)
        if not targets:
            continue
        current = float(getattr(spec, field))
        for value in values:
            if float(value) == current:
                continue
            candidate_spec = EquiformerV3Spec.from_mapping(
                {**spec.to_dict(), field: float(value)},
                strict=True,
            )
            regenerated = equiformer_v3_direct_model_program(
                candidate_spec,
                task_contract=program.task_contract,
            )
            parent_structure = tuple((node.id, node.op) for node in program.nodes)
            candidate_structure = tuple((node.id, node.op) for node in regenerated.nodes)
            if candidate_structure != parent_structure:
                continue
            actions.append(
                V3MutationAction(
                    action_id="{}={:.2f}".format(field, float(value)),
                    field=field,
                    value=float(value),
                    current_value=current,
                    targets=targets,
                    description=_MUTATION_DESCRIPTIONS[field],
                )
            )
    return tuple(actions)


def build_v3_first_round_patch(
    parent: ArchitectureProgram,
    action_id: str,
    registry: PrimitiveRegistry,
    *,
    hypothesis: Mapping[str, Any] = None,
) -> TypedPatch:
    """Materialize one catalog action as a complete transactional Typed Patch."""

    actions = {item.action_id: item for item in v3_first_round_mutation_catalog(parent)}
    if action_id not in actions:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EVO_006",
                "selected V3 mutation action is not available for this parent",
                actual=action_id,
                details={"available": sorted(actions)},
            )
        ])
    action = actions[action_id]
    nodes = {node.id: node for node in parent.nodes}
    edits = []
    preconditions = []
    postconditions = []
    for target in action.targets:
        node = nodes[target.node_id]
        attrs = dict(node.attrs)
        attrs[target.attr] = action.value
        edits.append(PatchEdit("change_attrs", node.id, {"attrs": attrs}))
        preconditions.append({
            "kind": "node_attr_equals",
            "node_id": node.id,
            "attr": target.attr,
            "value": target.current_value,
        })
        postconditions.append({
            "kind": "node_attr_equals",
            "node_id": node.id,
            "attr": target.attr,
            "value": action.value,
        })
    parameter_target = "program.parameters.equiformer_v3_spec.{}".format(action.field)
    edits.append(PatchEdit("change_parameters", parameter_target, {"value": action.value}))
    claim = {
        "claim": action.description,
        "action_id": action.action_id,
        "mutation_family": action.field,
        "current_value": action.current_value,
        "proposed_value": action.value,
        "status": "未经训练验证的候选假设",
    }
    claim.update(dict(hypothesis or {}))
    return TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        claim,
        tuple([item.node_id for item in action.targets] + [parameter_target]),
        tuple(edits),
        tuple(preconditions),
        tuple(postconditions),
        {
            "parameter_shapes": "unchanged",
            "equivariance_contract": "unchanged",
            "training_effect": "hypothesis_only",
        },
    )


def apply_v3_first_round_patch(
    parent: ArchitectureProgram,
    patch: TypedPatch,
    registry: PrimitiveRegistry,
) -> Tuple[ArchitectureProgram, EquiformerV3Spec]:
    """Apply, type-check, and prove the patch equals regeneration from its V3 spec."""

    child = apply_typed_patch(parent, patch, registry)
    inference = TypeChecker(registry).check(child)
    del inference
    spec = v3_program_spec(child)
    regenerated = equiformer_v3_direct_model_program(
        spec,
        dtype=str(child.parameters.get("dtype", "float32")),
        task_contract=child.task_contract,
    )
    actual_id = architecture_id(child, registry)
    expected_id = architecture_id(regenerated, registry)
    if actual_id != expected_id:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_EVO_007",
                "V3 Typed Patch does not equal regeneration from the updated official spec",
                expected=expected_id,
                actual=actual_id,
            )
        ])
    return child, spec


def v3_evolution_state(program: ArchitectureProgram) -> str:
    spec = v3_program_spec(program)
    payload = {field: float(getattr(spec, field)) for field in sorted(_MUTATION_VALUES)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def choose_deterministic_v3_action(
    program: ArchitectureProgram,
    *,
    round_index: int,
    seen_states: Sequence[str] = (),
) -> V3MutationAction:
    """Choose a reproducible diverse action for smoke tests and offline fallback."""

    actions = v3_first_round_mutation_catalog(program)
    fields = tuple(_MUTATION_VALUES)
    preferred = fields[(int(round_index) - 1) % len(fields)]
    ordered = tuple(item for item in actions if item.field == preferred) + tuple(
        item for item in actions if item.field != preferred
    )
    seen = set(seen_states)
    spec = v3_program_spec(program)
    base = spec.to_dict()
    for action in ordered:
        candidate = dict(base)
        candidate[action.field] = action.value
        state = json.dumps(
            {field: float(candidate[field]) for field in sorted(_MUTATION_VALUES)},
            sort_keys=True,
            separators=(",", ":"),
        )
        if state not in seen:
            return action
    raise DSLValidationError([
        Diagnostic("E_V3_EVO_008", "no unseen first-round V3 mutation action remains")
    ])
