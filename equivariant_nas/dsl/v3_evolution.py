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
from .reference_programs import equiformer_v3_direct_model_program, equiformer_v3_energy_model_program
from .registry import PrimitiveRegistry
from .backends.equiformer_v3_spec import EquiformerV3Spec


V3_FIRST_ROUND_MUTATION_VERSION = "v3-shape-preserving-2-three-stage"


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


@dataclass(frozen=True)
class V3MutationRegion:
    """One router-visible V3 leaf factor and its uniquely owned edit region."""

    factor_id: str
    region_id: str
    field: str
    description: str

    def to_dict(
        self,
        *,
        actions: Sequence[V3MutationAction] = (),
    ) -> Mapping[str, Any]:
        selected = tuple(item for item in actions if item.field == self.field)
        targets = selected[0].targets if selected else ()
        return {
            "factor_id": self.factor_id,
            "region_id": self.region_id,
            "mutation_family": self.field,
            "description": self.description,
            "editable_targets": [item.node_id for item in targets]
            + ["program.parameters.equiformer_v3_spec.{}".format(self.field)],
            "allowed_action_ids": [item.action_id for item in selected],
            "allowed_values": [item.value for item in selected],
            "preserved_contracts": [
                "equivariant input/output types unchanged",
                "parameter shapes unchanged",
                "all repeated runtime sites updated atomically",
            ],
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

_MUTATION_REGIONS = (
    V3MutationRegion("V3.F1", "v3_alpha_dropout", "alpha_drop", _MUTATION_DESCRIPTIONS["alpha_drop"]),
    V3MutationRegion(
        "V3.F2",
        "v3_attention_weight_dropout",
        "attn_weights_drop",
        _MUTATION_DESCRIPTIONS["attn_weights_drop"],
    ),
    V3MutationRegion("V3.F3", "v3_value_dropout", "value_drop", _MUTATION_DESCRIPTIONS["value_drop"]),
    V3MutationRegion(
        "V3.F4",
        "v3_graph_stochastic_depth",
        "drop_path_rate",
        _MUTATION_DESCRIPTIONS["drop_path_rate"],
    ),
    V3MutationRegion(
        "V3.F5",
        "v3_equivariant_projection_dropout",
        "proj_drop",
        _MUTATION_DESCRIPTIONS["proj_drop"],
    ),
    V3MutationRegion("V3.F6", "v3_ffn_dropout", "ffn_drop", _MUTATION_DESCRIPTIONS["ffn_drop"]),
)


def v3_mutation_regions(
    actions: Sequence[V3MutationAction] = (),
) -> Tuple[V3MutationRegion, ...]:
    """Return certified regions, optionally restricted to currently available actions."""

    if not actions:
        return _MUTATION_REGIONS
    fields = {item.field for item in actions}
    return tuple(item for item in _MUTATION_REGIONS if item.field in fields)


def v3_region_by_id(region_id: str) -> V3MutationRegion:
    matches = tuple(item for item in _MUTATION_REGIONS if item.region_id == str(region_id))
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic("E_V3_EVO_009", "unknown V3 mutation region", actual=str(region_id))
        ])
    return matches[0]


def v3_region_for_field(field: str) -> V3MutationRegion:
    matches = tuple(item for item in _MUTATION_REGIONS if item.field == str(field))
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic("E_V3_EVO_010", "unknown V3 mutation family", actual=str(field))
        ])
    return matches[0]


def parse_v3_router_response(
    value: Mapping[str, Any],
    regions: Sequence[V3MutationRegion],
) -> Mapping[str, Any]:
    """Validate the V3 Factor Router response without accepting an edit or value."""

    required = {"factor_id", "region_id", "rationale", "evidence_refs", "expected_value", "risk"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise DSLValidationError([
            Diagnostic("E_V3_LLM_001", "V3 factor router response does not match the exact schema")
        ])
    matches = tuple(item for item in regions if item.region_id == str(value["region_id"]))
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_002",
                "V3 factor router selected an unauthorized region",
                actual=str(value["region_id"]),
            )
        ])
    region = matches[0]
    if str(value["factor_id"]) != region.factor_id:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_003",
                "V3 factor router selected a factor that does not own the region",
                expected=region.factor_id,
                actual=str(value["factor_id"]),
            )
        ])
    if not str(value["rationale"]).strip() or not str(value["expected_value"]).strip():
        raise DSLValidationError([
            Diagnostic("E_V3_LLM_004", "V3 factor router omitted its decision rationale")
        ])
    if not isinstance(value["evidence_refs"], Sequence) or isinstance(value["evidence_refs"], (str, bytes)):
        raise DSLValidationError([
            Diagnostic("E_V3_LLM_005", "V3 factor router evidence_refs must be an array")
        ])
    return dict(value)


def parse_v3_critic_response(
    value: Mapping[str, Any],
    region: V3MutationRegion,
    *,
    immutable_mechanism: str = "",
) -> Mapping[str, Any]:
    """Validate a critic response and freeze the routed factor, region, and mechanism."""

    required = {
        "factor_id",
        "region_id",
        "claim",
        "mechanism",
        "edit_plan",
        "preserved_invariants",
        "evidence_refs",
        "uncertainty",
        "risk",
        "acceptance_metrics",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        fields = set(value) if isinstance(value, Mapping) else set()
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_006",
                "V3 region critic response does not match the exact schema",
                details={
                    "missing_fields": sorted(required - fields),
                    "unknown_fields": sorted(fields - required),
                },
            )
        ])
    if str(value["factor_id"]) != region.factor_id:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_007",
                "V3 region critic changed the routed factor",
                expected=region.factor_id,
                actual=str(value["factor_id"]),
            )
        ])
    if str(value["region_id"]) != region.region_id:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_008",
                "V3 region critic changed or widened the routed region",
                expected=region.region_id,
                actual=str(value["region_id"]),
            )
        ])
    if immutable_mechanism and str(value["mechanism"]) != immutable_mechanism:
        raise DSLValidationError([
            Diagnostic(
                "E_V3_LLM_009",
                "V3 critic protocol repair changed the proposed mechanism",
                expected=immutable_mechanism,
                actual=str(value["mechanism"]),
            )
        ])
    arrays = ("edit_plan", "preserved_invariants", "evidence_refs", "acceptance_metrics")
    if any(not isinstance(value[name], Sequence) or isinstance(value[name], (str, bytes)) for name in arrays):
        raise DSLValidationError([
            Diagnostic("E_V3_LLM_010", "V3 region critic list fields must be arrays")
        ])
    if not str(value["claim"]).strip() or not str(value["mechanism"]).strip() or not value["edit_plan"] or not value["acceptance_metrics"]:
        raise DSLValidationError([
            Diagnostic("E_V3_LLM_011", "V3 region critic omitted a falsifiable mechanism or edit plan")
        ])
    return dict(value)


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


def v3_program_from_spec(
    spec: EquiformerV3Spec,
    *,
    dtype: str = "float32",
    task_contract: str = "equiformer_v3_evolution_model",
) -> ArchitectureProgram:
    """Build the exact energy-only or direct Energy+Force V3 program for a frozen spec."""

    if spec.regress_stress:
        raise DSLValidationError([
            Diagnostic("E_V3_EVO_011", "V3 evolution does not yet support the stress head")
        ])
    builder = equiformer_v3_direct_model_program if spec.regress_forces else equiformer_v3_energy_model_program
    return builder(spec, dtype=dtype, task_contract=task_contract)


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
            regenerated = v3_program_from_spec(
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
    region = v3_region_for_field(action.field)
    claim = dict(hypothesis or {})
    if not str(claim.get("claim", "")).strip():
        claim["claim"] = action.description
    # These fields are compiler-owned identifiers.  An LLM hypothesis may add
    # scientific detail, but cannot relabel the routed factor or selected action.
    claim.update({
        "factor_id": region.factor_id,
        "region_id": region.region_id,
        "action_id": action.action_id,
        "mutation_family": action.field,
        "current_value": action.current_value,
        "proposed_value": action.value,
        "status": "未经训练验证的候选假设",
    })
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
    regenerated = v3_program_from_spec(
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
