"""Auditable Planner-Synthesizer-Repairer contracts for LLM generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .ast import ArchitectureProgram
from .completion import program_completion_frontier
from .diagnostics import DSLValidationError, Diagnostic
from .language import VocabularyDecision, describe_active_vocabulary
from .motifs import MotifRegistry
from .patch import TypedPatch, patch_protocol_schema
from .registry import PrimitiveRegistry
from .regions import RegionDefinition
from .task import TaskContract, task_reasoning_context


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    split: str
    kind: str
    payload: Mapping[str, Any]
    status: str = "measurement"

    def __post_init__(self) -> None:
        if self.status not in ("measurement", "inference", "hypothesis"):
            raise DSLValidationError([Diagnostic("E_LLM_001", "invalid evidence status", actual=self.status)])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "split": self.split,
            "kind": self.kind,
            "payload": dict(self.payload),
            "status": self.status,
        }


def _load_json_object(text: str, role: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        raise DSLValidationError([Diagnostic("E_LLM_006", "{} response must be text".format(role), actual=type(text).__name__)])
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```" or lines[0].strip() not in ("```", "```json", "```JSON"):
            raise DSLValidationError([Diagnostic("E_LLM_007", "{} response has an invalid code fence".format(role))])
        stripped = "\n".join(lines[1:-1]).strip()
    elif "```" in stripped:
        # Some providers prepend a short explanation despite an explicit JSON-only
        # contract. Accept exactly one fenced JSON object deterministically; the
        # original response remains in the append-only evidence store for audit.
        import re

        blocks = re.findall(r"```(?:json|JSON)?\s*\n?(.*?)```", stripped, flags=re.DOTALL)
        if len(blocks) != 1 or stripped.count("```") != 2:
            raise DSLValidationError([Diagnostic("E_LLM_007", "{} response must contain exactly one JSON code fence".format(role))])
        stripped = blocks[0].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise DSLValidationError([
            Diagnostic(
                "E_LLM_008",
                "{} response is not one JSON object".format(role),
                details={"line": exc.lineno, "column": exc.colno, "reason": exc.msg},
            )
        ])
    if not isinstance(value, dict):
        raise DSLValidationError([Diagnostic("E_LLM_009", "{} response must be a JSON object".format(role))])
    return value


def _visible_evidence(task: TaskContract, evidence: Sequence[EvidenceItem]) -> Tuple[EvidenceItem, ...]:
    visible = []
    for item in evidence:
        task.assert_evidence_visible(item.split)
        visible.append(item)
    return tuple(visible)


def planner_prompt(
    task: TaskContract,
    parent: ArchitectureProgram,
    evidence: Sequence[EvidenceItem],
    vocabulary: VocabularyDecision,
    scope_candidates: Sequence[str],
    primitives: PrimitiveRegistry = None,
    motifs: MotifRegistry = None,
) -> Dict[str, str]:
    visible = _visible_evidence(task, evidence)
    system = (
        "You are the scientific planner for a typed equivariant architecture search. "
        "Propose one falsifiable structural hypothesis. Separate measured evidence from "
        "inference and uncertainty. Do not output code or an architecture patch. Never "
        "modify the task, data split, training protocol, evaluator, or trusted equivariance rules. "
        "Return exactly one JSON object matching response_schema, with no Markdown or commentary."
    )
    payload = {
        "task_id": task.task_id,
        "group": task.group.to_dict(),
        "output_type": task.output_type.to_dict(),
        "trusted_task_semantics": dict(task_reasoning_context(task)),
        # Patches are source-location based. Never expose canonicalized node ids
        # here because they are semantic-hash labels, not editable source ids.
        "parent_program": parent.to_dict(),
        "evidence": [item.to_dict() for item in visible],
        "visible_vocabulary": list(vocabulary.visible),
        "vocabulary_contracts": (
            list(describe_active_vocabulary(vocabulary, primitives, motifs))
            if primitives is not None and motifs is not None
            else []
        ),
        "scope_candidates": list(scope_candidates),
        "task_completion_frontier": (
            list(
                program_completion_frontier(
                    parent,
                    task.output_type,
                    primitives,
                    allowed_ops=tuple(name for name in vocabulary.visible if name.startswith("core.")),
                    max_steps=3,
                )
            )
            if primitives is not None
            else []
        ),
        "scope_semantics": [
            "scope is an immutable capability set, not a topic label",
            "include every existing node that a later patch must target",
            "if the hypothesis requires rewiring a downstream consumer, include that consumer in scope now",
            "program outputs are separate capabilities named output:<name>; include one only when the output source must change",
            "repair cannot widen scope after planning",
            "inserted new node ids do not need to appear in scope, but the existing insert_before/insert_after target does",
        ],
        "response_schema": {
            "claim": "falsifiable structural claim",
            "scope": ["one or more allowed node or output:<name> capabilities"],
            "abstract_goals": ["typed structural property"],
            "evidence_refs": ["evidence ids"],
            "uncertainty": "main uncertainty",
            "risk": "main correctness or optimization risk",
        },
    }
    return {"system": system, "user": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def region_router_prompt(
    task: TaskContract,
    parent: ArchitectureProgram,
    evidence: Sequence[EvidenceItem],
    regions: Sequence[RegionDefinition],
) -> Dict[str, str]:
    visible = _visible_evidence(task, evidence)
    payload = {
        "task_id": task.task_id,
        "trusted_task_semantics": dict(task_reasoning_context(task)),
        "parent_summary": {
            "program_id": parent.program_id,
            "language_version": parent.language_version,
            "node_ids": [node.id for node in parent.nodes],
        },
        "measured_evidence": [item.to_dict() for item in visible],
        "regions": [item.to_dict() for item in regions],
        "response_schema": {
            "factor_id": "exactly one capability-enabled leaf factor id",
            "region_id": "exactly one registered region id",
            "rationale": "why this region has the highest expected information value",
            "evidence_refs": ["visible evidence ids"],
            "expected_value": "falsifiable expected benefit",
            "risk": "main risk",
        },
    }
    return {
        "system": (
            "You are the leaf-factor router for a typed equivariant architecture search. "
            "Select exactly one capability-enabled leaf factor and its uniquely owned region. "
            "Do not propose node edits or code. "
            "Return exactly one JSON object matching response_schema."
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def parse_region_router_response(text: str, regions: Sequence[RegionDefinition]) -> Dict[str, Any]:
    value = _load_json_object(text, "region_router")
    required = {"factor_id", "region_id", "rationale", "evidence_refs", "expected_value", "risk"}
    if set(value) != required:
        raise DSLValidationError([Diagnostic("E_LLM_011", "region router response does not match the exact schema")])
    allowed = {item.region_id for item in regions}
    if value["region_id"] not in allowed:
        raise DSLValidationError([
            Diagnostic("E_LLM_012", "region router selected an unauthorized region", actual=str(value["region_id"]))
        ])
    selected = next(item for item in regions if item.region_id == value["region_id"])
    if value["factor_id"] != selected.factor_id:
        raise DSLValidationError([
            Diagnostic(
                "E_LLM_017",
                "factor router selected a factor that does not own the selected region",
                expected=selected.factor_id,
                actual=str(value["factor_id"]),
            )
        ])
    if not value["rationale"] or not value["expected_value"]:
        raise DSLValidationError([Diagnostic("E_LLM_013", "region router omitted its decision rationale")])
    return value


def region_critic_prompt(
    task: TaskContract,
    parent: ArchitectureProgram,
    evidence: Sequence[EvidenceItem],
    region: RegionDefinition,
    router_response: Mapping[str, Any],
    vocabulary: VocabularyDecision,
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
) -> Dict[str, str]:
    visible = _visible_evidence(task, evidence)
    visible_ids = set(region.boundary_sources) | {
        item for item in region.editable_targets if not item.startswith("output:")
    }
    local_nodes = [node.to_dict() for node in parent.nodes if node.id in visible_ids]
    payload = {
        "task_id": task.task_id,
        "selected_region": region.to_dict(),
        "router_decision": dict(router_response),
        "local_parent_ast": local_nodes,
        "local_outputs": [
            item.to_dict() for item in parent.outputs
            if "output:" + item.name in region.editable_targets
        ],
        "evidence": [item.to_dict() for item in visible],
        "allowed_vocabulary_contracts": list(describe_active_vocabulary(vocabulary, primitives, motifs)),
        "response_schema": {
            "factor_id": region.factor_id,
            "region_id": region.region_id,
            "claim": "one falsifiable local structural hypothesis",
            "mechanism": "why the proposed local computation could affect the target",
            "edit_plan": ["ordered abstract edits inside the region"],
            "preserved_invariants": ["region and task invariants that must remain true"],
            "evidence_refs": ["visible evidence ids"],
            "uncertainty": "main uncertainty",
            "risk": "main correctness or optimization risk",
            "acceptance_metrics": ["validation-only measurements that could falsify the claim"],
        },
    }
    return {
        "system": (
            "You are the independent leaf-factor critic. Analyze only the selected factor, region, and declared boundary. "
            "Produce a mechanistic, falsifiable edit plan but no patch or source code. Do not widen the region. "
            "Return exactly one JSON object matching response_schema."
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def parse_region_critic_response(text: str, region: RegionDefinition) -> Dict[str, Any]:
    value = _load_json_object(text, "region_critic")
    required = {
        "factor_id", "region_id", "claim", "mechanism", "edit_plan", "preserved_invariants",
        "evidence_refs", "uncertainty", "risk", "acceptance_metrics",
    }
    if set(value) != required:
        raise DSLValidationError([Diagnostic("E_LLM_014", "region critic response does not match the exact schema")])
    if value["region_id"] != region.region_id:
        raise DSLValidationError([
            Diagnostic("E_LLM_015", "region critic widened or changed the routed region", expected=region.region_id, actual=str(value["region_id"]))
        ])
    if value["factor_id"] != region.factor_id:
        raise DSLValidationError([
            Diagnostic("E_LLM_018", "factor critic changed the routed leaf factor", expected=region.factor_id, actual=str(value["factor_id"]))
        ])
    if not value["claim"] or not value["mechanism"] or not value["edit_plan"] or not value["acceptance_metrics"]:
        raise DSLValidationError([Diagnostic("E_LLM_016", "region critic omitted a falsifiable mechanism or edit plan")])
    return value


def region_critic_repair_prompt(
    original_prompt: Mapping[str, str],
    response_text: str,
    diagnostics: Sequence[Diagnostic],
    region: RegionDefinition,
) -> Dict[str, str]:
    """Repair only the critic protocol envelope without changing its routed factor."""

    original_request = json.loads(original_prompt["user"])
    payload = {
        "instruction": (
            "Rewrite the rejected region critic response as exactly one JSON object matching "
            "response_schema. Preserve the original scientific claim and evidence. Do not change "
            "factor_id, region_id, mechanism, or proposed intervention. Do not add Markdown, code "
            "fences, source code, new evidence, or extra keys."
        ),
        "immutable_factor_id": region.factor_id,
        "immutable_region_id": region.region_id,
        "response_schema": original_request["response_schema"],
        "rejected_response": response_text[:12000],
        "parser_diagnostics": [item.to_dict() for item in diagnostics],
    }
    return {
        "system": (
            "You are a strict region-critic JSON protocol repairer. Output one JSON object and "
            "nothing else. The routed factor and region are immutable."
        ),
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def planner_repair_prompt(
    original_prompt: Mapping[str, str],
    response_text: str,
    diagnostics: Sequence[Diagnostic],
) -> Dict[str, str]:
    payload = {
        "instruction": "Rewrite the rejected planner response as exactly one JSON object. Preserve only claims supported by the original response. Do not add Markdown, code fences, prose, architecture code, or new evidence.",
        "original_request": json.loads(original_prompt["user"]),
        "rejected_response": response_text[:12000],
        "parser_diagnostics": [item.to_dict() for item in diagnostics],
    }
    return {
        "system": "You are a strict JSON protocol repairer. Output one JSON object and nothing else.",
        "user": json.dumps(payload, ensure_ascii=False, sort_keys=True),
    }


def synthesizer_prompt(
    task: TaskContract,
    parent: ArchitectureProgram,
    plan: Mapping[str, Any],
    vocabulary: VocabularyDecision,
    parent_architecture_id: str,
    primitives: PrimitiveRegistry = None,
    motifs: MotifRegistry = None,
) -> Dict[str, str]:
    system = (
        "You synthesize strict EvoEquiLang typed patches. Use only the visible vocabulary "
        "and declared scope. Existing node target ids MUST be copied exactly from parent_program; "
        "program output targets MUST use output:<name>. "
        "never invent canonical n0000-style aliases. Every edit MUST contain exactly kind, "
        "target, and payload. Return one JSON object matching authoritative_patch_schema and no prose. "
        "The compiler, task contract, and trusted kernel are immutable. Expected accuracy "
        "effects are hypotheses, never measurements."
    )
    parameter_scope = [item for item in plan["scope"] if str(item).startswith("constructor.")]
    worked_example = (
        {
            "explanation": "For constructor-factor regions, emit only change_parameters edits on exact scoped paths.",
            "edits": [
                {
                    "kind": "change_parameters",
                    "target": parameter_scope[0],
                    "payload": {"value": "choose one capability-admitted value with the correct JSON type"},
                }
            ],
        }
        if parameter_scope
        else {
            "explanation": "Illustrative graph edit list only. Copy actual ids, ports, ops, attrs, and references from parent_program and visible_vocabulary.",
            "edits": [
                {
                    "kind": "insert_before",
                    "target": "existing_scoped_node_id",
                    "payload": {
                        "node": {
                            "id": "new_unique_node_id",
                            "op": "visible.op",
                            "inputs": {"x": ["existing_predecessor_id"]},
                            "attrs": {},
                            "outputs": ["out"],
                        }
                    },
                },
                {
                    "kind": "rewire_port",
                    "target": "existing_scoped_node_id",
                    "payload": {"port": "x", "references": ["new_unique_node_id"]},
                },
            ],
        }
    )
    payload = {
        "task_id": task.task_id,
        "parent_architecture_id": parent_architecture_id,
        "language_version": parent.language_version,
        "parent_program": parent.to_dict(),
        "plan": dict(plan),
        "visible_vocabulary": list(vocabulary.visible),
        "task_completion_frontier": (
            list(
                program_completion_frontier(
                    parent,
                    task.output_type,
                    primitives,
                    allowed_ops=tuple(name for name in vocabulary.visible if name.startswith("core.")),
                    max_steps=3,
                )
            )
            if primitives is not None
            else []
        ),
        "vocabulary_contracts": (
            list(describe_active_vocabulary(vocabulary, primitives, motifs))
            if primitives is not None and motifs is not None
            else []
        ),
        "authoritative_patch_schema": patch_protocol_schema(parent_architecture_id, parent.language_version, tuple(plan["scope"])),
        "worked_edit_example": worked_example,
    }
    return {"system": system, "user": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def repair_prompt(
    failed_patch: TypedPatch,
    diagnostics: Sequence[Diagnostic],
    vocabulary: VocabularyDecision,
    *,
    parent: ArchitectureProgram = None,
    parent_architecture_id: str = "",
    rejected_response: str = "",
    primitives: PrimitiveRegistry = None,
    motifs: MotifRegistry = None,
    immutable_scope: Sequence[str] = (),
    completion_suggestions: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, str]:
    system = (
        "Repair a typed patch using only the compiler diagnostics. Preserve its scientific "
        "hypothesis, parent, language version, and scope. Do not weaken postconditions or "
        "change the task. Use ONLY edits shaped as {kind,target,payload}; anchor, mode, "
        "new_subgraph, insert_node, and rewire are not valid fields or edit kinds. Existing "
        "target ids MUST be copied exactly from parent_program. Return one strict JSON object only."
    )
    resolved_parent_id = parent_architecture_id or failed_patch.parent_architecture_id
    language_version = parent.language_version if parent is not None else failed_patch.language_version
    payload = {
        "failed_patch": failed_patch.to_dict(),
        "rejected_response": rejected_response[:12000],
        "diagnostics": [item.to_dict() for item in diagnostics],
        "visible_vocabulary": list(vocabulary.visible),
        "vocabulary_contracts": (
            list(describe_active_vocabulary(vocabulary, primitives, motifs))
            if primitives is not None and motifs is not None
            else []
        ),
        "parent_program": parent.to_dict() if parent is not None else None,
        "authoritative_patch_schema": patch_protocol_schema(resolved_parent_id, language_version, immutable_scope),
        "trusted_completion_suggestions": [dict(item) for item in completion_suggestions],
        "repair_rules": [
            "Return the complete patch envelope, not only the edits array.",
            "Each edit has exactly kind, target, payload.",
            "Use insert_before or insert_after with payload.node to add a node.",
            "Use rewire_port with payload.port and payload.references to reconnect a consumer.",
            "Use rewire_output with payload.reference and target output:<name> to reconnect a declared program output.",
            "All edit targets must be existing source node ids or output:<name> capabilities within scope.",
            "The scope array is immutable and must exactly equal authoritative_patch_schema.properties.scope.const.",
            "preconditions and postconditions must be empty or use only authoritative_patch_schema.condition_contracts; never place natural-language condition fields in them.",
            "Every inserted node must contribute to a declared program output; dead subgraphs are rejected.",
            "Inserted node ids must be new and references must use parent_program source ids.",
            "Trusted completion suggestions come from compiler type rules, not measurements; use only scope-compatible reachable paths.",
            "A completion suggestion is optional advice, not permission to widen scope or alter the scientific hypothesis.",
        ],
    }
    return {"system": system, "user": json.dumps(payload, ensure_ascii=False, sort_keys=True)}


def parse_planner_response(text: str, allowed_scopes: Sequence[str]) -> Dict[str, Any]:
    value = _load_json_object(text, "planner")
    required = {"claim", "scope", "abstract_goals", "evidence_refs", "uncertainty", "risk"}
    if not isinstance(value, dict) or set(value) != required:
        raise DSLValidationError([Diagnostic("E_LLM_002", "planner response does not match the exact schema")])
    if not value["claim"] or not value["abstract_goals"]:
        raise DSLValidationError([Diagnostic("E_LLM_003", "planner must provide a claim and abstract goals")])
    invalid = set(value["scope"]) - set(allowed_scopes)
    if invalid:
        raise DSLValidationError([Diagnostic("E_LLM_004", "planner selected an unauthorized scope", details={"scopes": sorted(invalid)})])
    return value


def parse_patch_response(text: str) -> TypedPatch:
    value = _load_json_object(text, "patch")
    return TypedPatch.from_dict(value)
