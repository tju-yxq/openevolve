"""End-to-end typed candidate generation over an OpenEvolve-compatible LLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .compiler import CompilationArtifact, Compiler
from .canonicalize import COMPILER_SEMANTICS_VERSION
from .diagnostics import DSLValidationError, Diagnostic
from .evidence_store import EvidenceStore
from .language import VocabularyDecision
from .llm_protocol import EvidenceItem, parse_patch_response, parse_planner_response, parse_region_critic_response, parse_region_router_response, planner_prompt, planner_repair_prompt, region_critic_prompt, region_router_prompt, repair_prompt, synthesizer_prompt
from .patch import TypedPatch, apply_typed_patch
from .repair_completion import completion_repair_suggestions
from .regions import RegionDefinition, region_by_id, validate_region_transition
from .task import TaskContract
from .task import validate_task_reasoning


@dataclass(frozen=True)
class GenerationResult:
    parent: CompilationArtifact
    child: CompilationArtifact
    patch: TypedPatch
    planner_response: Mapping[str, Any]
    repair_count: int
    planner_repair_count: int = 0
    router_response: Mapping[str, Any] = field(default_factory=dict)
    critic_response: Mapping[str, Any] = field(default_factory=dict)
    region_audit: Mapping[str, Any] = field(default_factory=dict)


class DSLGenerationEngine:
    """Generate, repair, compile, and persist one typed child transaction."""

    def __init__(self, compiler: Compiler, task: TaskContract, vocabulary: VocabularyDecision, store: EvidenceStore, *, model_name: str, repair_attempts: int = 2):
        self.compiler = compiler
        self.task = task
        self.vocabulary = vocabulary
        self.store = store
        self.model_name = model_name
        self.repair_attempts = int(repair_attempts)

    async def generate(self, ensemble, parent_program, evidence: Sequence[EvidenceItem], scope_candidates: Sequence[str]) -> GenerationResult:
        parent = self.compiler.analyze(parent_program, self.task)
        parent_id = self.store.add_compiled_candidate(parent, self.task)
        plan_prompt = planner_prompt(
            self.task,
            parent_program,
            evidence,
            self.vocabulary,
            scope_candidates,
            self.compiler.primitives,
            self.compiler.motifs,
        )
        current_plan_prompt = plan_prompt
        plan_text = await self._call(ensemble, plan_prompt)
        plan = None
        planner_repair_count = 0
        for plan_attempt in range(self.repair_attempts + 1):
            self.store.add_prompt_run(
                parent_id,
                role="planner" if plan_attempt == 0 else "planner_repairer",
                model=self.model_name,
                prompt=current_plan_prompt,
                response_text=plan_text,
                vocabulary=self.vocabulary,
                evidence_ids=[item.evidence_id for item in evidence],
                visible_splits=[item.split for item in evidence],
                token_usage={},
            )
            try:
                plan = parse_planner_response(plan_text, scope_candidates)
                validate_task_reasoning(self.task, plan)
                planner_repair_count = plan_attempt
                break
            except DSLValidationError as exc:
                if plan_attempt >= self.repair_attempts:
                    self.store.add_compiler_run(parent_id, COMPILER_SEMANTICS_VERSION, "planner_protocol_failed", diagnostics=exc.diagnostics)
                    raise
                current_plan_prompt = planner_repair_prompt(plan_prompt, plan_text, exc.diagnostics)
                plan_text = await self._call(ensemble, current_plan_prompt)
        if plan is None:
            raise RuntimeError("planner repair loop exited without a plan")
        synth_prompt = synthesizer_prompt(
            self.task,
            parent_program,
            plan,
            self.vocabulary,
            parent_id,
            self.compiler.primitives,
            self.compiler.motifs,
        )
        response = await self._call(ensemble, synth_prompt)
        current_prompt = synth_prompt
        diagnostics = ()
        failed_patch = None
        for attempt in range(self.repair_attempts + 1):
            child_program = None
            self.store.add_prompt_run(
                parent_id,
                role="synthesizer" if attempt == 0 else "repairer",
                model=self.model_name,
                prompt=current_prompt,
                response_text=response,
                vocabulary=self.vocabulary,
                evidence_ids=[item.evidence_id for item in evidence],
                visible_splits=[item.split for item in evidence],
                token_usage={},
            )
            try:
                patch = parse_patch_response(response)
                if tuple(patch.scope) != tuple(plan["scope"]):
                    raise DSLValidationError([
                        Diagnostic(
                            "E_LLM_010",
                            "repair or synthesis changed the planner-authorized scope",
                            expected=str(list(plan["scope"])),
                            actual=str(list(patch.scope)),
                        )
                    ])
                failed_patch = patch
                child_program = apply_typed_patch(
                    parent_program,
                    patch,
                    self.compiler.primitives,
                    task_contract_hash=self.task.content_hash(),
                    expected_parent_id=parent_id,
                    validate_child_with_core_registry=False,
                )
                child = self.compiler.analyze(child_program, self.task)
                if child.architecture_id == parent_id or self.store.candidate_exists(child.architecture_id):
                    raise DSLValidationError([
                        Diagnostic("E_SEARCH_001", "generated child is a duplicate semantic architecture", actual=child.architecture_id)
                    ])
                child_id = self.store.add_compiled_candidate(child, self.task)
                self.store.add_patch(patch, child_architecture_id=child_id)
                self.store.add_compiler_run(
                    child_id,
                    COMPILER_SEMANTICS_VERSION,
                    "success",
                    inference=child.inference,
                    rewrite_trace=child.rewrite_trace,
                    rewrite_registry_hash=child.rewrite_registry_hash,
                )
                return GenerationResult(parent, child, patch, plan, attempt, planner_repair_count)
            except (DSLValidationError, ValueError) as exc:
                diagnostics = exc.diagnostics if isinstance(exc, DSLValidationError) else ()
                if attempt >= self.repair_attempts:
                    self.store.add_compiler_run(parent_id, COMPILER_SEMANTICS_VERSION, "generation_failed", diagnostics=diagnostics)
                    raise
                if failed_patch is None:
                    failed_patch = TypedPatch("1.0", parent_id, parent_program.language_version, dict(plan), tuple(plan["scope"]), ())
                suggestions = (
                    completion_repair_suggestions(
                        child_program,
                        diagnostics,
                        self.compiler.primitives,
                        allowed_ops=tuple(name for name in self.vocabulary.visible if name.startswith("core.")),
                        authorized_scope=tuple(plan["scope"]),
                    )
                    if child_program is not None and diagnostics
                    else ()
                )
                correction = repair_prompt(
                    failed_patch,
                    diagnostics,
                    self.vocabulary,
                    parent=parent_program,
                    parent_architecture_id=parent_id,
                    rejected_response=response,
                    primitives=self.compiler.primitives,
                    motifs=self.compiler.motifs,
                    immutable_scope=tuple(plan["scope"]),
                    completion_suggestions=suggestions,
                )
                response = await self._call(ensemble, correction)
                current_prompt = correction
        raise RuntimeError("unreachable generation state")

    async def generate_region_candidate(
        self,
        ensemble,
        parent_program,
        evidence: Sequence[EvidenceItem],
        regions: Sequence[RegionDefinition],
    ) -> GenerationResult:
        """Generate one candidate through Router, Critic, Synthesizer, then conditional Repair."""

        if not regions:
            raise DSLValidationError([Diagnostic("E_REGION_007", "no certified region is available for this parent")])
        parent = self.compiler.analyze(parent_program, self.task)
        parent_id = self.store.add_compiled_candidate(parent, self.task)

        router_request = region_router_prompt(self.task, parent_program, evidence, regions)
        router_text = await self._call(ensemble, router_request)
        self.store.add_prompt_run(
            parent_id,
            role="region_router",
            model=self.model_name,
            prompt=router_request,
            response_text=router_text,
            vocabulary=self.vocabulary,
            evidence_ids=[item.evidence_id for item in evidence],
            visible_splits=[item.split for item in evidence],
            token_usage={},
        )
        router = parse_region_router_response(router_text, regions)
        region = region_by_id(regions, str(router["region_id"]))

        critic_request = region_critic_prompt(
            self.task,
            parent_program,
            evidence,
            region,
            router,
            self.vocabulary,
            self.compiler.primitives,
            self.compiler.motifs,
        )
        critic_text = await self._call(ensemble, critic_request)
        self.store.add_prompt_run(
            parent_id,
            role="region_critic",
            model=self.model_name,
            prompt=critic_request,
            response_text=critic_text,
            vocabulary=self.vocabulary,
            evidence_ids=[item.evidence_id for item in evidence],
            visible_splits=[item.split for item in evidence],
            token_usage={},
        )
        critic = parse_region_critic_response(critic_text, region)
        plan = {
            "claim": critic["claim"],
            "scope": list(region.editable_targets),
            "abstract_goals": list(critic["edit_plan"]),
            "evidence_refs": list(critic["evidence_refs"]),
            "uncertainty": critic["uncertainty"],
            "risk": critic["risk"],
            "region_id": region.region_id,
            "mechanism": critic["mechanism"],
            "preserved_invariants": list(critic["preserved_invariants"]),
            "acceptance_metrics": list(critic["acceptance_metrics"]),
            "allowed_region_ops": list(region.allowed_ops),
            "boundary_sources": list(region.boundary_sources),
            "region_backend_capability": region.backend_capability,
        }
        synth_request = synthesizer_prompt(
            self.task,
            parent_program,
            plan,
            self.vocabulary,
            parent_id,
            self.compiler.primitives,
            self.compiler.motifs,
        )
        response = await self._call(ensemble, synth_request)
        current_prompt = synth_request
        diagnostics = ()
        failed_patch = None
        for attempt in range(self.repair_attempts + 1):
            child_program = None
            self.store.add_prompt_run(
                parent_id,
                role="patch_synthesizer" if attempt == 0 else "compiler_guided_repairer",
                model=self.model_name,
                prompt=current_prompt,
                response_text=response,
                vocabulary=self.vocabulary,
                evidence_ids=[item.evidence_id for item in evidence],
                visible_splits=[item.split for item in evidence],
                token_usage={},
            )
            try:
                patch = parse_patch_response(response)
                if tuple(patch.scope) != tuple(region.editable_targets):
                    raise DSLValidationError([
                        Diagnostic(
                            "E_LLM_010",
                            "synthesis or repair changed the routed region scope",
                            expected=str(list(region.editable_targets)),
                            actual=str(list(patch.scope)),
                        )
                    ])
                failed_patch = patch
                child_program = apply_typed_patch(
                    parent_program,
                    patch,
                    self.compiler.primitives,
                    task_contract_hash=self.task.content_hash(),
                    expected_parent_id=parent_id,
                    validate_child_with_core_registry=False,
                )
                region_audit = validate_region_transition(parent_program, child_program, region)
                child = self.compiler.analyze(child_program, self.task)
                lowering = self.compiler.plan_lowering(child_program, self.task)
                if lowering.mode != region.backend_capability:
                    raise DSLValidationError([
                        Diagnostic(
                            "E_REGION_008",
                            "generated region has no certified executable lowering",
                            expected=region.backend_capability,
                            actual=lowering.mode,
                            details=lowering.to_dict(),
                        )
                    ])
                if child.architecture_id == parent_id or self.store.candidate_exists(child.architecture_id):
                    raise DSLValidationError([
                        Diagnostic("E_SEARCH_001", "generated child is a duplicate semantic architecture", actual=child.architecture_id)
                    ])
                child_id = self.store.add_compiled_candidate(child, self.task)
                self.store.add_patch(patch, child_architecture_id=child_id)
                self.store.add_compiler_run(
                    child_id,
                    COMPILER_SEMANTICS_VERSION,
                    "success",
                    inference=child.inference,
                    rewrite_trace=child.rewrite_trace,
                    rewrite_registry_hash=child.rewrite_registry_hash,
                )
                return GenerationResult(
                    parent,
                    child,
                    patch,
                    plan,
                    attempt,
                    0,
                    router,
                    critic,
                    dict(region_audit, lowering_plan=lowering.to_dict()),
                )
            except (DSLValidationError, ValueError) as exc:
                diagnostics = exc.diagnostics if isinstance(exc, DSLValidationError) else ()
                if attempt >= self.repair_attempts:
                    self.store.add_compiler_run(parent_id, COMPILER_SEMANTICS_VERSION, "generation_failed", diagnostics=diagnostics)
                    raise
                if failed_patch is None:
                    failed_patch = TypedPatch("1.0", parent_id, parent_program.language_version, dict(plan), tuple(region.editable_targets), ())
                suggestions = (
                    completion_repair_suggestions(
                        child_program,
                        diagnostics,
                        self.compiler.primitives,
                        allowed_ops=tuple(name for name in self.vocabulary.visible if name.startswith("core.")),
                        authorized_scope=tuple(region.editable_targets),
                    )
                    if child_program is not None and diagnostics
                    else ()
                )
                correction = repair_prompt(
                    failed_patch,
                    diagnostics,
                    self.vocabulary,
                    parent=parent_program,
                    parent_architecture_id=parent_id,
                    rejected_response=response,
                    primitives=self.compiler.primitives,
                    motifs=self.compiler.motifs,
                    immutable_scope=tuple(region.editable_targets),
                    completion_suggestions=suggestions,
                )
                response = await self._call(ensemble, correction)
                current_prompt = correction
        raise RuntimeError("unreachable region generation state")

    @staticmethod
    async def _call(ensemble, prompt):
        return await ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
