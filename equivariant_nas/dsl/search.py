"""End-to-end typed candidate generation over an OpenEvolve-compatible LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .compiler import CompilationArtifact, Compiler
from .diagnostics import DSLValidationError, Diagnostic
from .evidence_store import EvidenceStore
from .language import VocabularyDecision
from .llm_protocol import EvidenceItem, parse_patch_response, parse_planner_response, planner_prompt, planner_repair_prompt, repair_prompt, synthesizer_prompt
from .patch import TypedPatch, apply_typed_patch
from .repair_completion import completion_repair_suggestions
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
                    self.store.add_compiler_run(parent_id, "evoequilang-1", "planner_protocol_failed", diagnostics=exc.diagnostics)
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
                self.store.add_compiler_run(child_id, "evoequilang-1", "success", inference=child.inference)
                return GenerationResult(parent, child, patch, plan, attempt, planner_repair_count)
            except (DSLValidationError, ValueError) as exc:
                diagnostics = exc.diagnostics if isinstance(exc, DSLValidationError) else ()
                if attempt >= self.repair_attempts:
                    self.store.add_compiler_run(parent_id, "evoequilang-1", "generation_failed", diagnostics=diagnostics)
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

    @staticmethod
    async def _call(ensemble, prompt):
        return await ensemble.generate_with_context(
            system_message=prompt["system"],
            messages=[{"role": "user", "content": prompt["user"]}],
        )
