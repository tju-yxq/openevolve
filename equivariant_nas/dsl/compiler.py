"""Compiler facade and conservative legacy Equiformer V1 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from ..builder import build_equiformer
from ..spec import ArchitectureSpec
from .ast import ArchitectureProgram
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import InferenceResult, TypeChecker
from .motifs import MotifRegistry, expand_motifs
from .registry import PrimitiveRegistry
from .task import TaskContract


@dataclass(frozen=True)
class CompilationArtifact:
    source_program: ArchitectureProgram
    expanded_program: ArchitectureProgram
    inference: InferenceResult
    backend: str
    architecture_id: str


class Compiler:
    def __init__(self, primitives: PrimitiveRegistry, motifs: Optional[MotifRegistry] = None):
        self.primitives = primitives
        self.motifs = motifs or MotifRegistry()

    def analyze(self, program: ArchitectureProgram, task: Optional[TaskContract] = None) -> CompilationArtifact:
        expanded = expand_motifs(program, self.motifs)
        inference = TypeChecker(self.primitives).check(expanded)
        task_hash = "unresolved-task-contract"
        if task is not None:
            if program.task_contract != task.task_id:
                raise DSLValidationError([
                    Diagnostic("E_TASK_005", "program references a different task contract", expected=task.task_id, actual=program.task_contract)
                ])
            if len(program.outputs) != 1 or program.outputs[0].expected_type != task.output_type:
                raise DSLValidationError([Diagnostic("E_TASK_006", "program output declaration differs from the task output contract")])
            task_hash = task.content_hash()
        return CompilationArtifact(
            source_program=program,
            expanded_program=expanded,
            inference=inference,
            backend="backend-neutral",
            architecture_id=architecture_id(expanded, self.primitives, task_contract_hash=task_hash),
        )

    def lower_legacy_equiformer_v1(
        self,
        program: ArchitectureProgram,
        equiformer_root: str,
        task_mean=None,
        task_std=None,
        atomref=None,
    ):
        artifact = self.analyze(program)
        annotations = program.annotations
        if annotations.get("legacy_backend") != "equiformer_v1":
            raise DSLValidationError([Diagnostic("E_BACKEND_001", "program is not registered for the legacy V1 backend")])
        lock = str(annotations.get("legacy_lock_architecture_id", ""))
        if artifact.architecture_id != lock:
            raise DSLValidationError([
                Diagnostic(
                    "E_BACKEND_002",
                    "the imported V1 graph changed and cannot use constructor-locked lowering",
                    expected=lock,
                    actual=artifact.architecture_id,
                    repairs=("use node-level lowering for the modified operations", "re-import an unchanged legacy spec"),
                )
            ])
        spec = ArchitectureSpec.from_dict(annotations["legacy_architecture_spec"])
        return build_equiformer(spec, equiformer_root, task_mean=task_mean, task_std=task_std, atomref=atomref)
