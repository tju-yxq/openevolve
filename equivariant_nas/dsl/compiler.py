"""Compiler facade and conservative legacy Equiformer V1 adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Optional, Tuple

from ..builder import build_equiformer
from ..spec import ArchitectureSpec
from .ast import ArchitectureProgram, Node
from .canonicalize import architecture_id
from .diagnostics import DSLValidationError, Diagnostic
from .inference import InferenceResult, TypeChecker
from .motifs import MotifRegistry, expand_motifs
from .registry import PrimitiveRegistry
from .rewrites import RewriteStep, apply_strict_rewrites
from .task import TaskContract


@dataclass(frozen=True)
class CompilationArtifact:
    source_program: ArchitectureProgram
    expanded_program: ArchitectureProgram
    inference: InferenceResult
    backend: str
    architecture_id: str
    rewrite_trace: Tuple[RewriteStep, ...] = ()
    rewrite_registry_hash: str = ""


@dataclass(frozen=True)
class LoweringPlan:
    mode: str
    backend_family: str
    backend_semantics_version: str
    reference_model_identity: str = ""
    supported_regions: Tuple[str, ...] = ()
    unsupported_nodes: Tuple[str, ...] = ()
    details: Mapping[str, Any] = None

    def to_dict(self):
        return {
            "mode": self.mode,
            "backend_family": self.backend_family,
            "backend_semantics_version": self.backend_semantics_version,
            "reference_model_identity": self.reference_model_identity,
            "supported_regions": list(self.supported_regions),
            "unsupported_nodes": list(self.unsupported_nodes),
            "details": dict(self.details or {}),
        }

    def content_hash(self) -> str:
        text = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Compiler:
    def __init__(self, primitives: PrimitiveRegistry, motifs: Optional[MotifRegistry] = None):
        self.primitives = primitives
        self.motifs = motifs or MotifRegistry()

    def analyze(self, program: ArchitectureProgram, task: Optional[TaskContract] = None) -> CompilationArtifact:
        expanded_source = expand_motifs(program, self.motifs)
        rewrite_result = apply_strict_rewrites(expanded_source)
        expanded = rewrite_result.program
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
            rewrite_trace=rewrite_result.trace,
            rewrite_registry_hash=rewrite_result.registry_hash,
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

    def plan_lowering(
        self,
        program: ArchitectureProgram,
        task: Optional[TaskContract] = None,
    ) -> LoweringPlan:
        """Choose an explicit executable semantics before any model is built."""

        artifact = self.analyze(program, task)
        if program.annotations.get("legacy_backend") != "equiformer_v1":
            return LoweringPlan(
                "exact_node_graph",
                "e3nn_graph",
                "e3nn-graph-v1",
                supported_regions=("fully_lowered_core_graph",),
                details={"architecture_id": artifact.architecture_id},
            )

        unresolved = self.analyze(program)
        lock = str(program.annotations.get("legacy_lock_architecture_id", ""))
        if lock and unresolved.architecture_id == lock:
            return LoweringPlan(
                "exact_reference",
                "equiformer_v1",
                "equiformer-v1-official-constructor-v1",
                reference_model_identity="official_equiformer_v1_graph_attention_transformer",
                supported_regions=("reference_model",),
                details={"legacy_lock_architecture_id": lock},
            )

        from .backends.hybrid_v1 import parse_v1_readout_hybrid
        from .regions import v1_region_registry

        hybrid = parse_v1_readout_hybrid(program)
        if hybrid is not None:
            restored_nodes = tuple(
                Node(
                    "graph_pool",
                    "core.global_pool",
                    {"x": ("scalar_readout",)},
                    declared_types={"out": program.outputs[0].expected_type},
                )
                if node.id == "graph_pool"
                else node
                for node in program.nodes
            )
            restored = replace(program, nodes=restored_nodes)
            restored_id = self.analyze(restored).architecture_id
            if restored_id != lock:
                return LoweringPlan(
                    "representation_only",
                    "none",
                    "representation-flow-only-v1",
                    reference_model_identity="official_equiformer_v1_graph_attention_transformer",
                    unsupported_nodes=tuple(node.id for node in program.nodes),
                    details={
                        "reason": "readout-shaped graph also changed structure outside the certified region",
                        "legacy_lock_architecture_id": lock,
                        "restored_architecture_id": restored_id,
                    },
                )
            regions = v1_region_registry(program)
            return LoweringPlan(
                "exact_hybrid",
                "equiformer_v1",
                "equiformer-v1-readout-hybrid-v1",
                reference_model_identity="official_equiformer_v1_graph_attention_transformer",
                supported_regions=tuple(item.region_id for item in regions),
                details=hybrid.to_dict(),
            )
        return LoweringPlan(
            "representation_only",
            "none",
            "representation-flow-only-v1",
            reference_model_identity="official_equiformer_v1_graph_attention_transformer",
            unsupported_nodes=tuple(node.id for node in program.nodes),
            details={
                "reason": "modified imported V1 graph has no certified exact or hybrid lowering",
                "legacy_lock_architecture_id": lock,
                "actual_architecture_id": unresolved.architecture_id,
            },
        )
