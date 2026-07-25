"""EvoEquiLang: a typed DSL for equivariant neural architectures."""

from .ast import ArchitectureProgram, InputPort, Node, OutputPort
from .canonicalize import BACKEND_SEMANTICS_VERSION, COMPILER_SEMANTICS_VERSION, architecture_id, canonicalize
from .compiler import Compiler
from .completion import AvailableValue, CompletionAction, CompletionDistance, HoleSink, TypedHole, complete_typed_hole, materialize_completion_patch, program_completion_frontier
from .cost import CostEstimate, enforce_static_resource_contract, estimate_static_cost
from .diagnostics import DSLValidationError, Diagnostic
from .evidence_store import EvidenceStore
from .groups import GroupSpec
from .inference import InferenceResult, TypeChecker
from .irreps import Irrep, Irreps
from .language import LanguageVersion, VocabularyDecision, describe_active_vocabulary, select_active_vocabulary
from .llm_protocol import EvidenceItem, parse_patch_response, parse_planner_response, planner_prompt, repair_prompt, synthesizer_prompt
from .motifs import MotifDefinition, MotifRegistry, expand_motifs
from .patch import PatchEdit, TypedPatch, apply_typed_patch, patch_protocol_schema
from .reference_motifs import reference_motif_registry
from .reference_programs import import_equiformer_v1
from .repair_completion import completion_repair_suggestions
from .rewrites import RewriteRuleDescriptor, RewriteStep, StrictRewriteResult, apply_strict_rewrites, strict_rewrite_registry_hash
from .registry import PrimitiveRegistry, core_registry
from .task import ResourceContract, TaskContract, task_reasoning_context, validate_task_reasoning
from .search import DSLGenerationEngine, GenerationResult
from .types import Carrier, EquivarianceLevel, EquivariantType, Frame

__all__ = [
    "ArchitectureProgram",
    "BACKEND_SEMANTICS_VERSION",
    "Carrier",
    "AvailableValue",
    "CompletionAction",
    "CompletionDistance",
    "CostEstimate",
    "Compiler",
    "COMPILER_SEMANTICS_VERSION",
    "DSLValidationError",
    "DSLGenerationEngine",
    "Diagnostic",
    "EquivarianceLevel",
    "EquivariantType",
    "EvidenceItem",
    "EvidenceStore",
    "Frame",
    "HoleSink",
    "GroupSpec",
    "GenerationResult",
    "InferenceResult",
    "InputPort",
    "Irrep",
    "Irreps",
    "LanguageVersion",
    "MotifDefinition",
    "MotifRegistry",
    "Node",
    "OutputPort",
    "PrimitiveRegistry",
    "PatchEdit",
    "TypedPatch",
    "TypedHole",
    "TypeChecker",
    "ResourceContract",
    "RewriteRuleDescriptor",
    "RewriteStep",
    "StrictRewriteResult",
    "TaskContract",
    "task_reasoning_context",
    "validate_task_reasoning",
    "architecture_id",
    "apply_typed_patch",
    "apply_strict_rewrites",
    "patch_protocol_schema",
    "canonicalize",
    "complete_typed_hole",
    "completion_repair_suggestions",
    "materialize_completion_patch",
    "program_completion_frontier",
    "core_registry",
    "describe_active_vocabulary",
    "enforce_static_resource_contract",
    "estimate_static_cost",
    "expand_motifs",
    "import_equiformer_v1",
    "reference_motif_registry",
    "parse_patch_response",
    "parse_planner_response",
    "planner_prompt",
    "repair_prompt",
    "select_active_vocabulary",
    "strict_rewrite_registry_hash",
    "synthesizer_prompt",
    "VocabularyDecision",
]
