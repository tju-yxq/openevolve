"""Optional executable backends for typed architecture programs."""

from .e3nn_backend import BackendSupportReport, E3NNGraphBackend, FusedSubgraph
from .lowering import CERTIFIED_EXACTNESS, DependencyRequirement, LoweringRule, LoweringRuleRegistry
from .hybrid_v1 import V1ReadoutHybridSpec, parse_v1_readout_hybrid
from .equiformer_v1_constructor import baseline_constructor_parameters, changed_constructor_parameters, effective_v1_spec
from .equiformer_v1_builder import add_equiformer_to_path, build_equiformer, count_trainable_parameters
from .equiformer_v1_spec import ArchitectureSpec, SpecValidationError, baseline_spec
from .equiformer_v3_spec import (
    EquiformerV3Spec,
    V3_REFERENCE_COMMIT,
    V3SpecImportManifest,
    baseline_v3_spec,
    official_v3_oc_spec,
)
from .v3_checkpoint import (
    equiformer_v3_direct_checkpoint_manifest,
    equiformer_v3_direct_parameter_mapping,
    export_equiformer_v3_direct_checkpoint,
    load_equiformer_v3_direct_checkpoint,
)
from .equiformer_v2_backend import (
    V2_REFERENCE_COMMIT,
    EquiformerV2GraphBackend,
    build_v2_so2_path,
    find_v2_fusion_patterns,
    load_equiformer_v2_modules,
)
from .v3_runtime import (
    V3_OPERATOR_SEMANTICS,
    equiformer_v3_source_available,
    load_equiformer_v3_modules,
    resolve_equiformer_v3_package_path,
)

__all__ = [
    "BackendSupportReport",
    "CERTIFIED_EXACTNESS",
    "DependencyRequirement",
    "E3NNGraphBackend",
    "EquiformerV2GraphBackend",
    "FusedSubgraph",
    "LoweringRule",
    "LoweringRuleRegistry",
    "V2_REFERENCE_COMMIT",
    "V3_REFERENCE_COMMIT",
    "V3SpecImportManifest",
    "V3_OPERATOR_SEMANTICS",
    "V1ReadoutHybridSpec",
    "parse_v1_readout_hybrid",
    "baseline_constructor_parameters",
    "changed_constructor_parameters",
    "effective_v1_spec",
    "add_equiformer_to_path",
    "build_equiformer",
    "count_trainable_parameters",
    "ArchitectureSpec",
    "EquiformerV3Spec",
    "equiformer_v3_direct_checkpoint_manifest",
    "equiformer_v3_direct_parameter_mapping",
    "export_equiformer_v3_direct_checkpoint",
    "SpecValidationError",
    "baseline_spec",
    "baseline_v3_spec",
    "official_v3_oc_spec",
    "build_v2_so2_path",
    "find_v2_fusion_patterns",
    "load_equiformer_v2_modules",
    "load_equiformer_v3_modules",
    "load_equiformer_v3_direct_checkpoint",
    "equiformer_v3_source_available",
    "resolve_equiformer_v3_package_path",
]
