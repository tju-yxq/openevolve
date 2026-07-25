"""Optional executable backends for typed architecture programs."""

from .e3nn_backend import BackendSupportReport, E3NNGraphBackend, FusedSubgraph
from .qm9_model import build_qm9_dsl_model
from .equiformer_v2_backend import (
    V2_REFERENCE_COMMIT,
    EquiformerV2GraphBackend,
    build_v2_so2_path,
    find_v2_fusion_patterns,
    load_equiformer_v2_modules,
)

__all__ = [
    "BackendSupportReport",
    "E3NNGraphBackend",
    "EquiformerV2GraphBackend",
    "FusedSubgraph",
    "V2_REFERENCE_COMMIT",
    "build_qm9_dsl_model",
    "build_v2_so2_path",
    "find_v2_fusion_patterns",
    "load_equiformer_v2_modules",
]
