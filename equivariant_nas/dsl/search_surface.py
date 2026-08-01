"""Canonical, auditable search exposure over the versioned execution registry.

The core registry is an execution and compatibility registry.  It intentionally
contains historical versions, checkpoint-specific parameterizations, layout
adapters, and temporary fused operators.  Treating every registered name as an
independent LLM mutation choice creates fake architectural novelty and many
semantically duplicate candidates.

This module keeps the execution registry intact while assigning every concrete
primitive and motif to exactly one search role:

* ``generatable``: an LLM-authored patch may introduce the concrete realization;
* ``completion_only``: trusted typed completion may introduce it as an adapter;
* ``context_only``: existing/imported programs may contain it, but new patches
  must not introduce it.

Concrete realizations are additionally grouped under canonical mathematical
families for prompt presentation and novelty accounting.  The grouping does not
change executable semantics or architecture identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .diagnostics import DSLValidationError, Diagnostic
from .motifs import MotifRegistry
from .registry import PrimitiveRegistry


SEARCH_SURFACE_VERSION = "evoequilang-canonical-search-surface-v1"


_CONTEXT_ONLY_PRIMITIVES = frozenset({
    "core.categorical_embedding@2",
    "core.change_multiplicity@1",
    "core.cutoff_envelope@1",
    "core.distance@1",
    "core.edge_lift@1",
    "core.edge_frame_gate_activation@1",
    "core.endpoint_gather@1",
    "core.equivariant_merge_norm@1",
    "core.global_pool@1",
    "core.identity@1",
    "core.invariant_dropout@1",
    "core.invariant_weight@1",
    "core.norm_activation@1",
    "core.radial_basis@1",
    "core.relative_position@1",
    "core.residual_add@1",
    "core.s2_activation@1",
    "core.s2_gated_swiglu_merge@1",
    "core.s2_swiglu@1",
    "core.scalar_linear@3",
    "core.scalar_linear@4",
    "core.segment_mean@1",
    "core.segment_softmax@1",
    "core.segment_sum@1",
    "core.separable_s2_activation@1",
    "core.stochastic_depth@1",
    "core.so3_linear@2",
})


_COMPLETION_ONLY_PRIMITIVES = frozenset({
    "core.axisymmetric_spherical_lift@1",
    "core.categorical_one_hot@1",
    "core.categorical_remap@1",
    "core.constant_scale@1",
    "core.endpoint_gather@2",
    "core.flatten_invariant_axes@1",
    "core.from_edge_frame@1",
    "core.from_edge_frame@2",
    "core.head_merge@1",
    "core.head_merge@2",
    "core.head_split@1",
    "core.head_split@2",
    "core.invariant_slice@1",
    "core.irrep_pad@1",
    "core.irrep_select@2",
    "core.irrep_slice@1",
    "core.periodic_displacement@1",
    "core.periodic_displacement@2",
    "core.relative_displacement@2",
    "core.relative_displacement@3",
    "core.select_scalars@1",
    "core.select_scalars@2",
    "core.grid_project@1",
    "core.grid_unproject@1",
    "core.grid_split@1",
    "core.grid_concat@1",
    "core.to_edge_frame@1",
    "core.to_edge_frame@2",
})


_GENERATABLE_PRIMITIVES = frozenset({
    "core.categorical_embedding@1",
    "core.cutoff_envelope@2",
    "core.degreewise_invariant_scale@1",
    "core.distance@2",
    "core.equivariant_channel_concat@1",
    "core.equivariant_dropout@1",
    "core.equivariant_norm@1",
    "core.fixed_gaussian_radial_basis@1",
    "core.gate@1",
    "core.gaussian_radial_basis@1",
    "core.graph_stochastic_depth@1",
    "core.grid_channel_linear@1",
    "core.grid_dropout@1",
    "core.grid_pointwise_activation@1",
    "core.grid_pointwise_product@1",
    "core.headwise_scalar_contraction@1",
    "core.headwise_scalar_contraction@2",
    "core.headwise_scalar_contraction@3",
    "core.invariant_compatibility@1",
    "core.invariant_concat@1",
    "core.invariant_product@1",
    "core.invariant_scale@1",
    "core.invariant_scale@2",
    "core.irrep_concat@1",
    "core.irrep_layer_norm@1",
    "core.irrep_linear@1",
    "core.irrep_linear@2",
    "core.residual_add@2",
    "core.scalar_activation@1",
    "core.scalar_dropout@1",
    "core.scalar_layer_norm@1",
    "core.scalar_linear@1",
    "core.scalar_linear@2",
    "core.scalar_offset@1",
    "core.segment_reduce@1",
    "core.segment_softmax@2",
    "core.segment_softmax@3",
    "core.so2_convolution@1",
    "core.so2_linear@1",
    "core.so2_linear@2",
    "core.so3_linear@1",
    "core.spherical_harmonics@1",
    "core.tensor_product@1",
    "core.tensor_product@2",
    "core.tensor_product@3",
    "core.tensor_product@4",
    "core.tensor_product@5",
})


_GENERATABLE_MOTIFS = frozenset({
    "motif.v1_feed_forward@1",
    "motif.v1_radial_profile@1",
})


_CONTEXT_ONLY_MOTIFS = frozenset({
    "motif.v1_initial_message@1",
    "motif.v1_multilevel_readout@1",
    "motif.v1_residual_message@1",
    "motif.v2_so2_residual_message@1",
    "motif.v3_edge_degree_embedding@1",
    "motif.v3_transformer_block@1",
})


_CANONICAL_OVERRIDES = {
    # Mathematical aliases and version families that must not count as distinct
    # innovation merely because their concrete runtime layout differs.
    "core.change_multiplicity": "canonical.equivariant_linear",
    "core.irrep_linear": "canonical.equivariant_linear",
    "core.so3_linear": "canonical.equivariant_linear",
    "core.invariant_concat": "canonical.typed_concat",
    "core.equivariant_channel_concat": "canonical.typed_concat",
    "core.irrep_concat": "canonical.typed_concat",
    "core.constant_scale": "canonical.typed_multiply",
    "core.degreewise_invariant_scale": "canonical.typed_multiply",
    "core.invariant_product": "canonical.typed_multiply",
    "core.invariant_scale": "canonical.typed_multiply",
    "core.invariant_weight": "canonical.typed_multiply",
    "core.edge_lift": "canonical.endpoint_gather",
    "core.endpoint_gather": "canonical.endpoint_gather",
    "core.global_pool": "canonical.segment_reduce",
    "core.segment_mean": "canonical.segment_reduce",
    "core.segment_reduce": "canonical.segment_reduce",
    "core.segment_sum": "canonical.segment_reduce",
    "core.fixed_gaussian_radial_basis": "canonical.radial_basis",
    "core.gaussian_radial_basis": "canonical.radial_basis",
    "core.radial_basis": "canonical.radial_basis",
    "core.relative_position": "canonical.relative_displacement",
    "core.relative_displacement": "canonical.relative_displacement",
    "core.invariant_slice": "canonical.typed_select",
    "core.irrep_select": "canonical.typed_select",
    "core.irrep_slice": "canonical.typed_select",
    "core.select_scalars": "canonical.typed_select",
    "core.equivariant_dropout": "canonical.typed_dropout",
    "core.invariant_dropout": "canonical.typed_dropout",
    "core.scalar_dropout": "canonical.typed_dropout",
    "core.equivariant_merge_norm": "canonical.equivariant_normalization",
    "core.equivariant_norm": "canonical.equivariant_normalization",
    "core.irrep_layer_norm": "canonical.equivariant_normalization",
    "core.graph_stochastic_depth": "canonical.graph_drop_path",
    "core.stochastic_depth": "canonical.graph_drop_path",
    "core.invariant_compatibility": "canonical.invariant_inner_product",
    "core.gate": "canonical.invariant_gate",
    "core.edge_frame_gate_activation": "canonical.invariant_gate",
    "core.grid_project": "canonical.s2_grid_projection",
    "core.grid_unproject": "canonical.s2_grid_projection",
    "core.grid_split": "canonical.s2_grid_layout",
    "core.grid_concat": "canonical.s2_grid_layout",
    "core.grid_pointwise_activation": "canonical.s2_grid_pointwise",
    "core.grid_pointwise_product": "canonical.s2_grid_pointwise",
    "core.grid_channel_linear": "canonical.s2_grid_channel_linear",
    "core.grid_dropout": "canonical.typed_dropout",
}


def _base_name(name: str) -> str:
    return name.rsplit("@", 1)[0]


def _canonical_name(name: str) -> str:
    base = _base_name(name)
    return _CANONICAL_OVERRIDES.get(base, "canonical.{}".format(base.split(".", 1)[1]))


@dataclass(frozen=True)
class CanonicalSearchSurface:
    version: str
    generatable_primitives: Tuple[str, ...]
    completion_only_primitives: Tuple[str, ...]
    context_only_primitives: Tuple[str, ...]
    generatable_motifs: Tuple[str, ...]
    context_only_motifs: Tuple[str, ...]
    canonical_families: Mapping[str, Tuple[str, ...]]

    def content_hash(self) -> str:
        payload = self.to_dict(include_hash=False)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def role_of(self, name: str) -> str:
        if name in self.generatable_primitives or name in self.generatable_motifs:
            return "generatable"
        if name in self.completion_only_primitives:
            return "completion_only"
        if name in self.context_only_primitives or name in self.context_only_motifs:
            return "context_only"
        raise KeyError(name)

    def canonical_name(self, concrete_name: str) -> str:
        for canonical, realizations in self.canonical_families.items():
            if concrete_name in realizations:
                return canonical
        raise KeyError(concrete_name)

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        generatable_canonical = {
            self.canonical_name(name)
            for name in self.generatable_primitives
        }
        payload: Dict[str, Any] = {
            "version": self.version,
            "generatable_primitives": list(self.generatable_primitives),
            "completion_only_primitives": list(self.completion_only_primitives),
            "context_only_primitives": list(self.context_only_primitives),
            "generatable_motifs": list(self.generatable_motifs),
            "context_only_motifs": list(self.context_only_motifs),
            "canonical_families": {
                name: list(values) for name, values in sorted(self.canonical_families.items())
            },
            "counts": {
                "concrete_primitive_entries": (
                    len(self.generatable_primitives)
                    + len(self.completion_only_primitives)
                    + len(self.context_only_primitives)
                ),
                "canonical_primitive_families": len(self.canonical_families),
                "generatable_concrete_primitives": len(self.generatable_primitives),
                "generatable_canonical_families": len(generatable_canonical),
                "completion_only_primitives": len(self.completion_only_primitives),
                "context_only_primitives": len(self.context_only_primitives),
                "generatable_motifs": len(self.generatable_motifs),
                "context_only_motifs": len(self.context_only_motifs),
            },
        }
        if include_hash:
            payload["content_hash"] = self.content_hash()
        return payload


def default_canonical_search_surface(
    primitives: PrimitiveRegistry,
    motifs: MotifRegistry,
) -> CanonicalSearchSurface:
    """Return the exhaustive v1 canonical search surface for the current registry.

    The function intentionally fails when a new registry entry has not been
    classified.  This prevents future implementation-only primitives from
    silently becoming LLM mutation choices.
    """

    primitive_roles = (
        _GENERATABLE_PRIMITIVES
        | _COMPLETION_ONLY_PRIMITIVES
        | _CONTEXT_ONLY_PRIMITIVES
    )
    duplicate_primitive_roles = (
        (_GENERATABLE_PRIMITIVES & _COMPLETION_ONLY_PRIMITIVES)
        | (_GENERATABLE_PRIMITIVES & _CONTEXT_ONLY_PRIMITIVES)
        | (_COMPLETION_ONLY_PRIMITIVES & _CONTEXT_ONLY_PRIMITIVES)
    )
    registered_primitives = set(primitives.names())
    primitive_missing = registered_primitives - primitive_roles
    primitive_extra = primitive_roles - registered_primitives

    motif_roles = _GENERATABLE_MOTIFS | _CONTEXT_ONLY_MOTIFS
    duplicate_motif_roles = _GENERATABLE_MOTIFS & _CONTEXT_ONLY_MOTIFS
    registered_motifs = set(motifs.names())
    motif_missing = registered_motifs - motif_roles
    motif_extra = motif_roles - registered_motifs

    if (
        duplicate_primitive_roles
        or primitive_missing
        or primitive_extra
        or duplicate_motif_roles
        or motif_missing
        or motif_extra
    ):
        raise DSLValidationError([
            Diagnostic(
                "E_SEARCH_SURFACE_001",
                "canonical search surface must classify every registry entry exactly once",
                details={
                    "duplicate_primitive_roles": sorted(duplicate_primitive_roles),
                    "missing_primitives": sorted(primitive_missing),
                    "unknown_primitives": sorted(primitive_extra),
                    "duplicate_motif_roles": sorted(duplicate_motif_roles),
                    "missing_motifs": sorted(motif_missing),
                    "unknown_motifs": sorted(motif_extra),
                },
            )
        ])

    families: Dict[str, list[str]] = {}
    for name in sorted(registered_primitives):
        families.setdefault(_canonical_name(name), []).append(name)
    return CanonicalSearchSurface(
        version=SEARCH_SURFACE_VERSION,
        generatable_primitives=tuple(sorted(_GENERATABLE_PRIMITIVES)),
        completion_only_primitives=tuple(sorted(_COMPLETION_ONLY_PRIMITIVES)),
        context_only_primitives=tuple(sorted(_CONTEXT_ONLY_PRIMITIVES)),
        generatable_motifs=tuple(sorted(_GENERATABLE_MOTIFS)),
        context_only_motifs=tuple(sorted(_CONTEXT_ONLY_MOTIFS)),
        canonical_families={
            name: tuple(values) for name, values in sorted(families.items())
        },
    )
