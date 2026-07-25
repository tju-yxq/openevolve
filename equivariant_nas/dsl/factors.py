"""Capability-gated functional factors for the formal V1 search slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .diagnostics import DSLValidationError, Diagnostic


@dataclass(frozen=True)
class LeafFactorDefinition:
    factor_id: str
    family_id: str
    scientific_role: str
    region_id: str
    parameter_paths: Tuple[str, ...] = ()
    required_backend_capability: str = "analysis_only"
    identity_option: Mapping[str, Any] = None
    alternative_options: Tuple[Mapping[str, Any], ...] = ()
    rejected_options: Tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "family_id": self.family_id,
            "scientific_role": self.scientific_role,
            "region_id": self.region_id,
            "parameter_paths": list(self.parameter_paths),
            "required_backend_capability": self.required_backend_capability,
            "identity_option": dict(self.identity_option or {}),
            "alternative_options": [dict(item) for item in self.alternative_options],
            "rejected_options": [dict(item) for item in self.rejected_options],
        }


@dataclass(frozen=True)
class CapabilityProfile:
    architecture_family: str
    backend_semantics_version: str
    enabled_factors: Tuple[LeafFactorDefinition, ...]
    disabled_factors: Mapping[str, str]

    def factor(self, factor_id: str) -> LeafFactorDefinition:
        matches = [item for item in self.enabled_factors if item.factor_id == factor_id]
        if len(matches) != 1:
            raise DSLValidationError([
                Diagnostic("E_FACTOR_001", "factor is not uniquely enabled by the capability profile", actual=factor_id)
            ])
        return matches[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_family": self.architecture_family,
            "backend_semantics_version": self.backend_semantics_version,
            "enabled_factors": [item.to_dict() for item in self.enabled_factors],
            "disabled_factors": dict(self.disabled_factors),
        }


def equiformer_v1_capability_profile() -> CapabilityProfile:
    """Return the deliberately small, numerically faithful formal-V1 slice."""

    factors = (
        LeafFactorDefinition(
            "F2.2",
            "F2",
            "Radial encoding family and capacity.",
            "v1_radial_encoding",
            (
                "constructor.operator.basis_type",
                "constructor.operator.num_basis",
                "constructor.operator.radial_hidden",
            ),
            "exact_v1_constructor",
            {"basis_type": "gaussian", "num_basis": 128, "radial_hidden": [64, 64]},
            (
                {"basis_type": "gaussian", "num_basis": 96, "radial_hidden": [96, 96]},
            ),
            (
                {
                    "basis_type": "bessel",
                    "num_basis": 64,
                    "radial_hidden": [64, 64],
                    "rejection_reason": "A100 formal-v1 symmetry gate failed at seed 201",
                },
            ),
        ),
        LeafFactorDefinition(
            "F4.4",
            "F4",
            "Multi-head organization inside invariant attention routing.",
            "v1_attention_heads",
            ("constructor.operator.num_heads",),
            "exact_v1_constructor",
            {"num_heads": 4},
            ({"num_heads": 2}, {"num_heads": 8}),
        ),
        LeafFactorDefinition(
            "F5.3",
            "F5",
            "Equivariant normalization and degree-rescaling stability.",
            "v1_normalization",
            ("constructor.action.norm_layer", "constructor.action.rescale_degree"),
            "exact_v1_constructor",
            {"norm_layer": "layer", "rescale_degree": False},
            (
                {"norm_layer": "layer", "rescale_degree": True},
            ),
            (
                {
                    "norm_layer": "instance",
                    "rescale_degree": False,
                    "rejection_reason": "A100 formal-v1 symmetry gate failed at seed 201",
                },
            ),
        ),
        LeafFactorDefinition(
            "F6.3",
            "F6",
            "Multi-level invariant readout and output construction.",
            "v1_readout",
            (),
            "exact_hybrid",
            {"readout": "terminal"},
            ({"readout": "multilevel"},),
        ),
    )
    return CapabilityProfile(
        "EquiformerV1",
        "equiformer-v1-formal-dsl-v1",
        factors,
        {
            "F3.3": "SO(2) kernel fusion remains experimental and is not admitted to formal V1 ranking.",
            "F5.2": "V2 S2 activations are not yet constructor-faithful V1 regions.",
            "G1": "Global representation migration is disabled in the first formal experiment.",
            "G2": "Macro topology and resource migration are disabled in the first formal experiment.",
        },
    )


def factor_by_region(profile: CapabilityProfile, region_id: str) -> LeafFactorDefinition:
    matches = [item for item in profile.enabled_factors if item.region_id == region_id]
    if len(matches) != 1:
        raise DSLValidationError([
            Diagnostic("E_FACTOR_002", "region does not have unique factor ownership", actual=region_id)
        ])
    return matches[0]


def validate_unique_factor_ownership(factors: Sequence[LeafFactorDefinition]) -> None:
    owners = {}
    for factor in factors:
        for path in factor.parameter_paths:
            if path in owners:
                raise DSLValidationError([
                    Diagnostic(
                        "E_FACTOR_003",
                        "parameter path is owned by multiple leaf factors",
                        actual=path,
                        details={"owners": [owners[path], factor.factor_id]},
                    )
                ])
            owners[path] = factor.factor_id
