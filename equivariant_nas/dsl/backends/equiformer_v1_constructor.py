"""Exact lowering of certified DSL constructor parameters to Equiformer V1."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Tuple

from .equiformer_v1_spec import ArchitectureSpec
from ..diagnostics import DSLValidationError, Diagnostic
from ..factors import equiformer_v1_capability_profile


def baseline_constructor_parameters(spec: ArchitectureSpec) -> Dict[str, Any]:
    return {
        "constructor.operator.basis_type": spec.operator.basis_type,
        "constructor.operator.num_basis": spec.operator.num_basis,
        "constructor.operator.radial_hidden": list(spec.operator.radial_hidden),
        "constructor.operator.num_heads": spec.operator.num_heads,
        "constructor.action.norm_layer": spec.action.norm_layer,
        "constructor.action.rescale_degree": spec.action.rescale_degree,
    }


def _allowed_paths() -> Tuple[str, ...]:
    profile = equiformer_v1_capability_profile()
    return tuple(sorted(path for factor in profile.enabled_factors for path in factor.parameter_paths))


def effective_v1_spec(program) -> ArchitectureSpec:
    if program.annotations.get("reference_backend") != "equiformer_v1":
        raise DSLValidationError([Diagnostic("E_V1_CONSTRUCTOR_001", "program is not an imported Equiformer V1 architecture")])
    base = ArchitectureSpec.from_dict(program.annotations["equiformer_v1_spec"])
    allowed = set(_allowed_paths())
    unknown = sorted(key for key in program.parameters if key.startswith("constructor.") and key not in allowed)
    if unknown:
        raise DSLValidationError([
            Diagnostic("E_V1_CONSTRUCTOR_002", "program contains uncertified constructor parameters", details={"parameters": unknown})
        ])
    values = baseline_constructor_parameters(base)
    values.update({key: value for key, value in program.parameters.items() if key in allowed})
    operator = replace(
        base.operator,
        basis_type=str(values["constructor.operator.basis_type"]),
        num_basis=int(values["constructor.operator.num_basis"]),
        radial_hidden=tuple(int(item) for item in values["constructor.operator.radial_hidden"]),
        num_heads=int(values["constructor.operator.num_heads"]),
    )
    action = replace(
        base.action,
        norm_layer=str(values["constructor.action.norm_layer"]),
        rescale_degree=bool(values["constructor.action.rescale_degree"]),
    )
    try:
        return replace(base, operator=operator, action=action).validate()
    except ValueError as exc:
        raise DSLValidationError([
            Diagnostic("E_V1_CONSTRUCTOR_003", "constructor factor values violate the official V1 grammar", actual=str(exc))
        ])


def changed_constructor_parameters(program) -> Tuple[str, ...]:
    base = ArchitectureSpec.from_dict(program.annotations["equiformer_v1_spec"])
    baseline = baseline_constructor_parameters(base)
    return tuple(sorted(
        key for key in _allowed_paths()
        if program.parameters.get(key, baseline[key]) != baseline[key]
    ))


def restored_constructor_parameters(program):
    base = ArchitectureSpec.from_dict(program.annotations["equiformer_v1_spec"])
    parameters = dict(program.parameters)
    parameters.update(baseline_constructor_parameters(base))
    return replace(program, parameters=parameters)
