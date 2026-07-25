from dataclasses import replace

import pytest

from equivariant_nas.dsl import (
    Compiler,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    core_registry,
    equiformer_v1_capability_profile,
    import_equiformer_v1,
    reference_motif_registry,
    validate_region_transition,
    validate_unique_factor_ownership,
    v1_region_registry,
)
from equivariant_nas.dsl.backends.equiformer_v1_constructor import effective_v1_spec
from equivariant_nas.spec import baseline_spec


def _parent():
    return import_equiformer_v1(baseline_spec())


def _patch(parent, compiler, path, value, factor_id):
    return TypedPatch(
        "1.0",
        compiler.analyze(parent).architecture_id,
        parent.language_version,
        {"factor_id": factor_id, "claim": "formal V1 factor mutation"},
        (path,),
        (PatchEdit("change_parameters", path, {"value": value}),),
    )


def test_formal_v1_factor_ownership_is_unique():
    profile = equiformer_v1_capability_profile()
    validate_unique_factor_ownership(profile.enabled_factors)
    assert {item.factor_id for item in profile.enabled_factors} == {"F2.2", "F4.4", "F5.3", "F6.3"}


@pytest.mark.parametrize(
    "factor_id,region_id,path,value,expected",
    (
        ("F2.2", "v1_radial_encoding", "constructor.operator.basis_type", "bessel", ("operator", "basis_type", "bessel")),
        ("F4.4", "v1_attention_heads", "constructor.operator.num_heads", 8, ("operator", "num_heads", 8)),
        ("F5.3", "v1_normalization", "constructor.action.norm_layer", "instance", ("action", "norm_layer", "instance")),
    ),
)
def test_constructor_factor_patch_has_exact_official_lowering(factor_id, region_id, path, value, expected):
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    parent_id = compiler.analyze(parent).architecture_id
    child = apply_typed_patch(
        parent,
        _patch(parent, compiler, path, value, factor_id),
        compiler.primitives,
        expected_parent_id=parent_id,
        validate_child_with_core_registry=False,
    )
    region = next(item for item in v1_region_registry(parent) if item.region_id == region_id)
    audit = validate_region_transition(parent, child, region)
    plan = compiler.plan_lowering(child)
    spec = effective_v1_spec(child)
    section, field, expected_value = expected
    assert getattr(getattr(spec, section), field) == expected_value
    assert audit["factor_id"] == factor_id
    assert audit["changed_parameters"] == [path]
    assert plan.mode == "exact_constructor"
    assert region_id in plan.supported_regions


def test_factor_patch_cannot_change_an_unowned_constructor_parameter():
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    patch = _patch(parent, compiler, "constructor.operator.num_heads", 8, "F4.4")
    child = apply_typed_patch(
        parent,
        patch,
        compiler.primitives,
        expected_parent_id=compiler.analyze(parent).architecture_id,
        validate_child_with_core_registry=False,
    )
    radial = next(item for item in v1_region_registry(parent) if item.factor_id == "F2.2")
    with pytest.raises(Exception):
        validate_region_transition(parent, child, radial)


def test_uncertified_generic_graph_is_not_formally_trainable():
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    generic = replace(parent, annotations={}, parameters={})
    assert compiler.plan_lowering(generic).mode == "experimental_node_graph"
