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
from equivariant_nas.dsl.backends import baseline_spec
from scripts.run_dsl_evolution import _validate_root_isolated_factor_transition


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
    assert {
        item.factor_id: item.required_backend_capability
        for item in profile.enabled_factors
    } == {
        "F2.2": "exact_constructor",
        "F4.4": "exact_constructor",
        "F5.3": "exact_constructor",
        "F6.3": "exact_hybrid",
    }


@pytest.mark.parametrize(
    "factor_id,region_id,path,value,expected",
    (
        ("F4.4", "v1_attention_heads", "constructor.operator.num_heads", 8, ("operator", "num_heads", 8)),
        ("F5.3", "v1_normalization", "constructor.action.rescale_degree", True, ("action", "rescale_degree", True)),
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
    assert region.backend_capability == plan.mode
    assert region_id in plan.supported_regions


def test_radial_factor_requires_and_lowers_the_complete_preregistered_option():
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    parent_id = compiler.analyze(parent).architecture_id
    paths = (
        "constructor.operator.basis_type",
        "constructor.operator.num_basis",
        "constructor.operator.radial_hidden",
    )
    values = ("gaussian", 96, [96, 96])
    patch = TypedPatch(
        "1.0",
        parent_id,
        parent.language_version,
        {"factor_id": "F2.2", "claim": "certified radial alternative"},
        paths,
        tuple(PatchEdit("change_parameters", path, {"value": value}) for path, value in zip(paths, values)),
    )
    child = apply_typed_patch(
        parent,
        patch,
        compiler.primitives,
        expected_parent_id=parent_id,
        validate_child_with_core_registry=False,
    )
    radial = next(item for item in v1_region_registry(parent) if item.factor_id == "F2.2")
    audit = validate_region_transition(parent, child, radial)
    spec = effective_v1_spec(child)
    assert spec.operator.basis_type == "gaussian"
    assert spec.operator.num_basis == 96
    assert spec.operator.radial_hidden == (96, 96)
    assert audit["changed_parameters"] == [
        "constructor.operator.num_basis",
        "constructor.operator.radial_hidden",
    ]
    assert radial.backend_capability == compiler.plan_lowering(child).mode


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


def test_formal_coverage_rejects_a_single_edit_applied_to_an_already_mutated_parent():
    root = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    root_id = compiler.analyze(root).architecture_id
    normalization_child = apply_typed_patch(
        root,
        _patch(root, compiler, "constructor.action.rescale_degree", True, "F5.3"),
        compiler.primitives,
        expected_parent_id=root_id,
        validate_child_with_core_registry=False,
    )
    compound_child = apply_typed_patch(
        normalization_child,
        _patch(normalization_child, compiler, "constructor.operator.num_heads", 8, "F4.4"),
        compiler.primitives,
        expected_parent_id=compiler.analyze(normalization_child).architecture_id,
        validate_child_with_core_registry=False,
    )
    with pytest.raises(Exception, match="outside the selected factor"):
        _validate_root_isolated_factor_transition(root, compound_child, "F4.4")


def test_constructor_factor_rejects_cross_product_of_individually_allowed_values():
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    parent_id = compiler.analyze(parent).architecture_id
    patch = TypedPatch(
        "1.0",
        parent_id,
        parent.language_version,
        {"factor_id": "F2.2", "claim": "invalid cross-product option"},
        ("constructor.operator.num_basis",),
        (PatchEdit("change_parameters", "constructor.operator.num_basis", {"value": 96}),),
    )
    child = apply_typed_patch(
        parent,
        patch,
        compiler.primitives,
        expected_parent_id=parent_id,
        validate_child_with_core_registry=False,
    )
    radial = next(item for item in v1_region_registry(parent) if item.factor_id == "F2.2")
    with pytest.raises(Exception, match="preregistered option"):
        validate_region_transition(parent, child, radial)


def test_formal_capability_excludes_a100_rejected_options():
    profile = equiformer_v1_capability_profile()
    radial = profile.factor("F2.2")
    normalization = profile.factor("F5.3")
    assert all(item["basis_type"] != "bessel" for item in radial.alternative_options)
    assert all(item["norm_layer"] != "instance" for item in normalization.alternative_options)
    assert all(item["norm_layer"] != "graph" for item in normalization.alternative_options)
    assert {tuple(item["radial_hidden"]) for item in radial.alternative_options} >= {(64, 64), (96, 96)}
    assert {item["norm_layer"] for item in normalization.alternative_options} >= {"layer", "fast_layer"}
    assert radial.rejected_options and normalization.rejected_options


def test_uncertified_generic_graph_is_not_formally_trainable():
    parent = _parent()
    compiler = Compiler(core_registry(), reference_motif_registry())
    generic = replace(parent, annotations={}, parameters={})
    assert compiler.plan_lowering(generic).mode == "experimental_node_graph"
