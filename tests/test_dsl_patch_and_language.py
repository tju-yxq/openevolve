from dataclasses import replace

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    DSLValidationError,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    LanguageVersion,
    Node,
    OutputPort,
    PatchEdit,
    TypedPatch,
    apply_typed_patch,
    architecture_id,
    core_registry,
    reference_motif_registry,
    select_active_vocabulary,
)


def scalar_program():
    value_type = EquivariantType(GroupSpec.o3(), Carrier.NODE, Irreps.parse("2x0e", "O3"))
    return ArchitectureProgram(
        "1.0.0",
        "contract",
        (InputPort("x", value_type),),
        (Node("project", "core.irrep_linear", {"x": ("input:x",)}, {"out_irreps": "2x0e"}),),
        (OutputPort("y", "project", value_type),),
    )


def make_patch(parent, edit, scope=("project",)):
    registry = core_registry()
    return TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        {"claim": "test type-preserving structural transaction"},
        scope,
        (edit,),
    )


def test_typed_patch_commits_only_a_valid_child():
    parent = scalar_program()
    patch = make_patch(parent, PatchEdit("change_attrs", "project", {"attrs": {"out_irreps": "1x0e"}}))
    with pytest.raises(DSLValidationError):
        apply_typed_patch(parent, patch, core_registry())
    assert parent.nodes[0].attrs["out_irreps"] == "2x0e"


def test_typed_patch_cannot_escape_scope():
    parent = scalar_program()
    patch = make_patch(parent, PatchEdit("change_attrs", "project", {"attrs": {"out_irreps": "2x0e"}}), scope=("other",))
    with pytest.raises(DSLValidationError) as error:
        apply_typed_patch(parent, patch, core_registry())
    assert error.value.diagnostics[0].code == "E_PATCH_007"


def test_active_vocabulary_records_group_exclusions():
    primitives = core_registry()
    motifs = reference_motif_registry()
    language = LanguageVersion(
        "1.0.0",
        "",
        primitives.names(),
        motifs.names(),
        "2026-07-24T00:00:00Z",
        {},
    )
    decision = select_active_vocabulary(language, GroupSpec("SO2", 2), primitives, motifs)
    assert "core.identity@1" in decision.visible
    assert all(decision.excluded[name] == "incompatible_group" for name in motifs.names())


def test_frozen_language_binds_registry_contents_not_only_operation_names():
    primitives = core_registry()
    motifs = reference_motif_registry()
    language = LanguageVersion.from_registries("1.0.0", "", primitives, motifs, "now", {})
    assert set(language.primitive_hashes) == set(primitives.names())
    assert set(language.motif_hashes) == set(motifs.names())
    drifted = replace(
        language,
        primitive_hashes={**dict(language.primitive_hashes), "core.identity@1": "different-content"},
    )
    assert drifted.registry_hash() != language.registry_hash()
    with pytest.raises(DSLValidationError) as error:
        select_active_vocabulary(drifted, GroupSpec.o3(), primitives, motifs)
    assert error.value.diagnostics[0].code == "E_LANGUAGE_001"
