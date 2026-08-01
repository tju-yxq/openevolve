import json

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
    ResourceContract,
    TaskContract,
    TypedPatch,
    apply_typed_patch,
    architecture_id,
    core_registry,
    default_canonical_search_surface,
    reference_motif_registry,
    select_active_vocabulary,
    synthesizer_prompt,
)


def _objects():
    group = GroupSpec.o3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0e", "O3"))
    program = ArchitectureProgram(
        "2.2.0",
        "search-surface-test",
        (InputPort("x", value_type),),
        (Node("keep", "core.identity@1", {"x": ("input:x",)}),),
        (OutputPort("out", "keep", value_type),),
    )
    task = TaskContract(
        "search-surface-test",
        group,
        value_type,
        "fixed",
        ResourceContract(1_000_000, 1_000_000_000, 1.0),
    )
    primitives = core_registry()
    motifs = reference_motif_registry()
    language = LanguageVersion.from_registries("2.2.0", "", primitives, motifs, "now", {})
    surface = default_canonical_search_surface(primitives, motifs)
    vocabulary = select_active_vocabulary(
        language,
        group,
        primitives,
        motifs,
        search_surface=surface,
    )
    return program, task, primitives, motifs, surface, vocabulary


def test_canonical_search_surface_exhaustively_classifies_the_registry():
    _program, _task, primitives, motifs, surface, _vocabulary = _objects()
    payload = surface.to_dict()
    assert len(primitives.names()) == 103
    assert payload["counts"] == {
        "concrete_primitive_entries": 103,
        "canonical_primitive_families": 48,
        "generatable_concrete_primitives": 47,
        "generatable_canonical_families": 26,
        "completion_only_primitives": 29,
        "context_only_primitives": 27,
        "generatable_motifs": 2,
        "context_only_motifs": 6,
    }
    classified = (
        set(surface.generatable_primitives)
        | set(surface.completion_only_primitives)
        | set(surface.context_only_primitives)
    )
    assert classified == set(primitives.names())
    assert set(surface.generatable_motifs) | set(surface.context_only_motifs) == set(motifs.names())
    assert surface.canonical_name("core.tensor_product@1") == "canonical.tensor_product"
    assert surface.canonical_name("core.tensor_product@5") == "canonical.tensor_product"
    assert surface.canonical_name("core.invariant_scale@2") == "canonical.typed_multiply"
    assert surface.canonical_name("core.degreewise_invariant_scale@1") == "canonical.typed_multiply"
    assert surface.canonical_name("core.edge_frame_gate_activation@1") == "canonical.invariant_gate"
    assert surface.role_of("core.squeeze_unit_axis@1") == "completion_only"
    assert surface.canonical_name("core.squeeze_unit_axis@1") == "canonical.squeeze_unit_axis"


def test_active_vocabulary_separates_generation_completion_and_context_roles():
    _program, _task, _primitives, _motifs, surface, vocabulary = _objects()
    assert vocabulary.search_surface_version == surface.version
    assert vocabulary.search_surface_hash == surface.content_hash()
    assert "core.identity@1" not in vocabulary.visible
    assert "core.identity@1" in vocabulary.context_only
    assert "core.endpoint_gather@2" not in vocabulary.visible
    assert "core.endpoint_gather@2" in vocabulary.completion_only
    assert "core.tensor_product@2" in vocabulary.visible
    assert set(vocabulary.canonical_families["canonical.tensor_product"]) == {
        "core.tensor_product@1",
        "core.tensor_product@2",
        "core.tensor_product@3",
        "core.tensor_product@4",
        "core.tensor_product@5",
    }


def test_patch_execution_rejects_new_context_only_ops_but_preserves_imported_ones():
    program, _task, primitives, _motifs, _surface, vocabulary = _objects()
    parent_id = architecture_id(program, primitives)
    inserted = Node("new_identity", "core.identity@1", {"x": ("input:x",)})
    rejected = TypedPatch(
        "1.0",
        parent_id,
        program.language_version,
        {"claim": "attempt to introduce a compatibility-only alias"},
        ("keep",),
        (PatchEdit("insert_before", "keep", {"node": inserted.to_dict()}),),
    )
    with pytest.raises(DSLValidationError) as error:
        apply_typed_patch(program, rejected, primitives, allowed_new_ops=vocabulary.visible)
    assert error.value.diagnostics[0].code == "E_PATCH_018"

    preserved = TypedPatch(
        "1.0",
        parent_id,
        program.language_version,
        {"claim": "preserve an imported compatibility node"},
        ("keep",),
        (PatchEdit("replace_node", "keep", {"node": program.nodes[0].to_dict()}),),
    )
    child = apply_typed_patch(program, preserved, primitives, allowed_new_ops=vocabulary.visible)
    assert child.nodes[0].op == "core.identity@1"


def test_llm_prompt_groups_concrete_realizations_by_canonical_family():
    program, task, primitives, motifs, surface, vocabulary = _objects()
    parent_id = architecture_id(program, primitives, task_contract_hash=task.content_hash())
    plan = {
        "claim": "insert one typed transformation",
        "scope": ["keep"],
        "abstract_goals": ["change the local representation flow"],
        "evidence_refs": [],
        "uncertainty": "untrained",
        "risk": "type mismatch",
    }
    prompt = synthesizer_prompt(task, program, plan, vocabulary, parent_id, primitives, motifs)
    payload = json.loads(prompt["user"])
    assert payload["search_surface_version"] == surface.version
    assert "canonical.tensor_product" in payload["visible_vocabulary"]
    assert "core.tensor_product@1" not in payload["visible_vocabulary"]
    tensor_product = next(
        item for item in payload["vocabulary_contracts"]
        if item["name"] == "canonical.tensor_product"
    )
    assert {item["name"] for item in tensor_product["realizations"]} == {
        "core.tensor_product@1",
        "core.tensor_product@2",
        "core.tensor_product@3",
        "core.tensor_product@4",
        "core.tensor_product@5",
    }
    assert any(item["name"] == "core.identity@1" for item in payload["parent_operator_contracts"])
    assert "core.endpoint_gather@2" in payload["completion_only_vocabulary"]
