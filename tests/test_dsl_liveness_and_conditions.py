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
    Node,
    OutputPort,
    PatchEdit,
    TypedPatch,
    TypeChecker,
    apply_typed_patch,
    architecture_id,
    core_registry,
    parse_patch_response,
)


def base_program():
    group = GroupSpec.so3()
    node = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph = node.with_carrier(Carrier.GRAPH)
    return ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", node),),
        (
            Node("live", "core.identity", {"x": ("input:x",)}),
            Node("pool", "core.global_pool", {"x": ("live",)}),
        ),
        (OutputPort("prediction", "pool", graph),),
    )


def test_type_checker_rejects_well_typed_but_dead_subgraphs():
    program = base_program()
    dead = Node("unused", "core.identity", {"x": ("live",)})
    program = ArchitectureProgram(
        program.language_version,
        program.task_contract,
        program.inputs,
        program.nodes + (dead,),
        program.outputs,
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(program)
    assert error.value.diagnostics[0].code == "E_GRAPH_002"
    assert error.value.diagnostics[0].details["dead_nodes"] == ["unused"]


def test_patch_executes_structural_preconditions_and_postconditions():
    parent = base_program()
    registry = core_registry()
    patch = TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        {"claim": "rewire through an adapter"},
        ("pool",),
        (
            PatchEdit(
                "insert_before",
                "pool",
                {"node": Node("adapter", "core.identity", {"x": ("live",)}).to_dict()},
            ),
            PatchEdit("rewire_port", "pool", {"port": "x", "references": ["adapter"]}),
        ),
        ({"kind": "node_input_equals", "node_id": "pool", "port": "x", "references": ["live"]},),
        ({"kind": "node_input_equals", "node_id": "pool", "port": "x", "references": ["adapter"]},),
    )
    child = apply_typed_patch(parent, patch, registry)
    assert child.nodes[-1].inputs["x"] == ("adapter",)


def test_failed_executable_postcondition_rolls_back_patch():
    parent = base_program()
    registry = core_registry()
    patch = TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        {},
        ("pool",),
        (PatchEdit("change_attrs", "pool", {"attrs": {}}),),
        (),
        ({"kind": "output_source_is", "output": "prediction", "reference": "other"},),
    )
    with pytest.raises(DSLValidationError) as error:
        apply_typed_patch(parent, patch, registry)
    assert error.value.diagnostics[0].code == "E_PATCH_012"
    assert parent.outputs[0].source == "pool"


def test_patch_parser_rejects_natural_language_conditions():
    payload = {
        "patch_version": "1.0",
        "parent_architecture_id": "parent",
        "language_version": "1.0.0",
        "hypothesis": {},
        "scope": ["pool"],
        "edits": [{"kind": "change_attrs", "target": "pool", "payload": {"attrs": {}}}],
        "postconditions": [{"condition": "program output is the new node"}],
    }
    with pytest.raises(DSLValidationError) as error:
        parse_patch_response(json.dumps(payload))
    assert error.value.diagnostics[0].code == "E_PATCH_011"


def test_rewire_output_requires_its_own_explicit_scope_capability():
    parent = base_program()
    registry = core_registry()
    patch = TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        {},
        ("pool",),
        (PatchEdit("rewire_output", "output:prediction", {"reference": "pool:out"}),),
    )
    with pytest.raises(DSLValidationError) as error:
        apply_typed_patch(parent, patch, registry)
    assert error.value.diagnostics[0].code == "E_PATCH_007"

    authorized = TypedPatch(
        patch.patch_version,
        patch.parent_architecture_id,
        patch.language_version,
        patch.hypothesis,
        ("output:prediction",),
        patch.edits,
    )
    child = apply_typed_patch(parent, authorized, registry)
    assert child.outputs[0].source == "pool:out"


def test_patch_can_chain_from_an_inserted_node_without_widening_existing_scope():
    parent = base_program()
    registry = core_registry()
    graph_type = parent.outputs[0].expected_type
    first = Node("adapter_a", "core.identity", {"x": ("pool",)}, declared_types={"out": graph_type})
    second = Node("adapter_b", "core.identity", {"x": ("adapter_a",)}, declared_types={"out": graph_type})
    patch = TypedPatch(
        "1.0",
        architecture_id(parent, registry),
        parent.language_version,
        {},
        ("pool", "output:prediction"),
        (
            PatchEdit("insert_after", "pool", {"node": first.to_dict()}),
            PatchEdit("insert_after", "adapter_a", {"node": second.to_dict()}),
            PatchEdit("rewire_output", "output:prediction", {"reference": "adapter_b"}),
        ),
    )
    child = apply_typed_patch(parent, patch, registry)
    assert child.outputs[0].source == "adapter_b"
    assert [node.id for node in child.nodes][-2:] == ["adapter_a", "adapter_b"]
