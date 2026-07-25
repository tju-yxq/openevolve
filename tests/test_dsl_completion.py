from equivariant_nas.dsl import (
    AvailableValue,
    Carrier,
    EquivariantType,
    Frame,
    GroupSpec,
    HoleSink,
    Irreps,
    TypedHole,
    complete_typed_hole,
    completion_repair_suggestions,
    core_registry,
    materialize_completion_patch,
    program_completion_frontier,
)
from equivariant_nas.dsl import apply_typed_patch, architecture_id
from equivariant_nas.dsl import DSLValidationError, TypeChecker
import pytest
from equivariant_nas.dsl import ArchitectureProgram, InputPort, Node, OutputPort


def value_type(group, carrier, irreps, frame=Frame()):
    return EquivariantType(group, carrier, Irreps.parse(irreps, group.family), frame)


def test_completion_finds_select_scalars_then_global_pool_for_graph_scalar():
    group = GroupSpec.so3()
    hidden = value_type(group, Carrier.NODE, "8x0+4x1+2x2")
    target = value_type(group, Carrier.GRAPH, "1x0")
    result = complete_typed_hole(
        TypedHole("readout", target, (AvailableValue("block5", hidden),), max_steps=3),
        core_registry(),
    )
    assert result.reachable
    assert result.minimum_steps == 2
    operations = {item.op for item in result.path}
    assert "core.global_pool@1" in operations
    assert operations & {"core.select_scalars@1", "core.irrep_linear@1", "core.irrep_slice@1", "core.change_multiplicity@1"}
    assert result.carrier_distance == 1
    assert result.invariant_distance == 1


def test_completion_restores_edge_frame_before_node_aggregation():
    group = GroupSpec.so3()
    edge = value_type(group, Carrier.EDGE, "2x0+2x1", Frame("edge", "bond"))
    target = value_type(group, Carrier.NODE, "2x0+2x1")
    result = complete_typed_hole(
        TypedHole("message", target, (AvailableValue("edge_message", edge),), max_steps=3),
        core_registry(),
    )
    assert result.reachable
    assert result.minimum_steps == 2
    assert [item.op for item in result.path] == ["core.from_edge_frame@1", "core.segment_mean@1"] or [
        item.op for item in result.path
    ] == ["core.from_edge_frame@1", "core.segment_sum@1"]
    assert result.frame_distance == 1
    assert result.carrier_distance == 1


def test_completion_uses_tensor_product_to_create_a_new_scalar_path():
    group = GroupSpec.so3()
    vector = value_type(group, Carrier.NODE, "3x1")
    scalar = value_type(group, Carrier.NODE, "2x0")
    result = complete_typed_hole(
        TypedHole(
            "couple",
            scalar,
            (AvailableValue("left", vector), AvailableValue("right", vector)),
            allowed_ops=("core.tensor_product@1",),
            max_steps=1,
        ),
        core_registry(),
    )
    assert result.reachable
    assert result.minimum_steps == 1
    assert result.path[0].op == "core.tensor_product@1"
    assert result.path[0].attrs["out_irreps"] == "2x0"


def test_completion_reports_unreachable_group_mismatch_without_guessing():
    source_group = GroupSpec.so3()
    target_group = GroupSpec("SO2", 2)
    source = value_type(source_group, Carrier.NODE, "1x0")
    target = value_type(target_group, Carrier.NODE, "1xm0")
    result = complete_typed_hole(
        TypedHole("cross_group", target, (AvailableValue("x", source),), max_steps=3),
        core_registry(),
    )
    assert not result.reachable
    assert result.minimum_steps is None
    assert result.representation_distance is None
    assert "no legal completion" in result.reason


def test_completion_backend_distance_counts_unsupported_steps():
    group = GroupSpec.so3()
    hidden = value_type(group, Carrier.NODE, "2x0+1x1")
    target = value_type(group, Carrier.GRAPH, "1x0")
    result = complete_typed_hole(
        TypedHole("backend", target, (AvailableValue("x", hidden),), max_steps=3),
        core_registry(),
        backend_supported_ops=(
            "core.select_scalars@1",
            "core.irrep_linear@1",
            "core.irrep_slice@1",
            "core.change_multiplicity@1",
        ),
    )
    assert result.reachable
    assert result.backend_distance == 1


def test_program_completion_frontier_is_source_location_preserving():
    group = GroupSpec.so3()
    hidden = value_type(group, Carrier.NODE, "4x0+2x1")
    graph_scalar = value_type(group, Carrier.GRAPH, "1x0")
    program = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", hidden),),
        (
            Node("block5", "core.identity", {"x": ("input:x",)}, declared_types={"out": hidden}),
            Node("pool", "core.global_pool", {"x": ("block5",)}, declared_types={"out": hidden.with_carrier(Carrier.GRAPH)}),
        ),
        (OutputPort("prediction", "pool", hidden.with_carrier(Carrier.GRAPH)),),
    )
    frontier = program_completion_frontier(program, graph_scalar, core_registry(), max_steps=3)
    assert [item["source_node"] for item in frontier] == ["block5", "pool"]
    assert all(not item["source_node"].startswith("n000") for item in frontier)
    assert frontier[0]["distance"]["reachable"] is True


def test_completion_materializes_into_an_input_port_and_type_checks():
    group = GroupSpec.so3()
    hidden = value_type(group, Carrier.NODE, "2x0+1x1")
    scalar = value_type(group, Carrier.NODE, "1x0")
    graph_scalar = scalar.with_carrier(Carrier.GRAPH)
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("hidden", hidden), InputPort("fallback", scalar)),
        (Node("pool", "core.global_pool", {"x": ("input:fallback",)}),),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    registry = core_registry()
    hole = TypedHole("pool_adapter", scalar, (AvailableValue("input:hidden", hidden),), max_steps=1)
    result = complete_typed_hole(hole, registry)
    patch = materialize_completion_patch(parent, hole, result, HoleSink.input_port("pool", "x"), registry)
    child = apply_typed_patch(parent, patch, registry)
    assert child.nodes[-1].inputs["x"] == (result.path[-1].node_id,)
    assert all(node.annotations.get("origin") == "typed_completion" for node in child.nodes[:-1])


def test_completion_materializes_at_program_output_and_prunes_disconnected_readout():
    group = GroupSpec.so3()
    hidden = value_type(group, Carrier.NODE, "2x0+1x1")
    scalar = value_type(group, Carrier.NODE, "1x0")
    graph_scalar = scalar.with_carrier(Carrier.GRAPH)
    parent = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("hidden", hidden),),
        (
            Node("block", "core.identity", {"x": ("input:hidden",)}),
            Node("old_readout", "core.select_scalars", {"x": ("block",)}, {"multiplicity": 1}),
            Node("old_pool", "core.global_pool", {"x": ("old_readout",)}),
        ),
        (OutputPort("prediction", "old_pool", graph_scalar),),
    )
    registry = core_registry()
    hole = TypedHole("new_readout", graph_scalar, (AvailableValue("block", hidden),), max_steps=2)
    result = complete_typed_hole(hole, registry)
    assert result.reachable and result.minimum_steps == 2
    patch = materialize_completion_patch(parent, hole, result, HoleSink.program_output("prediction"), registry)
    child = apply_typed_patch(parent, patch, registry)
    ids = {node.id for node in child.nodes}
    assert "old_readout" not in ids
    assert "old_pool" not in ids
    assert child.outputs[0].source == result.path[-1].node_id
    assert {action.node_id for action in result.path} <= ids


def test_frame_diagnostic_becomes_a_scope_checked_completion_suggestion():
    group = GroupSpec.so3()
    node_type = value_type(group, Carrier.NODE, "1x0+1x1")
    candidate = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", node_type),),
        (
            Node("edge", "core.edge_lift", {"x": ("input:x",)}),
            Node("rotate", "core.to_edge_frame", {"x": ("edge",)}, {"frame_id": "bond"}),
            Node("aggregate", "core.segment_sum", {"x": ("rotate",)}),
        ),
        (OutputPort("prediction", "aggregate", node_type),),
    )
    registry = core_registry()
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(registry).check(candidate)
    suggestions = completion_repair_suggestions(
        candidate,
        error.value.diagnostics,
        registry,
        authorized_scope=("aggregate",),
    )
    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion["diagnostic_code"] == "E_FRAME_004"
    assert suggestion["scope_compatible"] is True
    assert suggestion["completion"]["reachable"] is True
    assert [item["op"] for item in suggestion["completion"]["path"]] == ["core.from_edge_frame@1"]


def test_output_diagnostic_suggests_pooling_but_cannot_authorize_output_scope():
    group = GroupSpec.so3()
    node_scalar = value_type(group, Carrier.NODE, "1x0")
    graph_scalar = node_scalar.with_carrier(Carrier.GRAPH)
    candidate = ArchitectureProgram(
        "1.0.0",
        "task",
        (InputPort("x", node_scalar),),
        (Node("identity", "core.identity", {"x": ("input:x",)}),),
        (OutputPort("prediction", "identity", graph_scalar),),
    )
    registry = core_registry()
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(registry).check(candidate)
    suggestions = completion_repair_suggestions(candidate, error.value.diagnostics, registry, authorized_scope=("identity",))
    assert len(suggestions) == 1
    assert suggestions[0]["required_scope"] == "output:prediction"
    assert suggestions[0]["scope_compatible"] is False
    assert suggestions[0]["completion"]["path"][0]["op"] == "core.global_pool@1"
