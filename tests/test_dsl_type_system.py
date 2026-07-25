from dataclasses import replace

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    DSLValidationError,
    EquivariantType,
    Frame,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)


def value_type(irreps="2x0e+1x1o", carrier=Carrier.NODE, frame=Frame()):
    return EquivariantType(GroupSpec.o3(), carrier, Irreps.parse(irreps, "O3"), frame)


def program(nodes, output_source, output_type, inputs=None):
    return ArchitectureProgram(
        "1.0.0",
        "test_contract",
        tuple(inputs or (InputPort("x", value_type()),)),
        tuple(nodes),
        (OutputPort("y", output_source, output_type),),
    )


def diagnostic_code(error):
    return error.value.diagnostics[0].code


def test_valid_frame_round_trip_and_aggregation_discharge_obligations():
    output = value_type(carrier=Carrier.NODE)
    candidate = program(
        (
            Node("edge", "core.edge_lift", {"x": ("input:x",)}),
            Node("to", "core.to_edge_frame", {"x": ("edge",)}, {"frame_id": "edge_f"}),
            Node("back", "core.from_edge_frame", {"x": ("to",)}, {"frame_id": "edge_f"}),
            Node("sum", "core.segment_sum", {"x": ("back",)}),
        ),
        "sum",
        output,
    )
    result = TypeChecker(core_registry()).check(candidate)
    assert not result.open_obligations
    assert {item.kind.value for item in result.obligations} >= {
        "FRAME_BALANCE",
        "PERMUTATION_SAFE_AGGREGATION",
        "OUTPUT_CONTRACT_MATCH",
    }


def test_edge_frame_cannot_be_aggregated_directly():
    candidate = program(
        (
            Node("edge", "core.edge_lift", {"x": ("input:x",)}),
            Node("to", "core.to_edge_frame", {"x": ("edge",)}, {"frame_id": "edge_f"}),
            Node("sum", "core.segment_sum", {"x": ("to",)}),
        ),
        "sum",
        value_type(carrier=Carrier.NODE),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(candidate)
    assert diagnostic_code(error) == "E_FRAME_004"


def test_elementwise_activation_rejects_non_scalar_irreps():
    candidate = program(
        (Node("act", "core.scalar_activation", {"x": ("input:x",)}),),
        "act",
        value_type(),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(candidate)
    assert diagnostic_code(error) == "E_NONLINEAR_001"


def test_tensor_product_rejects_impossible_output_irrep():
    scalar = value_type("1x0e", carrier=Carrier.EDGE)
    vector = value_type("1x1o", carrier=Carrier.EDGE)
    candidate = program(
        (Node("tp", "core.tensor_product", {"left": ("input:left",), "right": ("input:right",)}, {"out_irreps": "1x3o"}),),
        "tp",
        value_type("1x3o", carrier=Carrier.EDGE),
        (InputPort("left", scalar), InputPort("right", vector)),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(candidate)
    assert diagnostic_code(error) == "E_IRREP_012"


def test_graph_cycles_are_rejected_before_inference():
    candidate = program(
        (
            Node("a", "core.identity", {"x": ("b",)}),
            Node("b", "core.identity", {"x": ("a",)}),
        ),
        "a",
        value_type(),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(candidate)
    assert diagnostic_code(error) == "E_GRAPH_001"


def test_declared_type_cannot_override_inference():
    wrong = value_type("1x0e")
    candidate = program(
        (Node("id", "core.identity", {"x": ("input:x",)}, declared_types={"out": wrong}),),
        "id",
        value_type(),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(candidate)
    assert diagnostic_code(error) == "E_TYPE_005"
