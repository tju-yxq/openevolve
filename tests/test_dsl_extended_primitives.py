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
    ResourceContract,
    TaskContract,
    TypeChecker,
    core_registry,
    expand_motifs,
    reference_motif_registry,
)


def eq_type(group, carrier, irreps, measure="dimensionless"):
    return EquivariantType(group, carrier, Irreps.parse(irreps, group.family), measure=measure)


def test_geometry_pipeline_infers_relative_vector_distance_and_harmonics():
    group = GroupSpec.o3()
    position = eq_type(group, Carrier.NODE, "1x1o", "length")
    sh = eq_type(group, Carrier.EDGE, "1x0e+1x1o+1x2e")
    program = ArchitectureProgram(
        "1.0.0",
        "geometry",
        (InputPort("positions", position),),
        (
            Node("relative", "core.relative_position", {"source": ("input:positions",), "target": ("input:positions",)}),
            Node("sh", "core.spherical_harmonics", {"direction": ("relative",)}, {"lmax": 2}),
        ),
        (OutputPort("sh", "sh", sh),),
    )
    TypeChecker(core_registry()).check(program)


def test_gate_rejects_non_scalar_gate_features():
    group = GroupSpec.o3()
    vector = eq_type(group, Carrier.NODE, "1x1o")
    program = ArchitectureProgram(
        "1.0.0",
        "gate",
        (InputPort("gates", vector), InputPort("value", vector)),
        (Node("gate", "core.gate", {"gates": ("input:gates",), "value": ("input:value",)}),),
        (OutputPort("out", "gate", vector),),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(program)
    assert error.value.diagnostics[0].code == "E_GATE_002"


def test_v2_reference_motif_exposes_balanced_edge_frame_path():
    group = GroupSpec.o3()
    hidden = eq_type(group, Carrier.NODE, "4x0e+2x1o+1x2e")
    source = ArchitectureProgram(
        "1.0.0",
        "v2",
        (InputPort("x", hidden),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:x",)},
                {"hidden_irreps": str(hidden.irreps), "frame_id": "v2_edge"},
            ),
        ),
        (OutputPort("out", "v2", hidden),),
    )
    expanded = expand_motifs(source, reference_motif_registry())
    result = TypeChecker(core_registry()).check(expanded)
    assert any(node.op == "core.so2_convolution" for node in expanded.nodes)
    assert not result.open_obligations


def test_task_contract_blocks_test_evidence():
    group = GroupSpec.o3()
    output = eq_type(group, Carrier.GRAPH, "1x0e")
    with pytest.raises(DSLValidationError) as error:
        TaskContract(
            "bad",
            group,
            output,
            "protocol",
            ResourceContract(1000, 1000000, 1.5),
            allowed_evidence_splits=("train", "validation", "test"),
        )
    assert error.value.diagnostics[0].code == "E_TASK_002"


def test_dropout_probability_is_checked_statically():
    group = GroupSpec.o3()
    scalar = eq_type(group, Carrier.NODE, "1x0e")
    program = ArchitectureProgram(
        "1.0.0",
        "dropout",
        (InputPort("x", scalar),),
        (Node("drop", "core.invariant_dropout", {"x": ("input:x",)}, {"p": 1.0}),),
        (OutputPort("out", "drop", scalar),),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(program)
    assert error.value.diagnostics[0].code == "E_ATTR_003"
