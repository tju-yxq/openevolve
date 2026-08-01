from dataclasses import replace

import pytest

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)
from equivariant_nas.dsl.canonicalize import architecture_id
from equivariant_nas.dsl.diagnostics import DSLValidationError


def _scalar_type(*, carrier=Carrier.NODE, dtype="float32", measure="dimensionless"):
    group = GroupSpec.so3()
    return EquivariantType(
        group,
        carrier,
        Irreps.parse("1x0", "SO3"),
        dtype=dtype,
        measure=measure,
    )


def _single_node_program(node, input_ports, output_type):
    return ArchitectureProgram(
        "1.0.0",
        "contract_hardening",
        tuple(input_ports),
        (node,),
        (OutputPort("out", node.id, output_type),),
    )


def test_unknown_primitive_attributes_are_rejected_before_lowering():
    value_type = _scalar_type()
    program = _single_node_program(
        Node("identity", "core.identity", {"x": ("input:x",)}, {"unused": True}),
        (InputPort("x", value_type),),
        value_type,
    )

    with pytest.raises(DSLValidationError, match="primitive attributes do not match") as exc:
        TypeChecker(core_registry()).check(program)

    assert exc.value.diagnostics[0].code == "E_ATTR_005"
    assert exc.value.diagnostics[0].details["unknown"] == ["unused"]


def test_legacy_activation_alias_has_the_same_canonical_architecture_identity():
    value_type = _scalar_type()
    canonical = _single_node_program(
        Node("activation", "core.scalar_activation", {"x": ("input:x",)}, {"activation": "tanh"}),
        (InputPort("x", value_type),),
        value_type,
    )
    legacy = replace(
        canonical,
        nodes=(Node("activation", "core.scalar_activation", {"x": ("input:x",)}, {"function": "tanh"}),),
    )
    registry = core_registry()

    TypeChecker(registry).check(legacy)
    assert architecture_id(canonical, registry) == architecture_id(legacy, registry)


def test_alias_and_canonical_attribute_cannot_be_supplied_together():
    value_type = _scalar_type()
    program = _single_node_program(
        Node(
            "activation",
            "core.scalar_activation",
            {"x": ("input:x",)},
            {"activation": "relu", "function": "tanh"},
        ),
        (InputPort("x", value_type),),
        value_type,
    )

    with pytest.raises(DSLValidationError) as exc:
        TypeChecker(core_registry()).check(program)

    assert exc.value.diagnostics[0].code == "E_ATTR_004"


def test_residual_add_rejects_dtype_mismatch():
    left = _scalar_type(dtype="float32")
    right = _scalar_type(dtype="float64")
    program = _single_node_program(
        Node("residual", "core.residual_add", {"left": ("input:left",), "right": ("input:right",)}),
        (InputPort("left", left), InputPort("right", right)),
        left,
    )

    with pytest.raises(DSLValidationError) as exc:
        TypeChecker(core_registry()).check(program)

    assert exc.value.diagnostics[0].code == "E_TYPE_004"


def test_radial_basis_outputs_dimensionless_features():
    distance = _scalar_type(carrier=Carrier.EDGE, measure="length")
    basis = EquivariantType(
        distance.group,
        Carrier.EDGE,
        Irreps.parse("4x0", "SO3"),
        measure="dimensionless",
    )
    program = _single_node_program(
        Node("basis", "core.radial_basis", {"distance": ("input:distance",)}, {"num_basis": 4}),
        (InputPort("distance", distance),),
        basis,
    )

    inference = TypeChecker(core_registry()).check(program)
    assert inference.value_types["basis"].measure == "dimensionless"


def test_segment_softmax_rejects_dimensional_logits():
    logits = _scalar_type(carrier=Carrier.EDGE, measure="energy")
    program = _single_node_program(
        Node("weights", "core.segment_softmax", {"logits": ("input:logits",)}),
        (InputPort("logits", logits),),
        logits,
    )

    with pytest.raises(DSLValidationError) as exc:
        TypeChecker(core_registry()).check(program)

    assert exc.value.diagnostics[0].code == "E_UNIT_003"
