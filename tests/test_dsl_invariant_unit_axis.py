from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    DSLValidationError,
    InputPort,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend  # noqa: E402
from equivariant_nas.dsl.groups import GroupSpec  # noqa: E402
from equivariant_nas.dsl.irreps import Irreps  # noqa: E402
from equivariant_nas.dsl.types import (  # noqa: E402
    AxisSpec,
    Carrier,
    EquivarianceLevel,
    EquivariantTensorType,
    FeatureRole,
    Frame,
    InvariantTensorType,
    RepresentationLayout,
)


def _invariant_type(size=1, axis="energy_channel"):
    return InvariantTensorType(
        group=GroupSpec("SO3", 3),
        carrier=Carrier.GRAPH,
        irreps=Irreps.parse("{}x0".format(size), "SO3"),
        frame=Frame("invariant"),
        axes=(axis,),
        axis_specs=(
            AxisSpec(axis, size, FeatureRole.CHANNEL, "independent", 0),
        ),
        dtype="float32",
        measure="energy",
        level=EquivarianceLevel.CORE_CERTIFIED,
        feature_role=FeatureRole.CHANNEL,
    )


def _output_type(source):
    return replace(
        source,
        axes=(),
        axis_specs=(),
        layout=RepresentationLayout(storage="carrier_scalar"),
    )


def _program(input_type, *, axis="energy_channel"):
    return ArchitectureProgram(
        language_version="2.10.0",
        task_contract="unit_axis_layout_test",
        inputs=(InputPort("x", input_type),),
        nodes=(
            Node(
                "squeeze",
                "core.squeeze_unit_axis@1",
                {"x": ("input:x",)},
                {"axis": axis},
            ),
        ),
        outputs=(OutputPort("out", "squeeze", _output_type(input_type)),),
    )


def test_squeeze_unit_axis_has_typed_carrier_scalar_runtime_shape():
    registry = core_registry()
    program = _program(_invariant_type())
    inference = TypeChecker(registry).check(program)
    output_type = inference.value_types["squeeze"]
    assert output_type.axes == ()
    assert output_type.axis_specs == ()
    assert output_type.layout.storage == "carrier_scalar"

    model = E3NNGraphBackend(registry).build(program, inference)
    values = torch.tensor([[1.25], [-2.0], [3.5]], dtype=torch.float32)
    output = model({"x": values}, {})["out"]
    assert tuple(output.shape) == (3,)
    torch.testing.assert_close(output, values[:, 0], rtol=0.0, atol=0.0)


def test_squeeze_unit_axis_rejects_a_non_invariant_representation():
    input_type = EquivariantTensorType(
        group=GroupSpec("SO3", 3),
        carrier=Carrier.GRAPH,
        irreps=Irreps.parse("1x1", "SO3"),
        frame=Frame("global"),
        axes=("energy_channel",),
        axis_specs=(
            AxisSpec("energy_channel", 1, FeatureRole.CHANNEL, "independent", 0),
        ),
    )
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(_program(input_type))
    assert any(item.code == "E_UNIT_AXIS_001" for item in error.value.diagnostics)


def test_squeeze_unit_axis_rejects_a_nonunit_or_misnamed_axis():
    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(_program(_invariant_type(size=2)))
    assert any(item.code == "E_UNIT_AXIS_004" for item in error.value.diagnostics)

    with pytest.raises(DSLValidationError) as error:
        TypeChecker(core_registry()).check(
            _program(_invariant_type(), axis="wrong_axis")
        )
    assert any(item.code == "E_UNIT_AXIS_003" for item in error.value.diagnostics)


def test_squeeze_unit_axis_rejects_runtime_shape_that_breaks_the_type_contract():
    registry = core_registry()
    program = _program(_invariant_type())
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(registry).build(program, inference)
    with pytest.raises(RuntimeError, match=r"runtime shape \[carrier, 1\]"):
        model({"x": torch.zeros(3, 2)}, {})
