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
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import LoweringRule, LoweringRuleRegistry
from equivariant_nas.dsl.registry import PrimitiveDefinition, PrimitiveRegistry


def _split_type_rule(node, inputs, attrs):
    value = inputs["x"][0]
    return {"left": value, "right": value}, ()


def _multi_output_program():
    group = GroupSpec.so3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", "SO3"))
    return ArchitectureProgram(
        "1.0.0",
        "multi_output_runtime",
        (InputPort("x", value_type),),
        (
            Node(
                "split",
                "test.split@1",
                {"x": ("input:x",)},
                outputs=("left", "right"),
                declared_types={"left": value_type, "right": value_type},
            ),
        ),
        (
            OutputPort("left", "split:left", value_type),
            OutputPort("right", "split:right", value_type),
        ),
    )


def _registries(executor):
    primitives = PrimitiveRegistry()
    primitives.register(
        PrimitiveDefinition(
            "test.split",
            1,
            ("x",),
            ("left", "right"),
            _split_type_rule,
            group_families=("SO3",),
        )
    )
    lowering = LoweringRuleRegistry("multi_output_test")
    lowering.register(LoweringRule("test.split@1", executor))
    return primitives, lowering


def test_multi_output_lowering_binds_every_declared_port():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    def execute(context):
        value = context.resolved["x"][0]
        assert set(context.output_types) == {"left", "right"}
        return {"left": value + 1.0, "right": value - 1.0}

    primitives, lowering = _registries(execute)
    program = _multi_output_program()
    inference = TypeChecker(primitives).check(program)
    model = E3NNGraphBackend(primitives, lowering_rules=lowering).build(program, inference)
    value = torch.tensor([[2.0, 3.0]])
    outputs = model({"x": value}, {})

    assert torch.equal(outputs["left"], value + 1.0)
    assert torch.equal(outputs["right"], value - 1.0)


def test_multi_output_lowering_rejects_an_unkeyed_runtime_value():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn")

    primitives, lowering = _registries(lambda context: context.resolved["x"][0])
    program = _multi_output_program()
    inference = TypeChecker(primitives).check(program)
    model = E3NNGraphBackend(primitives, lowering_rules=lowering).build(program, inference)

    with pytest.raises(RuntimeError, match="must return a mapping"):
        model({"x": torch.zeros(1, 2)}, {})


def test_multi_output_nodes_require_explicit_port_references():
    primitives, _lowering = _registries(lambda context: {})
    program = _multi_output_program()
    invalid = ArchitectureProgram(
        program.language_version,
        program.task_contract,
        program.inputs,
        program.nodes,
        (OutputPort("invalid", "split", program.outputs[0].expected_type),),
    )

    with pytest.raises(Exception, match="unknown or forward value reference"):
        TypeChecker(primitives).check(invalid)
