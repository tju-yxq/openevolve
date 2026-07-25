import pytest


torch = pytest.importorskip("torch")
e3nn = pytest.importorskip("e3nn")
from e3nn import o3

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
    architecture_id,
)
from equivariant_nas.dsl.backends import E3NNGraphBackend


def test_compiled_tensor_product_is_rotation_equivariant():
    group = GroupSpec.o3()
    vector_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x1o", "O3"))
    output_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0e+1x1e+1x2e", "O3"))
    program = ArchitectureProgram(
        "1.0.0",
        "runtime_tensor_product",
        (InputPort("left", vector_type), InputPort("right", vector_type)),
        (
            Node(
                "tp",
                "core.tensor_product",
                {"left": ("input:left",), "right": ("input:right",)},
                {"out_irreps": str(output_type.irreps)},
            ),
        ),
        (OutputPort("out", "tp", output_type),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(registry).build(program, inference).double()
    left = torch.randn(8, vector_type.irreps.dimension, dtype=torch.float64)
    right = torch.randn(8, vector_type.irreps.dimension, dtype=torch.float64)
    reference = model({"left": left, "right": right}, {})["out"]
    rotation = o3.rand_matrix(dtype=torch.float64)
    vector_action = o3.Irreps(str(vector_type.irreps)).D_from_matrix(rotation)
    output_action = o3.Irreps(str(output_type.irreps)).D_from_matrix(rotation)
    rotated = model(
        {
            "left": left @ vector_action.transpose(0, 1),
            "right": right @ vector_action.transpose(0, 1),
        },
        {},
    )["out"]
    expected = reference @ output_action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1e-12)
    # e3nn 0.4.4 constructs random rotation matrices through a numerically
    # calibrated path whose float64 representation error is typically 1e-9.
    assert float(relative_error) < 1e-7


def test_strict_rewrites_preserve_identity_and_residual_runtime_values():
    group = GroupSpec.o3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0e+1x1o", "O3"))
    identity_program = ArchitectureProgram(
        "1.0.0",
        "rewrite_runtime",
        (InputPort("x", value_type),),
        (
            Node("identity_a", "core.identity", {"x": ("input:x",)}),
            Node("identity_b", "core.identity", {"x": ("identity_a",)}),
        ),
        (OutputPort("out", "identity_b", value_type),),
    )
    registry = core_registry()
    artifact = Compiler(registry).analyze(identity_program)
    assert artifact.expanded_program.nodes == ()
    identity_model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference).double()
    x = torch.randn(7, value_type.irreps.dimension, dtype=torch.float64)
    assert torch.equal(identity_model({"x": x}, {})["out"], x)

    left = ArchitectureProgram(
        "1.0.0",
        "rewrite_runtime",
        (InputPort("a", value_type), InputPort("b", value_type)),
        (Node("sum", "core.residual_add", {"left": ("input:a",), "right": ("input:b",)}),),
        (OutputPort("out", "sum", value_type),),
    )
    right = ArchitectureProgram(
        left.language_version,
        left.task_contract,
        left.inputs,
        (Node("sum", "core.residual_add", {"left": ("input:b",), "right": ("input:a",)}),),
        left.outputs,
    )
    left_artifact = Compiler(registry).analyze(left)
    right_artifact = Compiler(registry).analyze(right)
    assert architecture_id(left, registry) == architecture_id(right, registry)
    left_model = E3NNGraphBackend(registry).build(left_artifact.expanded_program, left_artifact.inference).double()
    right_model = E3NNGraphBackend(registry).build(right_artifact.expanded_program, right_artifact.inference).double()
    a = torch.randn_like(x)
    b = torch.randn_like(x)
    assert torch.equal(left_model({"a": a, "b": b}, {})["out"], right_model({"a": a, "b": b}, {})["out"])
