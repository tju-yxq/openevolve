import pytest


torch = pytest.importorskip("torch")
e3nn = pytest.importorskip("e3nn")
from e3nn import o3

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
