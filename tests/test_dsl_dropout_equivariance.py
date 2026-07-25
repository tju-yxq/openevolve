import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from e3nn import o3

from equivariant_nas.dsl import ArchitectureProgram, Carrier, EquivariantType, GroupSpec, InputPort, Irreps, Node, OutputPort, TypeChecker, core_registry
from equivariant_nas.dsl.backends import E3NNGraphBackend
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps


@pytest.mark.parametrize("op", ["core.invariant_dropout", "core.stochastic_depth"])
def test_dropout_uses_rotation_invariant_block_masks(op):
    group = GroupSpec.so3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0+3x1+2x2", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "dropout",
        (InputPort("x", value_type),),
        (Node("drop", op, {"x": ("input:x",)}, {"p": 0.5}),),
        (OutputPort("out", "drop", value_type),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(registry).build(program, inference).double().train()
    inputs = torch.randn(5, value_type.irreps.dimension, dtype=torch.float64)
    rotation = o3.rand_matrix(dtype=torch.float64)
    action = o3.Irreps(to_e3nn_irreps(value_type.irreps)).D_from_matrix(rotation)
    torch.manual_seed(7)
    reference = model({"x": inputs}, {})["out"]
    torch.manual_seed(7)
    rotated = model({"x": inputs @ action.transpose(0, 1)}, {})["out"]
    expected = reference @ action.transpose(0, 1)
    assert torch.allclose(rotated, expected, atol=1e-9, rtol=1e-9)
