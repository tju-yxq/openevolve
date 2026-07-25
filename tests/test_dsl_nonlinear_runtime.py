import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
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
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps


def _compile(program):
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    return E3NNGraphBackend(registry).build(program, inference).double()


def _action(irreps, rotation):
    return o3.Irreps(to_e3nn_irreps(irreps)).D_from_matrix(rotation)


def test_gate_scales_whole_irrep_blocks_and_is_rotation_equivariant():
    group = GroupSpec.so3()
    gates_type = EquivariantType(group, Carrier.NODE, Irreps.parse("5x0", "SO3"))
    values_type = EquivariantType(group, Carrier.NODE, Irreps.parse("3x1+2x2", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "gate_runtime",
        (InputPort("gates", gates_type), InputPort("values", values_type)),
        (Node("gate", "core.gate", {"gates": ("input:gates",), "value": ("input:values",)}),),
        (OutputPort("out", "gate", values_type),),
    )
    model = _compile(program).eval()
    gates = torch.sigmoid(torch.randn(7, gates_type.irreps.dimension, dtype=torch.float64))
    values = torch.randn(7, values_type.irreps.dimension, dtype=torch.float64)
    reference = model({"gates": gates, "values": values}, {})["out"]
    rotation = o3.rand_matrix(dtype=torch.float64)
    action = _action(values_type.irreps, rotation)
    rotated = model({"gates": gates, "values": values @ action.transpose(0, 1)}, {})["out"]
    assert torch.allclose(rotated, reference @ action.transpose(0, 1), atol=1e-8, rtol=1e-8)


def test_norm_activation_is_rotation_equivariant_with_mixed_degrees():
    group = GroupSpec.so3()
    value_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0+3x1+2x2", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "norm_activation_runtime",
        (InputPort("x", value_type),),
        (Node("activation", "core.norm_activation", {"x": ("input:x",)}, {"function": "silu"}),),
        (OutputPort("out", "activation", value_type),),
    )
    model = _compile(program).eval()
    values = torch.randn(9, value_type.irreps.dimension, dtype=torch.float64)
    reference = model({"x": values}, {})["out"]
    rotation = o3.rand_matrix(dtype=torch.float64)
    action = _action(value_type.irreps, rotation)
    rotated = model({"x": values @ action.transpose(0, 1)}, {})["out"]
    assert torch.allclose(rotated, reference @ action.transpose(0, 1), atol=1e-8, rtol=1e-8)


def test_segment_softmax_is_invariant_to_edge_order_within_segments():
    pytest.importorskip("torch_geometric")
    group = GroupSpec.so3()
    logits_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "segment_softmax_runtime",
        (InputPort("logits", logits_type),),
        (Node("weights", "core.segment_softmax", {"logits": ("input:logits",)}),),
        (OutputPort("out", "weights", logits_type),),
    )
    model = _compile(program).eval()
    logits = torch.randn(11, 1, dtype=torch.float64)
    edge_dst = torch.tensor([0, 1, 0, 2, 1, 2, 2, 0, 3, 3, 1], dtype=torch.long)
    reference = model({"logits": logits}, {"edge_dst": edge_dst})["out"]
    permutation = torch.tensor([8, 2, 10, 0, 6, 5, 1, 9, 4, 7, 3], dtype=torch.long)
    permuted = model(
        {"logits": logits.index_select(0, permutation)},
        {"edge_dst": edge_dst.index_select(0, permutation)},
    )["out"]
    assert torch.allclose(permuted, reference.index_select(0, permutation), atol=1e-12, rtol=1e-12)
    sums = torch.zeros(4, dtype=torch.float64)
    sums.index_add_(0, edge_dst, reference[:, 0])
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-12, rtol=1e-12)
