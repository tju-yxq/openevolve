import os

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
    expand_motifs,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import EquiformerV2GraphBackend, build_v2_so2_path
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps


V2_ROOT = os.environ.get("EQUIFORMER_V2_ROOT", "")
if not V2_ROOT:
    pytest.skip("EQUIFORMER_V2_ROOT is not configured", allow_module_level=True)


def test_official_v2_so2_path_is_rotation_equivariant():
    input_irreps = Irreps.parse("3x0+3x1+3x2", "SO3")
    output_irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    model = build_v2_so2_path(input_irreps, output_irreps, V2_ROOT).eval()
    features = torch.randn(12, input_irreps.dimension, dtype=torch.float32)
    edge_vectors = torch.randn(12, 3, dtype=torch.float32)
    torch.manual_seed(19)
    reference = model(features, edge_vectors)
    rotation = o3.rand_matrix(dtype=torch.float32)
    input_action = o3.Irreps(to_e3nn_irreps(input_irreps)).D_from_matrix(rotation)
    output_action = o3.Irreps(to_e3nn_irreps(output_irreps)).D_from_matrix(rotation)
    torch.manual_seed(19)
    rotated = model(features @ input_action.transpose(0, 1), edge_vectors @ rotation.transpose(0, 1))
    expected = reference @ output_action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1e-12)
    assert float(relative_error) < 1e-4


def test_official_v2_separable_s2_path_is_rotation_equivariant():
    irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    model = build_v2_so2_path(irreps, irreps, V2_ROOT, activation="separable_s2", grid_resolution=18).eval()
    features = torch.randn(6, irreps.dimension)
    edges = torch.randn(6, 3)
    torch.manual_seed(23)
    reference = model(features, edges)
    rotation = o3.rand_matrix(dtype=torch.float32)
    action = o3.Irreps(to_e3nn_irreps(irreps)).D_from_matrix(rotation)
    torch.manual_seed(23)
    rotated = model(features @ action.transpose(0, 1), edges @ rotation.transpose(0, 1))
    expected = reference @ action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1e-12)
    assert float(relative_error) < 5e-4


def test_complete_dsl_graph_lowers_v2_fusion_and_remains_rotation_equivariant():
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    value_type = EquivariantType(group, Carrier.NODE, irreps)
    source = ArchitectureProgram(
        "1.0.0",
        "v2_graph_runtime",
        (InputPort("x", value_type),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:x",)},
                {"hidden_irreps": str(irreps), "frame_id": "v2_edge"},
            ),
        ),
        (OutputPort("out", "v2", value_type),),
    )
    program = expand_motifs(source, reference_motif_registry())
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    model = EquiformerV2GraphBackend(registry, V2_ROOT).build(program, inference).eval()
    assert "equiformer-v2-so2-fusion" in model.backend_semantics_version

    features = torch.randn(5, irreps.dimension)
    edge_src = torch.tensor([0, 1, 2, 3, 4, 1, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 4, 0, 4, 0], dtype=torch.long)
    edges = torch.randn(edge_src.shape[0], 3)
    context = {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_vectors": edges,
        "num_nodes": features.shape[0],
    }
    reference = model({"x": features}, context)["out"]
    rotation = o3.rand_matrix(dtype=torch.float32)
    action = o3.Irreps(to_e3nn_irreps(irreps)).D_from_matrix(rotation)
    rotated = model(
        {"x": features @ action.transpose(0, 1)},
        dict(context, edge_vectors=edges @ rotation.transpose(0, 1)),
    )["out"]
    expected = reference @ action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1e-12)
    assert float(relative_error) < 8e-4

    loss = reference.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)
