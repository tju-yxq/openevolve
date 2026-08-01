import os
import sys
import types
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    Compiler,
    EquivariantType,
    Frame,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
    expand_motifs,
    import_equiformer_v1,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    E3NNGraphBackend,
    EquiformerV2GraphBackend,
    baseline_spec,
)
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps  # noqa: E402
from equivariant_nas.dsl.backends.qm9_model import build_qm9_dsl_model  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(
    os.environ.get(
        "EQUIFORMER_V2_ROOT",
        str(REPOSITORY_ROOT.parent / "equiformer_v3"),
    )
)


def _graph_context(num_nodes=5):
    edge_src = torch.tensor([0, 1, 2, 3, 4, 1, 3], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 4, 0, 4, 0], dtype=torch.long)
    return {
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "num_nodes": num_nodes,
        "batch": torch.zeros(num_nodes, dtype=torch.long),
        "num_graphs": 1,
    }


def _expanded_v2_program(channels=2):
    group = GroupSpec.so3()
    irreps = Irreps.parse(
        "{0}x0+{0}x1+{0}x2".format(channels),
        "SO3",
    )
    value_type = EquivariantType(group, Carrier.NODE, irreps)
    source = ArchitectureProgram(
        "1.0.0",
        "generic_v2_probe",
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
    return expand_motifs(source, reference_motif_registry()), irreps


def _require_v2_source():
    if not V2_ROOT.is_dir():
        pytest.skip("Equiformer V2 reference source is unavailable")
    return str(V2_ROOT)


def test_imported_v1_representation_flow_executes_by_generic_lowering():
    program = expand_motifs(
        import_equiformer_v1(baseline_spec()),
        reference_motif_registry(),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(registry)
    report = backend.support_report(program)

    assert report.supported
    assert not report.unsupported_nodes
    assert set(dict(report.node_exactness)) == {node.id for node in program.nodes}

    model = backend.build(program, inference).double().eval()
    context = _graph_context()
    edge_vectors = torch.randn(context["edge_src"].shape[0], 3, dtype=torch.float64)
    edge_irreps = program.inputs[1].value_type.irreps
    edge_sh = o3.spherical_harmonics(
        to_e3nn_irreps(edge_irreps),
        edge_vectors,
        normalize=True,
        normalization="component",
    )
    node_features = torch.randn(5, program.inputs[0].value_type.irreps.dimension, dtype=torch.float64)
    reference = model(
        {"node_features": node_features, "edge_sh": edge_sh},
        context,
    )["prediction"]

    rotation = o3.rand_matrix(dtype=torch.float64)
    rotated_edge_sh = o3.spherical_harmonics(
        to_e3nn_irreps(edge_irreps),
        edge_vectors @ rotation.transpose(0, 1),
        normalize=True,
        normalization="component",
    )
    rotated = model(
        {"node_features": node_features, "edge_sh": rotated_edge_sh},
        context,
    )["prediction"]
    assert torch.allclose(rotated, reference, atol=2.0e-7, rtol=2.0e-7)

    permutation = torch.tensor([2, 0, 4, 1, 3], dtype=torch.long)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.shape[0])
    permuted_context = dict(
        context,
        edge_src=inverse.index_select(0, context["edge_src"]),
        edge_dst=inverse.index_select(0, context["edge_dst"]),
        batch=context["batch"].index_select(0, permutation),
    )
    permuted = model(
        {
            "node_features": node_features.index_select(0, permutation),
            "edge_sh": edge_sh,
        },
        permuted_context,
    )["prediction"]
    assert torch.allclose(permuted, reference, atol=2.0e-10, rtol=2.0e-10)

    differentiable_features = node_features.clone().requires_grad_()
    output = model(
        {"node_features": differentiable_features, "edge_sh": edge_sh},
        context,
    )["prediction"]
    output.square().mean().backward()
    assert differentiable_features.grad is not None
    assert torch.isfinite(differentiable_features.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)


def test_v2_motif_executes_without_any_closed_subgraph_fusion():
    root = _require_v2_source()
    program, irreps = _expanded_v2_program()
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(registry, equiformer_v2_root=root)
    report = backend.support_report(program)

    assert report.supported
    assert not report.unsupported_nodes
    assert {node.op for node in program.nodes if "frame" in node.op or "s2" in node.op or "so2" in node.op} == {
        "core.to_edge_frame",
        "core.from_edge_frame",
        "core.so2_convolution",
        "core.separable_s2_activation",
    }

    model = backend.build(program, inference).eval()
    assert model.fused_subgraphs == ()
    assert model.lowering_rule_manifest["rule_count"] == 102

    features = torch.randn(5, irreps.dimension)
    context = _graph_context()
    edge_vectors = torch.randn(context["edge_src"].shape[0], 3)
    context = dict(context, edge_vectors=edge_vectors)
    torch.manual_seed(31)
    reference = model({"x": features}, context)["out"]

    rotation = o3.rand_matrix(dtype=torch.float32)
    action = o3.Irreps(to_e3nn_irreps(irreps)).D_from_matrix(rotation)
    torch.manual_seed(31)
    rotated = model(
        {"x": features @ action.transpose(0, 1)},
        dict(context, edge_vectors=edge_vectors @ rotation.transpose(0, 1)),
    )["out"]
    expected = reference @ action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1.0e-12)
    assert float(relative_error.detach()) < 8.0e-4

    differentiable_features = features.clone().requires_grad_()
    torch.manual_seed(37)
    output = model({"x": differentiable_features}, context)["out"]
    output.square().mean().backward()
    assert differentiable_features.grad is not None
    assert torch.isfinite(differentiable_features.grad).all()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)


def test_independent_edge_frame_round_trip_restores_value_and_gradient():
    root = _require_v2_source()
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    global_type = EquivariantType(group, Carrier.EDGE, irreps)
    program = ArchitectureProgram(
        "1.0.0",
        "edge_frame_round_trip",
        (InputPort("x", global_type),),
        (
            Node("to", "core.to_edge_frame", {"x": ("input:x",)}, {"frame_id": "bond"}),
            Node("back", "core.from_edge_frame", {"x": ("to",)}, {"frame_id": "bond"}),
        ),
        (OutputPort("out", "back", global_type),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(registry, equiformer_v2_root=root).build(program, inference)
    features = torch.randn(8, irreps.dimension, requires_grad=True)
    edge_vectors = torch.randn(8, 3)
    torch.manual_seed(41)
    output = model({"x": features}, {"edge_vectors": edge_vectors})["out"]

    assert torch.allclose(output, features, atol=2.0e-5, rtol=2.0e-5)
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


@pytest.mark.parametrize("separable", [False, True])
def test_s2_activations_have_independent_rotation_equivariant_lowering(separable):
    root = _require_v2_source()
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    value_type = EquivariantType(group, Carrier.EDGE, irreps, Frame("global"))
    inputs = [InputPort("x", value_type)]
    node_inputs = {"x": ("input:x",)}
    op = "core.s2_activation"
    scalar_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("2x0", "SO3"))
    if separable:
        inputs.append(InputPort("scalars", scalar_type))
        node_inputs["scalars"] = ("input:scalars",)
        op = "core.separable_s2_activation"
    program = ArchitectureProgram(
        "1.0.0",
        "independent_s2_activation",
        tuple(inputs),
        (Node("activation", op, node_inputs, {"grid_resolution": 18}),),
        (OutputPort("out", "activation", value_type),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    model = E3NNGraphBackend(registry, equiformer_v2_root=root).build(program, inference).eval()
    features = torch.randn(7, irreps.dimension, requires_grad=True)
    scalars = torch.randn(7, 2, requires_grad=True)
    values = {"x": features}
    if separable:
        values["scalars"] = scalars
    reference = model(values, {})["out"]

    rotation = o3.rand_matrix(dtype=torch.float32)
    action = o3.Irreps(to_e3nn_irreps(irreps)).D_from_matrix(rotation)
    rotated_values = {"x": features.detach() @ action.transpose(0, 1)}
    if separable:
        rotated_values["scalars"] = scalars.detach()
    rotated = model(rotated_values, {})["out"]
    expected = reference.detach() @ action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1.0e-12)
    assert float(relative_error.detach()) < 8.0e-4

    reference.square().mean().backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    if separable:
        assert scalars.grad is not None and bool(torch.isfinite(scalars.grad).all())


def test_v2_fusion_and_generic_lowering_match_forward_and_gradients():
    root = _require_v2_source()
    program, irreps = _expanded_v2_program()
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    torch.manual_seed(47)
    generic = E3NNGraphBackend(registry, equiformer_v2_root=root).build(program, inference).eval()
    torch.manual_seed(53)
    fused = EquiformerV2GraphBackend(registry, root).build(program, inference).eval()

    fused_path = fused.node_modules["v2__to_global"]
    fused_path.convolution.load_state_dict(generic.node_modules["v2__so2"].state_dict())
    fused_path.s2_activation.load_state_dict(generic.node_modules["v2__activation"].state_dict())
    fused.node_modules["v2__project"].load_state_dict(generic.node_modules["v2__project"].state_dict())

    context = _graph_context()
    context = dict(context, edge_vectors=torch.randn(context["edge_src"].shape[0], 3))
    features = torch.randn(5, irreps.dimension)
    torch.manual_seed(59)
    generic_output = generic({"x": features}, context)["out"]
    torch.manual_seed(59)
    fused_output = fused({"x": features}, context)["out"]
    assert torch.equal(generic_output, fused_output)

    generic_features = features.clone().requires_grad_()
    fused_features = features.clone().requires_grad_()
    torch.manual_seed(61)
    generic({"x": generic_features}, context)["out"].square().sum().backward()
    torch.manual_seed(61)
    fused({"x": fused_features}, context)["out"].square().sum().backward()
    assert torch.equal(generic_features.grad, fused_features.grad)
    assert torch.equal(
        generic.node_modules["v2__so2"].convolution.fc_m0.weight.grad,
        fused_path.convolution.convolution.fc_m0.weight.grad,
    )


def test_qm9_entry_can_explicitly_disable_v2_fusion(monkeypatch):
    root = _require_v2_source()
    group = GroupSpec.so3()
    irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
    node_type = EquivariantType(group, Carrier.NODE, irreps)
    node_scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", "SO3"))
    graph_scalar = EquivariantType(group, Carrier.GRAPH, Irreps.parse("1x0", "SO3"))
    program = ArchitectureProgram(
        "1.0.0",
        "qm9_alpha",
        (InputPort("node_features", node_type),),
        (
            Node(
                "v2",
                "motif.v2_so2_residual_message",
                {"x": ("input:node_features",)},
                {"hidden_irreps": str(irreps), "frame_id": "v2_edge"},
            ),
            Node(
                "readout",
                "core.select_scalars",
                {"x": ("v2",)},
                {"multiplicity": 1},
                declared_types={"out": node_scalar},
            ),
            Node(
                "pool",
                "core.global_pool",
                {"x": ("readout",)},
                declared_types={"out": graph_scalar},
            ),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    torch_cluster = types.ModuleType("torch_cluster")

    def radius_graph(pos, r, batch, max_num_neighbors):
        del r, batch, max_num_neighbors
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=pos.device)

    torch_cluster.radius_graph = radius_graph
    monkeypatch.setitem(sys.modules, "torch_cluster", torch_cluster)
    compiler = Compiler(core_registry(), reference_motif_registry())
    model = build_qm9_dsl_model(
        program,
        compiler,
        equiformer_v2_root=root,
        prefer_v2_fusion=False,
        allow_experimental_generic_lowering=True,
    ).eval()

    assert model.graph_model.fused_subgraphs == ()
    assert model.generic_lowering_admission["v2_fusion_preferred"] is False
    features = torch.randn(2, irreps.dimension, requires_grad=True)
    positions = torch.randn(2, 3)
    batch = torch.zeros(2, dtype=torch.long)
    torch.manual_seed(67)
    prediction = model(features, positions, batch)
    prediction.square().sum().backward()
    assert prediction.shape == (1, 1)
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
