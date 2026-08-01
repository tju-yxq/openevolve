import os
from dataclasses import replace
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from e3nn import o3  # noqa: E402

from equivariant_nas.dsl import (  # noqa: E402
    ArchitectureProgram,
    Carrier,
    DSLValidationError,
    EquivariantType,
    GroupSpec,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    TypeChecker,
    core_registry,
    expand_motifs,
    import_equiformer_v1,
    import_equiformer_v3,
    reference_motif_registry,
)
from equivariant_nas.dsl.backends import (  # noqa: E402
    E3NNGraphBackend,
    EquiformerV3Spec,
    V3_REFERENCE_COMMIT,
    baseline_spec,
    equiformer_v3_source_available,
    resolve_equiformer_v3_package_path,
)
from equivariant_nas.dsl.backends.e3nn_backend import to_e3nn_irreps  # noqa: E402


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
V3_ROOT = Path(
    os.environ.get(
        "EQUIFORMER_V3_ROOT",
        str(REPOSITORY_ROOT.parent / "equiformer_v3_official"),
    )
)


def _require_v3_source():
    if not equiformer_v3_source_available(str(V3_ROOT)):
        pytest.skip("official Equiformer V3 operator source is unavailable")
    return str(V3_ROOT)


def _small_v3_spec():
    return EquiformerV3Spec(
        num_layers=1,
        num_channels=2,
        attn_hidden_channels=2,
        ffn_hidden_channels=3,
        num_heads=1,
        lmax=2,
        mmax=1,
        attn_grid_resolution=(8, 6),
        ffn_grid_resolution=(8, 8),
        regress_forces=False,
        regress_stress=False,
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


def test_official_v3_source_identity_and_import_scope_are_explicit():
    root = _require_v3_source()
    package = resolve_equiformer_v3_package_path(root)
    assert package is not None
    assert package.name == "equiformer_v3"
    assert (package / "Jd.pt").is_file()

    program = import_equiformer_v3(_small_v3_spec())
    assert program.annotations["official_source_commit"] == V3_REFERENCE_COMMIT
    assert program.annotations["constructor_bypass"] is False
    assert program.annotations["reference_backend"] == "equiformer_v3_compositional"
    assert "not official constructor" in program.annotations["numerical_scope"]


def test_v3_program_expands_to_core_primitives_and_has_generic_support():
    root = _require_v3_source()
    program = import_equiformer_v3(_small_v3_spec())
    expanded = expand_motifs(program, reference_motif_registry())
    registry = core_registry()
    inference = TypeChecker(registry).check(expanded)
    backend = E3NNGraphBackend(
        registry,
        equiformer_v2_root=root,
        equiformer_v3_root=root,
    )
    report = backend.support_report(expanded)

    assert report.supported
    assert len(registry.names()) == 103
    assert all(node.op.startswith("core.") for node in expanded.nodes)
    assert {"core.s2_swiglu", "core.equivariant_merge_norm"}.issubset(
        {node.op for node in expanded.nodes}
    )
    assert len(inference.node_order) == len(expanded.nodes)
    assert not inference.open_obligations


def test_s2_swiglu_contract_rejects_unpaired_channels():
    group = GroupSpec.so3()
    input_type = EquivariantType(
        group,
        Carrier.NODE,
        Irreps.parse("3x0+3x1+3x2", "SO3"),
    )
    output_type = EquivariantType(
        group,
        Carrier.NODE,
        Irreps.parse("2x0+2x1+2x2", "SO3"),
    )
    program = ArchitectureProgram(
        "1.0.0",
        "v3_negative_contract",
        (InputPort("x", input_type),),
        (
            Node(
                "bad",
                "core.s2_swiglu",
                {"x": ("input:x",)},
                {"out_irreps": "2x0+2x1+2x2", "grid_resolution": [8, 8]},
            ),
        ),
        (OutputPort("out", "bad", output_type),),
    )
    with pytest.raises(DSLValidationError) as caught:
        TypeChecker(core_registry()).check(program)
    assert any(item.code == "E_V3_TYPE_005" for item in caught.value.diagnostics)


def test_merge_norm_contract_rejects_nonuniform_dense_layout():
    group = GroupSpec.so3()
    value_type = EquivariantType(
        group,
        Carrier.NODE,
        Irreps.parse("2x0+3x1+2x2", "SO3"),
    )
    program = ArchitectureProgram(
        "1.0.0",
        "v3_negative_norm",
        (InputPort("x", value_type),),
        (Node("bad", "core.equivariant_merge_norm", {"x": ("input:x",)}),),
        (OutputPort("out", "bad", value_type),),
    )
    with pytest.raises(DSLValidationError) as caught:
        TypeChecker(core_registry()).check(program)
    assert any(item.code == "E_V3_TYPE_003" for item in caught.value.diagnostics)


@pytest.mark.parametrize("operator", ["core.s2_swiglu", "core.equivariant_merge_norm"])
def test_v3_primitive_lowerings_are_rotation_equivariant_and_differentiable(operator):
    root = _require_v3_source()
    group = GroupSpec.so3()
    if operator == "core.s2_swiglu":
        input_irreps = Irreps.parse("4x0+4x1+4x2", "SO3")
        output_irreps = Irreps.parse("2x0+2x1+2x2", "SO3")
        attrs = {"out_irreps": str(output_irreps), "mmax": 2, "grid_resolution": [18, 18]}
    else:
        input_irreps = Irreps.parse("3x0+3x1+3x2", "SO3")
        output_irreps = input_irreps
        attrs = {"normalization": "component", "centering": True}
    input_type = EquivariantType(group, Carrier.NODE, input_irreps)
    output_type = EquivariantType(group, Carrier.NODE, output_irreps)
    program = ArchitectureProgram(
        "1.0.0",
        "v3_primitive_runtime",
        (InputPort("x", input_type),),
        (Node("op", operator, {"x": ("input:x",)}, attrs),),
        (OutputPort("out", "op", output_type),),
    )
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(registry, equiformer_v3_root=root)
    assert backend.support_report(program).supported
    model = backend.build(program, inference).double().eval()
    torch.manual_seed(123)
    features = torch.randn(6, input_irreps.dimension, dtype=torch.float64, requires_grad=True)
    reference = model({"x": features}, {})["out"]

    rotation = o3.rand_matrix(dtype=torch.float64)
    input_action = o3.Irreps(to_e3nn_irreps(input_irreps)).D_from_matrix(rotation)
    output_action = o3.Irreps(to_e3nn_irreps(output_irreps)).D_from_matrix(rotation)
    rotated = model({"x": features.detach() @ input_action.transpose(0, 1)}, {})["out"]
    expected = reference.detach() @ output_action.transpose(0, 1)
    relative_error = (rotated - expected).norm() / expected.norm().clamp_min(1.0e-12)
    assert float(relative_error.detach()) < 2.0e-5

    reference.square().mean().backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    if operator == "core.s2_swiglu":
        assert gradients == []
    else:
        assert gradients and all(value is not None and bool(torch.isfinite(value).all()) for value in gradients)


def test_small_v3_program_executes_without_constructor_bypass():
    root = _require_v3_source()
    program = expand_motifs(import_equiformer_v3(_small_v3_spec()), reference_motif_registry())
    registry = core_registry()
    inference = TypeChecker(registry).check(program)
    backend = E3NNGraphBackend(
        registry,
        equiformer_v2_root=root,
        equiformer_v3_root=root,
    )
    model = backend.build(program, inference).eval()
    assert model.fused_subgraphs == ()
    assert model.backend_semantics_version == "e3nn-graph-lowering-registry-v24"
    assert model.lowering_rule_manifest["rule_count"] == 103

    context = _graph_context()
    edge_vectors = torch.randn(context["edge_src"].shape[0], 3)
    context = dict(context, edge_vectors=edge_vectors)
    edge_irreps = program.inputs[1].value_type.irreps
    edge_sh = o3.spherical_harmonics(
        to_e3nn_irreps(edge_irreps),
        edge_vectors,
        normalize=True,
        normalization="component",
    )
    node_features = torch.randn(5, program.inputs[0].value_type.irreps.dimension, requires_grad=True)
    torch.manual_seed(101)
    output = model({"node_features": node_features, "edge_sh": edge_sh}, context)["energy"]
    assert output.shape == (1, 1)
    output.square().sum().backward()
    assert node_features.grad is not None and bool(torch.isfinite(node_features.grad).all())


def test_f63_readout_subgraph_is_fully_core_lowerable_but_not_official_v1_exact():
    parent = import_equiformer_v1(baseline_spec())
    graph_scalar = parent.outputs[0].expected_type
    child_nodes = tuple(
        Node(
            "graph_pool",
            "motif.v1_multilevel_readout",
            {"terminal": ("scalar_readout",), "aux": ("block2",)},
            declared_types={"out": graph_scalar},
        )
        if node.id == "graph_pool"
        else node
        for node in parent.nodes
    )
    child = replace(
        parent,
        nodes=child_nodes,
        program_id=parent.program_id + "_f63_core_probe",
        annotations=dict(
            parent.annotations,
            f63_scope="readout subgraph core-lowered; official V1 backbone remains representation-only",
        ),
    )
    expanded = expand_motifs(child, reference_motif_registry())
    registry = core_registry()
    TypeChecker(registry).check(expanded)
    report = E3NNGraphBackend(registry).support_report(expanded)

    f63_nodes = [node for node in expanded.nodes if node.id.startswith("graph_pool__")]
    assert report.supported
    assert [node.op for node in f63_nodes] == [
        "core.global_pool",
        "core.select_scalars",
        "core.global_pool",
        "core.irrep_concat",
        "core.irrep_linear",
    ]
    assert "constructor-locked" in parent.annotations["representation_scope"]
