import pytest
import torch

from equivariant_nas.dsl import (
    ArchitectureProgram,
    Carrier,
    Compiler,
    DSLValidationError,
    EquivariantTensorType,
    EquivariantType,
    GraphTopologyType,
    GroupSpec,
    IndexMapType,
    InputPort,
    Irreps,
    Node,
    OutputPort,
    core_registry,
    migrate_message_flow_to_explicit_topology,
    reference_motif_registry,
    value_type_from_dict,
)
from equivariant_nas.dsl.backends.e3nn_backend import E3NNGraphBackend
from equivariant_nas.dsl.backends.lowering import RuntimeValueKind
from equivariant_nas.dsl.motifs import expand_motifs
from equivariant_nas.dsl.inference import TypeChecker


def _types(target_size=4):
    group = GroupSpec.so3()
    node = EquivariantTensorType(group, Carrier.NODE, Irreps.parse("2x0", group.family))
    edge = node.with_carrier(Carrier.EDGE)
    source = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source", target_size=3)
    target = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "target", target_size=3)
    segment = IndexMapType(group, Carrier.EDGE, Carrier.NODE, "segment", target_size=target_size)
    batch = IndexMapType(group, Carrier.NODE, Carrier.GRAPH, "batch", target_size=1)
    topology = GraphTopologyType(group, source, target, batch, edge_order="stable")
    return node, edge, source, target, segment, topology


def _program(reduce="sum"):
    node, edge, source, target, segment, _ = _types()
    return ArchitectureProgram(
        "2.0.0",
        "explicit-topology",
        (
            InputPort("x", node),
            InputPort("source_index", source),
            InputPort("target_index", target),
            InputPort("segment_index", segment),
        ),
        (
            Node("source", "core.endpoint_gather", {"x": ("input:x",), "index": ("input:source_index",)}),
            Node("target", "core.endpoint_gather", {"x": ("input:x",), "index": ("input:target_index",)}),
            Node("message", "core.residual_add", {"left": ("source",), "right": ("target",)}),
            Node(
                "aggregate",
                "core.segment_reduce",
                {"x": ("message",), "index": ("input:segment_index",)},
                {"reduce": reduce, "normalization": "none"},
            ),
        ),
        (OutputPort("prediction", "aggregate", node),),
    )


def test_index_map_and_topology_types_roundtrip_with_direction_and_sizes():
    _node, _edge, source, target, _segment, topology = _types()
    assert value_type_from_dict(source.to_dict()) == source
    assert value_type_from_dict(topology.to_dict()) == topology
    assert topology.source_index.endpoint == "source"
    assert topology.target_index.endpoint == "target"
    assert topology.batch_index.target_size == 1


def test_explicit_gather_and_reduce_typecheck_and_lower_without_graph_context_side_channels():
    program = _program("sum")
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    backend = E3NNGraphBackend(registry)
    report = backend.support_report(artifact.expanded_program)

    assert report.supported
    runtime_kinds = dict(report.runtime_kinds)
    assert runtime_kinds["input:source_index"] == RuntimeValueKind.INDEX_MAP
    assert runtime_kinds["input:segment_index"] == RuntimeValueKind.INDEX_MAP
    assert runtime_kinds["aggregate"] == RuntimeValueKind.DENSE_TENSOR

    model = backend.build(artifact.expanded_program, artifact.inference)
    x = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    source = torch.tensor([0, 1, 2], dtype=torch.long)
    target = torch.tensor([1, 1, 3], dtype=torch.long)
    output = model(
        {
            "x": x,
            "source_index": {"indices": source, "target_size": 3},
            "target_index": {"indices": target, "target_size": 3},
            "segment_index": {"indices": target, "target_size": 4},
        },
        {},
    )["prediction"]
    messages = x.index_select(0, source) + x.index_select(0, target)
    expected = torch.zeros(4, 2).index_add_(0, target, messages)
    assert torch.equal(output, expected)
    assert torch.equal(output[0], torch.zeros(2))
    assert torch.equal(output[2], torch.zeros(2))


def test_explicit_mean_reduce_preserves_empty_targets_as_zero_rows():
    program = _program("mean")
    registry = core_registry()
    artifact = Compiler(registry).analyze(program)
    model = E3NNGraphBackend(registry).build(artifact.expanded_program, artifact.inference)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    source = torch.tensor([0, 1, 2], dtype=torch.long)
    target = torch.tensor([1, 1, 3], dtype=torch.long)
    output = model(
        {
            "x": x,
            "source_index": {"indices": source, "target_size": 3},
            "target_index": {"indices": target, "target_size": 3},
            "segment_index": {"indices": target, "target_size": 4},
        },
        {},
    )["prediction"]
    assert torch.equal(output[0], torch.zeros(2))
    assert torch.equal(output[2], torch.zeros(2))
    assert torch.allclose(output[1], torch.tensor([5.0, 7.0]))
    assert torch.allclose(output[3], torch.tensor([12.0, 14.0]))


def test_v2_explicit_topology_primitives_reject_legacy_or_wrong_direction_types():
    group = GroupSpec.so3()
    legacy = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", group.family))
    source = IndexMapType(group, Carrier.NODE, Carrier.EDGE, "source")
    program = ArchitectureProgram(
        "2.0.0",
        "negative",
        (InputPort("x", legacy), InputPort("index", source)),
        (Node("gather", "core.endpoint_gather", {"x": ("input:x",), "index": ("input:index",)}),),
        (OutputPort("prediction", "gather", legacy.with_carrier(Carrier.EDGE)),),
    )
    with pytest.raises(DSLValidationError) as error:
        Compiler(core_registry()).analyze(program)
    assert any(item.code == "E_TYPE_010" for item in error.value.diagnostics)

    node, _edge, _source, _target, segment, _topology = _types()
    wrong = ArchitectureProgram(
        "2.0.0",
        "wrong-direction",
        (InputPort("x", node), InputPort("index", segment)),
        (Node("gather", "core.endpoint_gather", {"x": ("input:x",), "index": ("input:index",)}),),
        (OutputPort("prediction", "gather", node.with_carrier(Carrier.EDGE)),),
    )
    with pytest.raises(DSLValidationError) as wrong_error:
        Compiler(core_registry()).analyze(wrong)
    assert any(item.code in ("E_INDEX_006", "E_INDEX_007") for item in wrong_error.value.diagnostics)


def test_legacy_message_flow_migrates_to_explicit_indices_with_identical_forward():
    group = GroupSpec.so3()
    node_type = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0", group.family))
    edge_sh_type = EquivariantType(group, Carrier.EDGE, Irreps.parse("1x0+1x1", group.family))
    hidden = EquivariantType(group, Carrier.NODE, Irreps.parse("2x0+2x1", group.family))
    scalar = EquivariantType(group, Carrier.NODE, Irreps.parse("1x0", group.family))
    graph_scalar = scalar.with_carrier(Carrier.GRAPH)
    source = ArchitectureProgram(
        "1.0.0",
        "message-migration",
        (InputPort("x", node_type), InputPort("edge_sh", edge_sh_type)),
        (
            Node(
                "message",
                "motif.v1_initial_message",
                {"x": ("input:x",), "edge_sh": ("input:edge_sh",)},
                {"hidden_irreps": str(hidden.irreps)},
            ),
            Node("scalar", "core.select_scalars", {"x": ("message",)}, {"multiplicity": 1}),
            Node("pool", "core.global_pool", {"x": ("scalar",)}),
        ),
        (OutputPort("prediction", "pool", graph_scalar),),
    )
    primitives = core_registry()
    motifs = reference_motif_registry()
    old_program = expand_motifs(source, motifs)
    migrated = migrate_message_flow_to_explicit_topology(source, primitives, motifs)
    new_program = migrated.program

    assert set(migrated.manifest.added_inputs) == {"source_index", "edge_to_node_index", "batch_index"}
    assert {item[1] for item in migrated.manifest.replaced_nodes} == {
        "core.edge_lift@1",
        "core.segment_sum@1",
        "core.global_pool@1",
    }
    assert all(node.op not in ("core.edge_lift@1", "core.segment_sum@1", "core.global_pool@1") for node in new_program.nodes)

    backend = E3NNGraphBackend(primitives)
    old_model = backend.build(old_program, TypeChecker(primitives).check(old_program)).eval()
    new_model = backend.build(new_program, TypeChecker(primitives).check(new_program)).eval()
    new_model.load_state_dict(old_model.state_dict())

    x = torch.randn(4, node_type.irreps.dimension)
    edge_src = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)
    edge_dst = torch.tensor([1, 2, 3, 0, 2], dtype=torch.long)
    edge_sh = torch.randn(edge_src.shape[0], edge_sh_type.irreps.dimension)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    old_output = old_model(
        {"x": x, "edge_sh": edge_sh},
        {
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "num_nodes": 4,
            "batch": batch,
            "num_graphs": 2,
        },
    )["prediction"]
    new_output = new_model(
        {
            "x": x,
            "edge_sh": edge_sh,
            "source_index": {"indices": edge_src, "target_size": edge_src.numel()},
            "edge_to_node_index": {"indices": edge_dst, "target_size": 4},
            "batch_index": {"indices": batch, "target_size": 2},
        },
        {},
    )["prediction"]
    assert torch.allclose(new_output, old_output, atol=1e-7, rtol=1e-7)


def test_explicit_topology_evidence_exports_forward_and_gradient_alignment(tmp_path):
    from scripts.export_dsl_explicit_topology_evidence import build_evidence

    evidence = build_evidence(tmp_path, pytest_summary="test")
    assert evidence["forward_max_abs_error"] <= 1.0e-7
    assert max(evidence["input_gradient_max_abs_error"].values()) <= 1.0e-7
    assert evidence["parameter_gradient_max_abs_error"] <= 1.0e-7
    assert evidence["explicit_required_context"] == []
    assert not evidence["claims"]["official_equiformer_v1_reproduced"]
    assert (tmp_path / "summary.json").is_file()
